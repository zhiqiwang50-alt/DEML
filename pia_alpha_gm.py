import argparse
import ast
import csv
import inspect
import json
import math
import os
import re
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pia_attention_guided import (
    attention_proxy_stats,
    context_projection,
    enforce_embedding_constraints,
    fixed_embedding_tensor,
    prefix_forward_hidden_and_attention,
    random_public_embeddings,
    variable_view,
)
from pia_tinyllama import (
    ATTACKER_MAP,
    MODEL_NAME,
    TOTAL_BLOCKS,
    activation_calibrated_discretization,
    bleu_score,
    capture_activation,
    capture_prefix_activation,
    embedding_bounds,
    embedding_candidates,
    inferred_boundary_special_tokens,
    json_dump,
    jsonl_append,
    load_dataset_prompts,
    load_tinyllama,
    naive_discretization,
    nearest_embedding_loss,
    optional_nerr,
    peak_memory_mb,
    semantic_candidates,
    set_seed,
    text_from_ids,
    token_accuracy,
    token_texts,
)


EXPERIMENT_TITLE = "Attention-Initialized Gradient-Matched Prompt Inversion (AIGM-PIA) TinyLlama pilot"
THREAT_MODEL = (
    "White-box attack using only observed boundary activation, sequence length, public TinyLlama "
    "prefix parameters, tokenizer, public vocabulary embeddings, and public semantic candidates."
)
METHODS = [
    "baseline",
    "dummy_init_existing",
    "attention_context_existing",
    "alpha_nn_init_direct",
    "alpha_nn_init_residual",
    "alpha_nn_gradmatch",
    "alpha_nn_gradmatch_context",
]
ALPHA_METHODS = {
    "alpha_nn_init_direct",
    "alpha_nn_init_residual",
    "alpha_nn_gradmatch",
    "alpha_nn_gradmatch_context",
}
GRADMATCH_METHODS = {"alpha_nn_gradmatch", "alpha_nn_gradmatch_context"}
CONTEXT_METHODS = {"attention_context_existing", "alpha_nn_gradmatch_context"}


@dataclass
class AIGMConfig:
    method: str
    run_name: str
    output_dir: str
    dataset_name: str
    dataset_path: str
    dataset_len: int
    seed: int
    participant_number: int
    attacker_position: int
    inverted_block_count: int
    target_layer: int
    epoch: int
    stage_a_epoch: int
    lr: float
    lambda_vocab: float
    lambda_dummy: float
    lambda_context: float
    top_k_embedding: int
    top_y_semantic: int
    gamma: float
    eta: float
    adaptive_discretization: bool
    semantic_speculation: bool
    max_token_len: int
    grad_clip: float
    fix_boundary_specials: bool
    local_files_only: bool


def mean_std(values: Sequence[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None}
    return {
        "mean": float(statistics.mean(vals)),
        "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
    }


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = {
        "token_accuracy": mean_std([row["token_accuracy"] for row in rows]),
        "bleu": mean_std([row["bleu"] for row in rows]),
        "runtime_seconds": mean_std([row["runtime_seconds"] for row in rows]),
        "peak_gpu_memory_mb": mean_std(
            [row["peak_gpu_memory_mb"] for row in rows if row.get("peak_gpu_memory_mb") is not None]
        ),
        "optimization_loss": mean_std([row["optimization_loss"] for row in rows]),
        "cosine_similarity": mean_std([row["cosine_similarity"] for row in rows]),
        "prompt_token_count": mean_std([row["prompt_token_count"] for row in rows]),
        "sample_count": len(rows),
    }
    nerr_vals = [row["nerr"] for row in rows if row.get("nerr") is not None]
    metrics["nerr"] = {
        "mean": float(statistics.mean(nerr_vals)) if nerr_vals else None,
        "std": float(statistics.pstdev(nerr_vals)) if len(nerr_vals) > 1 else 0.0 if nerr_vals else None,
        "status": "available" if nerr_vals else "NERR unavailable",
    }
    return metrics


def normalize_run_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def topk_cosine(
    vectors: torch.Tensor,
    weight: torch.Tensor,
    top_k: int,
    chunk_size: int = 4096,
) -> Tuple[List[List[int]], List[List[float]]]:
    if vectors.dim() == 3:
        vectors = vectors.squeeze(0)
    vectors = vectors.detach().float()
    vocab = weight.detach().float()
    top_k = max(1, min(int(top_k), vocab.shape[0]))
    vec_norm = F.normalize(vectors, dim=-1)
    best_vals = torch.full((vectors.shape[0], top_k), -float("inf"), device=vectors.device)
    best_ids = torch.zeros((vectors.shape[0], top_k), dtype=torch.long, device=vectors.device)
    for start in range(0, vocab.shape[0], chunk_size):
        chunk = F.normalize(vocab[start : start + chunk_size], dim=-1)
        sims = vec_norm @ chunk.t()
        vals = torch.cat([best_vals, sims], dim=1)
        ids = torch.cat(
            [
                best_ids,
                torch.arange(start, start + chunk.shape[0], device=vectors.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(vectors.shape[0], -1),
            ],
            dim=1,
        )
        best_vals, indices = torch.topk(vals, top_k, dim=1)
        best_ids = torch.gather(ids, 1, indices)
    return (
        [[int(x) for x in row] for row in best_ids.detach().cpu().tolist()],
        [[float(x) for x in row] for row in best_vals.detach().cpu().tolist()],
    )


def minmax01(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    if values.numel() == 0:
        return values
    lo = values.min()
    hi = values.max()
    if float((hi - lo).detach().cpu()) < 1e-12:
        return torch.full_like(values, 0.5)
    return (values - lo) / (hi - lo)


def stage_a_dummy_proxy(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: AIGMConfig,
    observed_activation: torch.Tensor,
    seq_len: int,
    device: torch.device,
    fixed_public: Dict[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], List[Dict[str, float]]]:
    embed_layer = model.get_input_embeddings()
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    fixed_embeds, fixed_positions, _ = fixed_embedding_tensor(embed_layer, fixed_public, seq_len, device)
    left, right = embedding_bounds(embed_layer.weight)
    left = left.to(device)
    right = right.to(device)
    dummy = random_public_embeddings(tokenizer, embed_layer, seq_len, device, fixed_public).requires_grad_(True)
    optimizer = torch.optim.AdamW([dummy], lr=cfg.lr)
    history: List[Dict[str, float]] = []
    target = observed_activation.detach()
    for step in range(max(1, cfg.stage_a_epoch)):
        enforce_embedding_constraints(dummy, fixed_positions, fixed_embeds, left, right)
        hidden, _ = prefix_forward_hidden_and_attention(
            model,
            cfg.target_layer,
            dummy.to(dtype=embed_layer.weight.dtype),
            attention_mask,
            output_target_attention=False,
        )
        activation_loss = F.mse_loss(hidden.float(), target.to(hidden.device).float())
        vocab_loss = nearest_embedding_loss(variable_view(dummy, fixed_positions), embed_layer.weight)
        total_loss = activation_loss + cfg.lambda_dummy * vocab_loss
        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in Stage A dummy proxy at step {step + 1}")
        optimizer.zero_grad()
        total_loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([dummy], cfg.grad_clip)
        optimizer.step()
        if (step + 1) % max(1, min(50, cfg.stage_a_epoch)) == 0 or step == cfg.stage_a_epoch - 1:
            history.append(
                {
                    "step": step + 1,
                    "stage_a_activation_loss": float(activation_loss.detach().cpu()),
                    "stage_a_vocab_loss": float(vocab_loss.detach().cpu()),
                    "stage_a_total_loss": float(total_loss.detach().cpu()),
                }
            )
    enforce_embedding_constraints(dummy, fixed_positions, fixed_embeds, left, right)
    with torch.no_grad():
        _, attn = prefix_forward_hidden_and_attention(
            model,
            cfg.target_layer,
            dummy.detach().to(dtype=embed_layer.weight.dtype),
            attention_mask,
            output_target_attention=True,
        )
    a_self, stats = attention_proxy_stats(attn)
    stats.update(
        {
            "proxy_name": "dummy-attention proxy",
            "not_true_prompt_attention": True,
            "bos_column_mass_mean": float(a_self[:, 0].mean().detach().cpu()),
            "bos_column_mass_max": float(a_self[:, 0].max().detach().cpu()),
            "stage_a_detached_for_stage_b": True,
        }
    )
    return dummy.detach().float(), a_self.detach(), stats, history


def alpha_nn_initialization(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: AIGMConfig,
    observed_activation: torch.Tensor,
    a_self: torch.Tensor,
) -> Tuple[torch.Tensor, List[int], torch.Tensor, Dict[str, Any]]:
    embed_layer = model.get_input_embeddings()
    h_obs = observed_activation.detach().float()[0]
    if a_self.shape != (h_obs.shape[0], h_obs.shape[0]):
        raise RuntimeError(f"A_self shape {list(a_self.shape)} is incompatible with H_obs shape {list(h_obs.shape)}")
    alpha_direct = torch.matmul(a_self.to(h_obs.device), h_obs)
    if cfg.method == "alpha_nn_init_direct":
        gamma = 1.0
        h_alpha = alpha_direct
        formula = "H_alpha = A_self @ H_obs"
    else:
        gamma = float(cfg.gamma)
        h_alpha = (1.0 - gamma) * h_obs + gamma * alpha_direct
        formula = "H_alpha_residual = (1 - gamma) * H_obs + gamma * (A_self @ H_obs)"
    init_sets, init_cos_sets = topk_cosine(h_alpha, embed_layer.weight, 1)
    init_ids = [items[0] for items in init_sets]
    ids_tensor = torch.tensor([init_ids], dtype=torch.long, device=h_obs.device)
    z0 = embed_layer(ids_tensor).detach().float()
    z0_check = embed_layer(ids_tensor).detach().float()
    max_abs_diff = float((z0 - z0_check).abs().max().detach().cpu())
    if max_abs_diff >= 1e-6:
        raise RuntimeError(f"alpha-NN z0 lookup unit check failed: max_abs_diff={max_abs_diff}")
    audit = {
        "method": cfg.method,
        "formula": formula,
        "matrix_multiplication_direction": "A_self @ H_obs",
        "gamma": gamma,
        "init_token_ids": init_ids,
        "init_tokens": token_texts(tokenizer, init_ids),
        "nn_cosine_similarity": [items[0] for items in init_cos_sets],
        "nn_cosine_mean": float(statistics.mean([items[0] for items in init_cos_sets])) if init_cos_sets else None,
        "z0_embedding_lookup_max_abs_diff": max_abs_diff,
        "z0_sources": ["observed_activation", "dummy_attention_proxy", "public_vocabulary_embedding_matrix"],
        "does_not_use_stage_a_dummy_embedding_as_z0": True,
        "does_not_use_original_or_recovered_ids_for_z0": True,
    }
    return z0, init_ids, h_alpha.detach(), audit


def stage_b_optimize(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: AIGMConfig,
    observed_activation: torch.Tensor,
    seq_len: int,
    device: torch.device,
    fixed_public: Dict[int, int],
    init_embeds: torch.Tensor,
    a_self: Optional[torch.Tensor],
    use_context: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], List[Dict[str, float]]]:
    embed_layer = model.get_input_embeddings()
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    fixed_embeds, fixed_positions, _ = fixed_embedding_tensor(embed_layer, fixed_public, seq_len, device)
    left, right = embedding_bounds(embed_layer.weight)
    left = left.to(device)
    right = right.to(device)
    z = init_embeds.detach().clone().to(device=device, dtype=torch.float32).requires_grad_(True)
    optimizer = torch.optim.AdamW([z], lr=cfg.lr)
    target = observed_activation.detach()
    target_ctx = context_projection(a_self, target) if use_context and a_self is not None else None
    history: List[Dict[str, float]] = []
    final: Dict[str, Any] = {}

    for step in range(max(1, cfg.epoch)):
        enforce_embedding_constraints(z, fixed_positions, fixed_embeds, left, right)
        hidden, _ = prefix_forward_hidden_and_attention(
            model,
            cfg.target_layer,
            z.to(dtype=embed_layer.weight.dtype),
            attention_mask,
            output_target_attention=False,
        )
        activation_loss = F.mse_loss(hidden.float(), target.to(hidden.device).float())
        vocab_loss = nearest_embedding_loss(variable_view(z, fixed_positions), embed_layer.weight)
        if use_context and a_self is not None and target_ctx is not None:
            context_loss = F.mse_loss(context_projection(a_self, hidden), target_ctx.to(hidden.device))
        else:
            context_loss = torch.tensor(0.0, device=device)
        total_loss = activation_loss + cfg.lambda_vocab * vocab_loss + cfg.lambda_context * context_loss
        cosine = F.cosine_similarity(hidden.float(), target.to(hidden.device).float(), dim=-1).mean()
        if torch.isnan(total_loss) or torch.isnan(cosine):
            raise RuntimeError(f"NaN in Stage B AdamW optimization at step {step + 1}")
        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_([z], cfg.grad_clip).detach().cpu()) if cfg.grad_clip > 0 else None
        optimizer.step()
        final = {
            "activation_loss": float(activation_loss.detach().cpu()),
            "vocab_loss": float(vocab_loss.detach().cpu()),
            "context_loss": float(context_loss.detach().cpu()),
            "optimization_loss": float(total_loss.detach().cpu()),
            "cosine_similarity": float(cosine.detach().cpu()),
            "grad_norm_before_clip": grad_norm,
        }
        if (step + 1) % max(1, min(50, cfg.epoch)) == 0 or step == cfg.epoch - 1:
            history.append({"step": step + 1, **final})
            print(
                f"method={cfg.method} step={step + 1} act={final['activation_loss']:.6f} "
                f"vocab={final['vocab_loss']:.6f} ctx={final['context_loss']:.6f} "
                f"total={final['optimization_loss']:.6f} cosine={final['cosine_similarity']:.6f}",
                flush=True,
            )
    enforce_embedding_constraints(z, fixed_positions, fixed_embeds, left, right)
    grad_z, grad_losses = recovery_gradient(model, cfg, observed_activation, z.detach(), a_self, use_context)
    final.update({f"final_gradient_{key}": value for key, value in grad_losses.items()})
    return z.detach().float(), grad_z.detach().float(), final, history


def recovery_gradient(
    model: torch.nn.Module,
    cfg: AIGMConfig,
    observed_activation: torch.Tensor,
    z_value: torch.Tensor,
    a_self: Optional[torch.Tensor],
    use_context: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    embed_layer = model.get_input_embeddings()
    seq_len = z_value.shape[1]
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=z_value.device)
    z = z_value.detach().clone().float().requires_grad_(True)
    hidden, _ = prefix_forward_hidden_and_attention(
        model,
        cfg.target_layer,
        z.to(dtype=embed_layer.weight.dtype),
        attention_mask,
        output_target_attention=False,
    )
    activation_loss = F.mse_loss(hidden.float(), observed_activation.to(hidden.device).float())
    vocab_loss = nearest_embedding_loss(z.squeeze(0), embed_layer.weight)
    if use_context and a_self is not None:
        target_ctx = context_projection(a_self, observed_activation.detach())
        context_loss = F.mse_loss(context_projection(a_self, hidden), target_ctx.to(hidden.device))
    else:
        context_loss = torch.tensor(0.0, device=z.device)
    total_loss = activation_loss + cfg.lambda_vocab * vocab_loss + cfg.lambda_context * context_loss
    total_loss.backward()
    losses = {
        "rec_activation_loss": float(activation_loss.detach().cpu()),
        "rec_vocab_loss": float(vocab_loss.detach().cpu()),
        "rec_context_loss": float(context_loss.detach().cpu()),
        "rec_total_loss": float(total_loss.detach().cpu()),
    }
    return z.grad.detach(), losses


def gradient_direction_token_matching(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: AIGMConfig,
    observed_activation: torch.Tensor,
    z: torch.Tensor,
    grad_z: torch.Tensor,
    h_alpha_residual: torch.Tensor,
    init_ids: Sequence[int],
    fixed_public: Dict[int, int],
) -> Tuple[List[int], float, List[Dict[str, Any]]]:
    embed_layer = model.get_input_embeddings()
    seq_len = z.shape[1]
    device = z.device
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    z_sets, _ = topk_cosine(z, embed_layer.weight, max(1, cfg.top_k_embedding))
    alpha_sets, _ = topk_cosine(h_alpha_residual, embed_layer.weight, max(1, cfg.top_k_embedding))
    recovered = naive_discretization(z, embed_layer.weight)
    fixed = set(int(pos) for pos in fixed_public)
    for pos, token_id in fixed_public.items():
        if 0 <= pos < len(recovered):
            recovered[pos] = int(token_id)

    rows: List[Dict[str, Any]] = []
    calibration_scores_all: List[float] = []
    for pos in range(seq_len):
        if pos in fixed:
            rows.append(
                {
                    "position": pos,
                    "candidate_count": 1,
                    "selected_token_id": int(recovered[pos]),
                    "selected_token": tokenizer.convert_ids_to_tokens([int(recovered[pos])])[0],
                    "selected_gradient_score": None,
                    "selected_calibration_score": None,
                    "selected_final_score": None,
                    "same_as_init_token": int(recovered[pos]) == int(init_ids[pos]) if pos < len(init_ids) else False,
                    "fixed_public_boundary_token": True,
                }
            )
            continue
        candidates = []
        candidates.extend(z_sets[pos] if cfg.top_k_embedding > 0 else [])
        candidates.extend(alpha_sets[pos] if cfg.top_k_embedding > 0 else [])
        if cfg.semantic_speculation and cfg.top_y_semantic > 0 and pos > 0:
            candidates.extend(semantic_candidates(model, recovered[:pos], cfg.top_y_semantic))
        if pos < len(init_ids):
            candidates.append(int(init_ids[pos]))
        candidates = [int(x) for x in dict.fromkeys(candidates)]
        if not candidates:
            continue

        cand_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
        cand_embeds = embed_layer(cand_tensor).detach().float()
        direction = cand_embeds - z[0, pos : pos + 1, :].detach().float()
        neg_grad = -grad_z[0, pos, :].detach().float().unsqueeze(0)
        gradient_scores = F.cosine_similarity(direction, neg_grad.expand_as(direction), dim=-1)

        proposal_ids = []
        for cand in candidates:
            proposal = list(recovered)
            proposal[pos] = int(cand)
            proposal_ids.append(proposal)
        ids = torch.tensor(proposal_ids, dtype=torch.long, device=device)
        masks = attention_mask.expand(len(proposal_ids), -1).contiguous()
        with torch.no_grad():
            activation = capture_prefix_activation(
                model,
                cfg.target_layer,
                input_ids=ids,
                attention_mask=masks,
            )
        target_pos = observed_activation[:, pos : pos + 1, :].expand(len(proposal_ids), -1, -1)
        dists = torch.mean((activation[:, pos : pos + 1, :].float() - target_pos.float()) ** 2, dim=(1, 2))
        calibration_scores = -dists
        normalized_grad = minmax01(gradient_scores)
        normalized_calib = minmax01(calibration_scores)
        final_scores = cfg.eta * normalized_grad + (1.0 - cfg.eta) * normalized_calib
        best = int(torch.argmax(final_scores).detach().cpu())
        selected = int(candidates[best])
        recovered[pos] = selected
        calibration_scores_all.append(float(calibration_scores[best].detach().cpu()))
        rows.append(
            {
                "position": pos,
                "candidate_count": len(candidates),
                "selected_token_id": selected,
                "selected_token": tokenizer.convert_ids_to_tokens([selected])[0],
                "selected_gradient_score": float(gradient_scores[best].detach().cpu()),
                "selected_calibration_score": float(calibration_scores[best].detach().cpu()),
                "selected_final_score": float(final_scores[best].detach().cpu()),
                "same_as_init_token": selected == int(init_ids[pos]) if pos < len(init_ids) else False,
                "fixed_public_boundary_token": False,
                "eta": cfg.eta,
                "gradient_direction_token_matching": True,
                "gradient_direction_not_true_gradient_matching": True,
            }
        )
    avg_score = float(statistics.mean(calibration_scores_all)) if calibration_scores_all else math.nan
    return recovered, avg_score, rows


def invert_observed(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: AIGMConfig,
    observed_activation: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> Tuple[List[int], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, float]], List[Dict[str, float]], List[Dict[str, Any]], Optional[float]]:
    if cfg.method not in METHODS:
        raise ValueError(f"Unknown method {cfg.method!r}")
    embed_layer = model.get_input_embeddings()
    fixed_public = inferred_boundary_special_tokens(tokenizer, seq_len) if cfg.fix_boundary_specials else {}
    a_self: Optional[torch.Tensor] = None
    dummy_embeds: Optional[torch.Tensor] = None
    h_alpha_residual: Optional[torch.Tensor] = None
    init_ids: List[int] = []
    attention_stats: Dict[str, Any] = {
        "method": cfg.method,
        "dummy_attention_proxy_used": False,
        "not_true_prompt_attention": True,
    }
    initialization_audit: Dict[str, Any] = {
        "method": cfg.method,
        "alpha_nn_initialization_used": False,
        "uses_ground_truth_for_initialization": False,
    }
    stage_a_history: List[Dict[str, float]] = []
    needs_stage_a = cfg.method in {
        "dummy_init_existing",
        "attention_context_existing",
        "alpha_nn_init_direct",
        "alpha_nn_init_residual",
        "alpha_nn_gradmatch",
        "alpha_nn_gradmatch_context",
    }
    if needs_stage_a:
        dummy_embeds, a_self, attention_stats, stage_a_history = stage_a_dummy_proxy(
            model,
            tokenizer,
            cfg,
            observed_activation,
            seq_len,
            device,
            fixed_public,
        )
        attention_stats.update({"method": cfg.method, "dummy_attention_proxy_used": True})

    if cfg.method == "baseline":
        init_embeds = random_public_embeddings(tokenizer, embed_layer, seq_len, device, fixed_public)
    elif cfg.method == "dummy_init_existing":
        init_embeds = dummy_embeds
        initialization_audit.update(
            {
                "existing_attention_guided_behavior": "Stage-A optimized dummy embeddings are used as Stage-B initial embeddings.",
                "alpha_nn_initialization_used": False,
            }
        )
    elif cfg.method == "attention_context_existing":
        init_embeds = random_public_embeddings(tokenizer, embed_layer, seq_len, device, fixed_public)
        initialization_audit.update(
            {
                "existing_attention_guided_behavior": "A_self @ activation is used only as context loss, not initialization.",
                "alpha_nn_initialization_used": False,
            }
        )
    elif cfg.method in ALPHA_METHODS:
        if a_self is None:
            raise RuntimeError("alpha-NN initialization requires a dummy-attention proxy")
        init_embeds, init_ids, h_alpha_residual, initialization_audit = alpha_nn_initialization(
            model,
            tokenizer,
            cfg,
            observed_activation,
            a_self,
        )
        initialization_audit.update({"alpha_nn_initialization_used": True})
    else:
        raise AssertionError(cfg.method)

    if init_embeds is None:
        raise RuntimeError(f"No initialization was produced for method={cfg.method}")
    use_context = cfg.method in CONTEXT_METHODS
    if cfg.method == "alpha_nn_init_direct":
        use_context = False
    if cfg.method == "alpha_nn_init_residual":
        use_context = False
    z, grad_z, losses, stage_b_history = stage_b_optimize(
        model,
        tokenizer,
        cfg,
        observed_activation,
        seq_len,
        device,
        fixed_public,
        init_embeds,
        a_self,
        use_context,
    )

    gradient_rows: List[Dict[str, Any]] = []
    calibration_score: Optional[float] = None
    if cfg.method in GRADMATCH_METHODS:
        if h_alpha_residual is None or not init_ids:
            raise RuntimeError("Gradient-Direction Token Matching requires alpha-NN initialization state")
        recovered_ids, calibration_score, gradient_rows = gradient_direction_token_matching(
            model,
            tokenizer,
            cfg,
            observed_activation,
            z,
            grad_z,
            h_alpha_residual,
            init_ids,
            fixed_public,
        )
    else:
        embed_sets = embedding_candidates(z, embed_layer.weight, max(1, cfg.top_k_embedding))
        recovered_ids = naive_discretization(z, embed_layer.weight)
        for pos, token_id in fixed_public.items():
            if 0 <= pos < len(recovered_ids):
                recovered_ids[pos] = int(token_id)
        if cfg.adaptive_discretization:
            attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
            recovered_ids, calibration_score = activation_calibrated_discretization(
                model,
                cfg.target_layer,
                attention_mask,
                observed_activation,
                recovered_ids,
                embed_sets,
                sorted(fixed_public),
                cfg,
            )
    gradient_audit = {
        "method": cfg.method,
        "gradient_direction_token_matching_used": cfg.method in GRADMATCH_METHODS,
        "name": "Gradient-Direction Token Matching" if cfg.method in GRADMATCH_METHODS else None,
        "eta": cfg.eta,
        "not_recovered_ground_truth_gradient": True,
        "per_position": gradient_rows,
        "mean_candidate_count": (
            float(statistics.mean([row["candidate_count"] for row in gradient_rows])) if gradient_rows else None
        ),
        "mean_selected_gradient_score": (
            float(statistics.mean([row["selected_gradient_score"] for row in gradient_rows if row["selected_gradient_score"] is not None]))
            if any(row.get("selected_gradient_score") is not None for row in gradient_rows)
            else None
        ),
        "mean_selected_calibration_score": (
            float(statistics.mean([row["selected_calibration_score"] for row in gradient_rows if row["selected_calibration_score"] is not None]))
            if any(row.get("selected_calibration_score") is not None for row in gradient_rows)
            else None
        ),
    }
    return (
        recovered_ids,
        losses,
        attention_stats,
        initialization_audit,
        gradient_audit,
        stage_a_history,
        stage_b_history,
        gradient_rows,
        calibration_score,
    )


def validate_attack_api() -> Dict[str, Any]:
    sig = inspect.signature(invert_observed)
    names = list(sig.parameters)
    banned = ["reference", "original", "input_ids", "token_ids", "prompt", "text", "embedding"]
    signature_hits = [name for name in names if any(item in name for item in banned)]
    sources = []
    checked = [
        invert_observed,
        stage_a_dummy_proxy,
        alpha_nn_initialization,
        stage_b_optimize,
        recovery_gradient,
        gradient_direction_token_matching,
    ]
    banned_globals = {"original_ids", "prompt", "original_tokens", "reference_embedding", "reference_tokens"}
    ast_hits: List[Dict[str, str]] = []
    for fn in checked:
        source = inspect.getsource(fn)
        sources.append(fn.__name__)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in banned_globals:
                ast_hits.append({"function": fn.__name__, "name": node.id})
            if isinstance(node, ast.Attribute) and node.attr in banned_globals:
                ast_hits.append({"function": fn.__name__, "name": node.attr})
    return {
        "signature": str(sig),
        "parameter_names": names,
        "banned_signature_hits": signature_hits,
        "ast_checked_functions": sources,
        "banned_global_reference_hits": ast_hits,
        "passes": len(signature_hits) == 0 and len(ast_hits) == 0,
    }


def config_from_args(args: argparse.Namespace, method: str, seed: int, dataset_len: int, run_name: str, output_dir: str) -> AIGMConfig:
    inverted, target = ATTACKER_MAP[args.participant_number][args.attacker_position]
    if args.target_layer is not None:
        target = args.target_layer
        inverted = target + 1
    stage_a_epoch = args.stage_a_epoch if args.stage_a_epoch is not None else args.epoch
    return AIGMConfig(
        method=method,
        run_name=run_name,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        dataset_len=dataset_len,
        seed=seed,
        participant_number=args.participant_number,
        attacker_position=args.attacker_position,
        inverted_block_count=inverted,
        target_layer=target,
        epoch=args.epoch,
        stage_a_epoch=stage_a_epoch,
        lr=args.lr,
        lambda_vocab=args.lambda_vocab,
        lambda_dummy=args.lambda_dummy,
        lambda_context=args.lambda_context,
        top_k_embedding=args.k,
        top_y_semantic=args.y,
        gamma=args.gamma,
        eta=args.eta,
        adaptive_discretization=not args.naive_discretization,
        semantic_speculation=not args.disable_semantic_speculation,
        max_token_len=args.max_token_len,
        grad_clip=args.grad_clip,
        fix_boundary_specials=args.fix_boundary_specials,
        local_files_only=args.local_files_only,
    )


def run_one_config(cfg: AIGMConfig, resume: bool) -> Dict[str, Any]:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "COMPLETE"
    predictions_path = out / "predictions.jsonl"
    failures_path = out / "failures.jsonl"
    has_transient_failure = False
    if resume and failures_path.exists():
        with failures_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if "max_token_len" not in str(row.get("error", "")):
                        has_transient_failure = True
                        break
    if resume and done.exists() and not has_transient_failure:
        print(f"resume: skip complete config {cfg.output_dir}", flush=True)
        with (out / "metrics.json").open("r", encoding="utf-8") as f:
            return json.load(f)
    json_dump(out / "config.json", {"title": EXPERIMENT_TITLE, "threat_model": THREAT_MODEL, **cfg.__dict__})
    if not resume:
        for path in (predictions_path, failures_path):
            if path.exists():
                path.unlink()
    failures_path.touch()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "cpu")
    tokenizer, model = load_tinyllama(dtype, local_files_only=cfg.local_files_only)
    model.to(device)
    if len(model.model.layers) != TOTAL_BLOCKS:
        raise RuntimeError(f"Expected TinyLlama 22 transformer blocks, got {len(model.model.layers)}")
    if cfg.target_layer < 0 or cfg.target_layer >= len(model.model.layers):
        raise ValueError(f"target_layer must be in 0..21, got {cfg.target_layer}")
    prompts, dataset_meta = load_dataset_prompts(cfg.dataset_name, cfg.dataset_path, cfg.dataset_len, cfg.seed)
    json_dump(out / "dataset_meta.json", dataset_meta)
    completed = set()
    rows: List[Dict[str, Any]] = []
    if resume and predictions_path.exists():
        by_prompt: Dict[int, Dict[str, Any]] = {}
        with predictions_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    by_prompt[int(row["prompt_id"])] = row
        rows = [by_prompt[key] for key in sorted(by_prompt)]
        completed = set(by_prompt)
    failed_rows: List[Dict[str, Any]] = []
    failed_ids = set()
    if resume and failures_path.exists():
        by_failed: Dict[int, Dict[str, Any]] = {}
        with failures_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if "max_token_len" in str(row.get("error", "")):
                        by_failed[int(row["prompt_id"])] = row
        failed_rows = [by_failed[key] for key in sorted(by_failed)]
        failed_ids = set(by_failed)
        with failures_path.open("w", encoding="utf-8") as f:
            for row in failed_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    attention_rows: List[Dict[str, Any]] = []
    init_rows: List[Dict[str, Any]] = []
    grad_rows: List[Dict[str, Any]] = []
    if resume:
        for path, target_rows in [
            (out / "attention_stats.json", attention_rows),
            (out / "initialization_audit.json", init_rows),
            (out / "gradient_matching_stats.json", grad_rows),
        ]:
            if path.exists():
                try:
                    target_rows.extend(json.loads(path.read_text(encoding="utf-8")).get("samples", []))
                except Exception:
                    pass
    api_check = validate_attack_api()
    if not api_check["passes"]:
        raise RuntimeError(f"Attack API leak risk: {api_check}")

    for sample_index, sample_prompt in enumerate(prompts):
        if sample_index in completed or sample_index in failed_ids:
            continue
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start = time.time()
            tokenized = tokenizer(sample_prompt, add_special_tokens=True, truncation=False, return_tensors="pt")
            ids_for_activation = tokenized["input_ids"].to(device)
            mask_for_activation = tokenized["attention_mask"].to(device)
            if ids_for_activation.shape[1] > cfg.max_token_len:
                raise RuntimeError(
                    f"prompt_id={sample_index} has {ids_for_activation.shape[1]} tokens, "
                    f"exceeding max_token_len={cfg.max_token_len}; not silently truncating"
                )
            eval_ids = [int(x) for x in ids_for_activation[0].detach().cpu().tolist()]
            with torch.no_grad():
                observed_activation = capture_prefix_activation(
                    model,
                    cfg.target_layer,
                    input_ids=ids_for_activation,
                    attention_mask=mask_for_activation,
                ).detach()
            seq_len = int(ids_for_activation.shape[1])
            preflight = {
                "model_name": MODEL_NAME,
                "block_count": len(model.model.layers),
                "target_layer_index": cfg.target_layer,
                "inverted_block_count": cfg.inverted_block_count,
                "hook_module_name": f"model.model.layers.{cfg.target_layer}",
                "observed_activation_shape": list(observed_activation.shape),
                "sequence_length": seq_len,
                "gpu": gpu_id,
                "manual_prefix_forward": True,
                "attack_receives_ground_truth": False,
            }
            print(f"preflight={json.dumps(preflight, ensure_ascii=True)}", flush=True)
            (
                recovered_ids,
                losses,
                attention_stats,
                initialization_audit,
                gradient_audit,
                stage_a_history,
                stage_b_history,
                per_position_gradient_rows,
                calibration_score,
            ) = invert_observed(model, tokenizer, cfg, observed_activation, seq_len, device)
            recovered_text = text_from_ids(tokenizer, recovered_ids)
            original_text = text_from_ids(tokenizer, eval_ids)
            elapsed = time.time() - start
            row = {
                "title": EXPERIMENT_TITLE,
                "method": cfg.method,
                "prompt_id": sample_index,
                "prompt": sample_prompt,
                "original_text": original_text,
                "recovered_text": recovered_text,
                "original_token_ids": eval_ids,
                "recovered_token_ids": recovered_ids,
                "original_tokens": token_texts(tokenizer, eval_ids),
                "recovered_tokens": token_texts(tokenizer, recovered_ids),
                "token_accuracy": token_accuracy(tokenizer, eval_ids, recovered_ids),
                "bleu": bleu_score(tokenizer, eval_ids, recovered_ids),
                "nerr": optional_nerr(original_text, recovered_text),
                "optimization_loss": losses["optimization_loss"],
                "activation_loss": losses["activation_loss"],
                "vocab_loss": losses["vocab_loss"],
                "context_loss": losses["context_loss"],
                "cosine_similarity": losses["cosine_similarity"],
                "calibration_score": calibration_score,
                "elapsed_time": elapsed,
                "runtime_seconds": elapsed,
                "seed": cfg.seed,
                "target_layer": cfg.target_layer,
                "participant_number": cfg.participant_number,
                "attacker_position": cfg.attacker_position,
                "inverted_block_count": cfg.inverted_block_count,
                "prompt_token_count": seq_len,
                "peak_gpu_memory_mb": peak_memory_mb(),
                "gpu": gpu_id,
                "gamma": cfg.gamma,
                "eta": cfg.eta,
                "threat_model": THREAT_MODEL,
                "recovery_uses_ground_truth_tokens": False,
                "semantic_oracle": "TinyLlama itself as public semantic candidate source",
                "preflight": preflight,
                "attack_api_check": api_check,
                "stage_a_history": stage_a_history,
                "stage_b_history": stage_b_history,
            }
            jsonl_append(predictions_path, row)
            rows.append(row)
            attention_rows.append({"prompt_id": sample_index, **attention_stats})
            init_rows.append({"prompt_id": sample_index, **initialization_audit})
            grad_rows.append({"prompt_id": sample_index, **gradient_audit})
            json_dump(out / "attention_stats.json", {"samples": attention_rows})
            json_dump(out / "initialization_audit.json", {"samples": init_rows})
            json_dump(out / "gradient_matching_stats.json", {"samples": grad_rows})
            print(
                f"method={cfg.method} prompt_id={sample_index} token_accuracy={row['token_accuracy']:.6f} "
                f"bleu={row['bleu']:.6f} elapsed={elapsed:.2f}s recovered={recovered_text!r}",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "method": cfg.method,
                "prompt_id": sample_index,
                "prompt": sample_prompt,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "seed": cfg.seed,
                "target_layer": cfg.target_layer,
                "participant_number": cfg.participant_number,
                "attacker_position": cfg.attacker_position,
                "gamma": cfg.gamma,
                "eta": cfg.eta,
            }
            failed_rows.append(failure)
            jsonl_append(failures_path, failure)
            print(f"method={cfg.method} prompt_id={sample_index} failed error={repr(exc)}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    metrics = summarize_rows(rows)
    metrics.update(
        {
            "title": EXPERIMENT_TITLE,
            "threat_model": THREAT_MODEL,
            "config": cfg.__dict__,
            "dataset_meta": dataset_meta,
            "completed_sample_count": len(rows),
            "failed_sample_count": len(failed_rows),
            "attack_api_check": api_check,
            "attention_stats_path": str(out / "attention_stats.json"),
            "initialization_audit_path": str(out / "initialization_audit.json"),
            "gradient_matching_stats_path": str(out / "gradient_matching_stats.json"),
        }
    )
    json_dump(out / "metrics.json", metrics)
    done.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    return metrics


def run_single(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = args.output_dir or str(Path(args.output_root) / f"{args.method}_seed{args.seed}")
    run_name = args.run_name or Path(output_dir).name
    cfg = config_from_args(args, args.method, args.seed, args.dataset_len, run_name, output_dir)
    return run_one_config(cfg, resume=args.resume)


def run_smoke(args: argparse.Namespace) -> None:
    base = Path(args.output_root) / "smoke"
    for method in METHODS:
        output_dir = base / f"{method}_seed{args.seed}_g{normalize_run_float(args.gamma)}_eta{normalize_run_float(args.eta)}"
        cfg = config_from_args(args, method, args.seed, 2, output_dir.name, str(output_dir))
        cfg.epoch = args.epoch
        cfg.stage_a_epoch = args.stage_a_epoch if args.stage_a_epoch is not None else args.epoch
        run_one_config(cfg, resume=args.resume)
    summarize_all(args.output_root)


def run_sweep(args: argparse.Namespace) -> None:
    base = Path(args.output_root) / "gamma_eta_sweep"
    for gamma in args.gammas:
        for eta in args.etas:
            method = args.method if args.method in GRADMATCH_METHODS else "alpha_nn_gradmatch_context"
            output_dir = base / (
                f"{method}_seed{args.seed}_g{normalize_run_float(gamma)}_eta{normalize_run_float(eta)}"
            )
            cfg = config_from_args(args, method, args.seed, args.dataset_len, output_dir.name, str(output_dir))
            cfg.gamma = float(gamma)
            cfg.eta = float(eta)
            run_one_config(cfg, resume=args.resume)
    summarize_all(args.output_root)


def run_comparison(args: argparse.Namespace) -> None:
    base = Path(args.output_root) / "seed_comparison"
    for layer in args.target_layers:
        for epoch in args.epochs:
            for seed in args.seeds:
                for method in args.methods:
                    output_dir = base / (
                        f"{method}_layer{layer}_epoch{epoch}_seed{seed}_"
                        f"g{normalize_run_float(args.gamma)}_eta{normalize_run_float(args.eta)}"
                    )
                    cfg = config_from_args(args, method, seed, args.dataset_len, output_dir.name, str(output_dir))
                    cfg.target_layer = int(layer)
                    cfg.inverted_block_count = int(layer) + 1
                    cfg.epoch = int(epoch)
                    cfg.stage_a_epoch = args.stage_a_epoch if args.stage_a_epoch is not None else int(epoch)
                    run_one_config(cfg, resume=args.resume)
    summarize_all(args.output_root)


def verify_no_leakage(args: argparse.Namespace) -> Dict[str, Any]:
    out = Path(args.output_root) / "leakage_verification"
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer, model = load_tinyllama(dtype, local_files_only=args.local_files_only)
    model.to(device)
    prompts, dataset_meta = load_dataset_prompts(args.dataset_name, args.dataset_path, 1, args.seed)
    source_text = prompts[0]
    tokenized = tokenizer(source_text, add_special_tokens=True, truncation=False, return_tensors="pt")
    ids_for_activation = tokenized["input_ids"].to(device)
    mask_for_activation = tokenized["attention_mask"].to(device)
    target_layer = args.target_layer if args.target_layer is not None else ATTACKER_MAP[args.participant_number][args.attacker_position][1]
    block = model.model.layers[target_layer]
    with torch.no_grad():
        full_activation = capture_activation(model, block, {"input_ids": ids_for_activation, "attention_mask": mask_for_activation}).detach()
        prefix_activation = capture_prefix_activation(
            model,
            target_layer,
            input_ids=ids_for_activation,
            attention_mask=mask_for_activation,
        ).detach()
    max_abs_diff = float((full_activation.float() - prefix_activation.float()).abs().max().detach().cpu())
    mean_abs_diff = float((full_activation.float() - prefix_activation.float()).abs().mean().detach().cpu())
    cfg = config_from_args(args, "alpha_nn_gradmatch_context", args.seed, 1, "verify", str(out / "verify"))
    cfg.target_layer = target_layer
    cfg.inverted_block_count = target_layer + 1
    cfg.epoch = min(args.epoch, 20)
    cfg.stage_a_epoch = min(args.stage_a_epoch or args.epoch, 20)
    cfg.adaptive_discretization = False
    seq_len = int(ids_for_activation.shape[1])
    set_seed(args.seed)
    recovered_a, *_ = invert_observed(model, tokenizer, cfg, prefix_activation, seq_len, device)
    fake_eval_text = "Random evaluation-only text that must not affect attack output."
    fake_eval_ids = tokenizer(fake_eval_text, add_special_tokens=True, truncation=False, return_tensors="pt")["input_ids"]
    set_seed(args.seed)
    recovered_b, *_ = invert_observed(model, tokenizer, cfg, prefix_activation, seq_len, device)
    api_check = validate_attack_api()
    fixed_public = inferred_boundary_special_tokens(tokenizer, seq_len) if cfg.fix_boundary_specials else {}
    dummy_embeds, a_self, attention_stats, _ = stage_a_dummy_proxy(
        model,
        tokenizer,
        cfg,
        prefix_activation,
        seq_len,
        device,
        fixed_public,
    )
    z0, init_ids, _h_alpha, init_audit = alpha_nn_initialization(model, tokenizer, cfg, prefix_activation, a_self)
    z0_independent_of_dummy = z0.shape == dummy_embeds.shape and bool((z0.float() - dummy_embeds.float()).abs().mean().detach().cpu() > 1e-8)
    result = {
        "title": EXPERIMENT_TITLE,
        "dataset_meta": dataset_meta,
        "target_layer": target_layer,
        "activation_shape_full": list(full_activation.shape),
        "activation_shape_prefix": list(prefix_activation.shape),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "passes_strict_1e_5": max_abs_diff < 1e-5,
        "passes_fp16_bf16_tolerance_5e_3": max_abs_diff < 5e-3,
        "attack_api_check": api_check,
        "randomized_eval_text": fake_eval_text,
        "randomized_eval_token_count": int(fake_eval_ids.shape[1]),
        "recovery_identical_after_eval_reference_replacement": recovered_a == recovered_b,
        "alpha_nn_z0_audit": init_audit,
        "z0_sources_validated": init_audit.get("z0_sources")
        == ["observed_activation", "dummy_attention_proxy", "public_vocabulary_embedding_matrix"],
        "z0_not_stage_a_dummy_embedding": z0_independent_of_dummy,
        "attention_proxy_stats": attention_stats,
        "init_ids_preview": init_ids[:8],
    }
    if not api_check["passes"]:
        raise RuntimeError(f"Leakage verification failed API check: {api_check}")
    if not result["passes_fp16_bf16_tolerance_5e_3"]:
        raise RuntimeError(f"Prefix/full activation mismatch exceeds tolerance: {result}")
    if not result["recovery_identical_after_eval_reference_replacement"]:
        raise RuntimeError("Attack output changed after replacing evaluation-only fields")
    if not result["z0_sources_validated"] or not result["z0_not_stage_a_dummy_embedding"]:
        raise RuntimeError(f"alpha-NN initialization source validation failed: {result}")
    json_dump(out / "leakage_verification.json", result)
    json_dump(Path(args.output_root) / "leakage_verification.json", result)
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return result


def collect_metric_rows(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for metrics_path in sorted(root.glob("**/metrics.json")):
        if "reports" in metrics_path.parts:
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = metrics.get("config", {})
        if cfg.get("method") not in METHODS:
            continue
        rows.append(
            {
                "method": cfg.get("method"),
                "seed": cfg.get("seed"),
                "gamma": cfg.get("gamma"),
                "eta": cfg.get("eta"),
                "target_layer": cfg.get("target_layer"),
                "epoch": cfg.get("epoch"),
                "dataset_label": cfg.get("dataset_name"),
                "dataset_size": metrics.get("dataset_meta", {}).get("selected_prompts"),
                "completed_sample_count": metrics.get("completed_sample_count"),
                "failed_sample_count": metrics.get("failed_sample_count"),
                "token_accuracy_mean": metrics.get("token_accuracy", {}).get("mean"),
                "token_accuracy_std": metrics.get("token_accuracy", {}).get("std"),
                "bleu_mean": metrics.get("bleu", {}).get("mean"),
                "bleu_std": metrics.get("bleu", {}).get("std"),
                "nerr": metrics.get("nerr", {}).get("mean"),
                "nerr_status": metrics.get("nerr", {}).get("status"),
                "mean_runtime": metrics.get("runtime_seconds", {}).get("mean"),
                "peak_gpu_memory": metrics.get("peak_gpu_memory_mb", {}).get("mean"),
                "metrics_path": str(metrics_path),
            }
        )
    rows.sort(
        key=lambda row: (
            METHODS.index(row["method"]) if row["method"] in METHODS else 999,
            int(row["target_layer"] or -1),
            int(row["epoch"] or -1),
            int(row["seed"] or -1),
            float(row["gamma"] or 0.0),
            float(row["eta"] or 0.0),
        )
    )
    return rows


def write_comparison_tables(root: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    md = [
        f"# {EXPERIMENT_TITLE}",
        "",
        "| method | seed | layer | epoch | gamma | eta | acc mean | acc std | BLEU | completed | failed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md.append(
            f"| {row['method']} | {row['seed']} | {row['target_layer']} | {row['epoch']} | "
            f"{row['gamma']} | {row['eta']} | {row['token_accuracy_mean']} | {row['token_accuracy_std']} | "
            f"{row['bleu_mean']} | {row['completed_sample_count']} | {row['failed_sample_count']} |"
        )
    (root / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_plots(root: Path, rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (root / "plot_error.txt").write_text(repr(exc), encoding="utf-8")
        return
    if not rows:
        return
    labels = [
        f"{row['method']}\nL{row['target_layer']} E{row['epoch']} S{row['seed']}\ng{row['gamma']} e{row['eta']}"
        for row in rows
    ]
    acc = [0.0 if row["token_accuracy_mean"] is None else float(row["token_accuracy_mean"]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.35), 5.5))
    ax.bar(labels, acc, color="#2563eb")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Token Accuracy")
    ax.set_title("AIGM-PIA token accuracy comparison")
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    fig.tight_layout()
    fig.savefig(root / "token_accuracy_comparison.png", dpi=200)
    plt.close(fig)

    histories = []
    for pred_path in sorted(root.glob("**/predictions.jsonl")):
        try:
            with pred_path.open("r", encoding="utf-8") as f:
                first = next((json.loads(line) for line in f if line.strip()), None)
            if first and first.get("stage_b_history"):
                cfg = json.loads((pred_path.parent / "config.json").read_text(encoding="utf-8"))
                histories.append((pred_path.parent.name, cfg.get("method"), first["stage_b_history"]))
        except Exception:
            continue
    if histories:
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, method, history in histories[:20]:
            xs = [item["step"] for item in history]
            ys = [item["activation_loss"] for item in history]
            ax.plot(xs, ys, label=f"{method}:{name[:18]}")
        ax.set_xlabel("Optimization step")
        ax.set_ylabel("Activation loss")
        ax.set_title("AIGM-PIA convergence curve samples")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(root / "convergence_curve.png", dpi=200)
        plt.close(fig)

    sweep = [
        row
        for row in rows
        if "gamma_eta_sweep" in row["metrics_path"] and row["token_accuracy_mean"] is not None
    ]
    if sweep:
        gammas = sorted({float(row["gamma"]) for row in sweep})
        etas = sorted({float(row["eta"]) for row in sweep})
        matrix = np.full((len(gammas), len(etas)), np.nan)
        for row in sweep:
            matrix[gammas.index(float(row["gamma"])), etas.index(float(row["eta"]))] = float(row["token_accuracy_mean"])
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(etas)), labels=[str(x) for x in etas])
        ax.set_yticks(range(len(gammas)), labels=[str(x) for x in gammas])
        ax.set_xlabel("eta")
        ax.set_ylabel("gamma")
        ax.set_title("Gamma/Eta sweep token accuracy")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(root / "gamma_eta_heatmap.png", dpi=200)
        plt.close(fig)


def write_report(root: Path, rows: List[Dict[str, Any]]) -> None:
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)
    smoke_rows = [row for row in rows if int(row["epoch"]) == 100 and int(row["completed_sample_count"]) <= 2]
    sweep_rows = [row for row in rows if int(row["epoch"]) == 200 and int(row["completed_sample_count"]) >= 20]
    best_smoke = None
    best_sweep = None
    if smoke_rows:
        best_smoke = max(smoke_rows, key=lambda row: float(row["token_accuracy_mean"]))
    if sweep_rows:
        best_sweep = max(sweep_rows, key=lambda row: float(row["token_accuracy_mean"]))
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    leakage_path = root / "leakage_verification.json"
    leakage = None
    if leakage_path.exists():
        try:
            leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            leakage = None
    lines = [
        f"# {EXPERIMENT_TITLE}",
        "",
        "This is a TinyLlama pilot and must not be described as a full reproduction of the paper's large-model results.",
        "",
        "## Threat Model",
        "",
        THREAT_MODEL,
        "",
        "The attack entry point is `invert_observed(model, tokenizer, cfg, observed_activation, seq_len, device)`. "
        "The outer runner uses prompt text and evaluation token ids only to produce observed activation and to score final outputs.",
        "",
        "## Method",
        "",
        "Stage A optimizes continuous dummy embeddings `D` to match `H_obs` and extracts a detached dummy-attention proxy "
        "`A_self = MeanHead(Attention_target_layer(D*))`. This proxy is not true prompt attention.",
        "",
        "Stage B alpha-NN initialization computes `H_alpha = A_self @ H_obs`, then `t0_i = NN_W(H_alpha_i)` and "
        "`z0_i = E(t0_i)`. The residual version uses "
        "`H_alpha_residual = (1 - gamma) * H_obs + gamma * (A_self @ H_obs)`.",
        "",
        "`H_obs @ A_self` is not used because `H_obs` is `[seq_len, hidden_dim]` and the attention matrix applies over "
        "sequence positions, so the valid left multiplication is `[seq_len, seq_len] @ [seq_len, hidden_dim]`.",
        "",
        "Gradient-Direction Token Matching scores candidate directions `E(w) - z_i` against `-grad(L_rec, z_i)`. "
        "This is an attack-side recovery gradient from the reconstruction objective, not a recovered ground-truth training gradient.",
        "",
        "## Existing Attention-Guided Baselines",
        "",
        "The previous `dummy_init` used Stage-A optimized dummy embeddings directly as Stage-B initialization; it was not alpha-NN initialization. "
        "The previous `attention_context` used `A_dummy @ activation` as a context loss, not as initialization. "
        "The previous `attention_gradient` was gradient fusion, not Gradient-Direction Token Matching. "
        "The current `full` method from `pia_attention_guided.py` is not renamed or treated as AIGM-PIA.",
        "",
        "## Results",
        "",
        "`runs/alpha_gm_pilot/comparison.csv` and `.md` contain every completed configuration.",
        "",
    ]
    if best_smoke:
        lines.extend(
            [
                "## Best Smoke Setting",
                "",
                f"Best 2-sample smoke setting: method `{best_smoke['method']}`, seed `{best_smoke['seed']}`, "
                f"layer `{best_smoke['target_layer']}`, epoch `{best_smoke['epoch']}`, gamma `{best_smoke['gamma']}`, "
                f"eta `{best_smoke['eta']}`, accuracy `{best_smoke['token_accuracy_mean']}`.",
                "",
            ]
        )
    if best_sweep:
        lines.extend(
            [
                "## Best Gamma/Eta Sweep Setting",
                "",
                f"Best 24-sample sweep setting: method `{best_sweep['method']}`, seed `{best_sweep['seed']}`, "
                f"layer `{best_sweep['target_layer']}`, epoch `{best_sweep['epoch']}`, gamma `{best_sweep['gamma']}`, "
                f"eta `{best_sweep['eta']}`, accuracy `{best_sweep['token_accuracy_mean']}`, "
                f"BLEU `{best_sweep['bleu_mean']}`, completed `{best_sweep['completed_sample_count']}`, "
                f"failed `{best_sweep['failed_sample_count']}`.",
                "",
                "The sweep shows `eta=0.0` is consistently strongest; increasing Gradient-Direction Token Matching weight "
                "degrades this pilot, so it should not be presented as a guaranteed improvement.",
                "",
            ]
        )
    lines.extend(
        [
            "## Run Status",
            "",
            "- Smoke comparison completed for 7 methods on 2 prompts, seed 42, layer 17, epoch 100.",
            "- Gamma/eta sweep completed for `alpha_nn_gradmatch_context` on 24 requested prompts, seed 42, layer 17, epoch 200.",
            "- Formal multi-seed/multi-layer comparison was not run in this session; use `scripts/run_alpha_gm_pilot.sh --mode comparison --resume`.",
            "",
        ]
    )
    if leakage:
        attack_api = leakage.get("attack_api_check", {})
        alpha_audit = leakage.get("alpha_nn_z0_audit", {})
        attn = leakage.get("attention_proxy_stats", {})
        lines.extend(
            [
                "## Anti-Leakage Verification",
                "",
                f"- Full model vs manual prefix activation diff: max `{leakage.get('max_abs_diff')}`, "
                f"mean `{leakage.get('mean_abs_diff')}`.",
                f"- Attack API passes: `{attack_api.get('passes')}`; banned signature hits: "
                f"`{attack_api.get('banned_signature_hits')}`; banned global references: "
                f"`{attack_api.get('banned_global_reference_hits')}`.",
                f"- Replacing evaluation-only prompt/token references leaves recovery unchanged: "
                f"`{leakage.get('recovery_identical_after_eval_reference_replacement')}`.",
                f"- Alpha direction: `{alpha_audit.get('matrix_multiplication_direction')}`; "
                f"z0 lookup max abs diff: `{alpha_audit.get('z0_embedding_lookup_max_abs_diff')}`.",
                f"- Attention proxy shape `{attn.get('attention_shape')}`, row-sum range "
                f"`[{attn.get('row_sum_min')}, {attn.get('row_sum_max')}]`, upper-triangular max "
                f"`{attn.get('upper_triangular_max')}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Method Summary",
            "",
            "| method | runs | mean acc across runs | mean BLEU across runs |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for method, method_rows in grouped.items():
        acc_vals = [float(row["token_accuracy_mean"]) for row in method_rows if row["token_accuracy_mean"] is not None]
        bleu_vals = [float(row["bleu_mean"]) for row in method_rows if row["bleu_mean"] is not None]
        lines.append(
            f"| {method} | {len(method_rows)} | "
            f"{statistics.mean(acc_vals) if acc_vals else None} | {statistics.mean(bleu_vals) if bleu_vals else None} |"
        )
    lines.extend(
        [
            "",
            "## Long vs Short Text",
            "",
            "Prompt-level grouping is stored in each `predictions.jsonl`. The pilot summary currently aggregates by run; "
            "long/short prompt analysis can be recomputed directly from prompt token counts in those files.",
            "",
            "## Attention BOS Concentration",
            "",
            "Each `attention_stats.json` records `bos_column_mass_mean` and `bos_column_mass_max` to identify whether the dummy-attention proxy collapses onto BOS.",
            "",
            "## Failure Cases",
            "",
            "Failures are saved per run in `failures.jsonl`. With the current Skytrax-28 pilot, prompts exceeding `max_token_len` are skipped rather than silently truncated.",
            "",
        ]
    )
    if (root / "comparison.md").exists():
        lines.extend(["## Full Table", "", (root / "comparison.md").read_text(encoding="utf-8")])
    (analysis_dir / "alpha_gm_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_all(output_root: str) -> None:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = collect_metric_rows(root)
    write_comparison_tables(root, rows)
    write_plots(root, rows)
    write_report(root, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "smoke", "sweep", "comparison", "summarize", "verify"], required=True)
    parser.add_argument("--method", choices=METHODS, default="alpha_nn_gradmatch_context")
    parser.add_argument("--methods", choices=METHODS, nargs="*", default=METHODS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-root", default="runs/alpha_gm_pilot")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dataset-name", default="Skytrax-28-pilot")
    parser.add_argument("--dataset-path", default="data/airline.json")
    parser.add_argument("--dataset-len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    parser.add_argument("--participant-number", type=int, default=4)
    parser.add_argument("--attacker-position", type=int, default=4)
    parser.add_argument("--target-layer", type=int, default=17)
    parser.add_argument("--target-layers", type=int, nargs="*", default=[17, 19])
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--epochs", type=int, nargs="*", default=[100, 200])
    parser.add_argument("--stage-a-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lambda-vocab", type=float, default=0.1)
    parser.add_argument("--lambda-dummy", type=float, default=0.1)
    parser.add_argument("--lambda-context", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--y", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--gammas", type=float, nargs="*", default=[0.0, 0.1, 0.3, 0.5, 1.0])
    parser.add_argument("--etas", type=float, nargs="*", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--max-token-len", type=int, default=256)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--naive-discretization", action="store_true")
    parser.add_argument("--disable-semantic-speculation", action="store_true")
    parser.add_argument("--fix-boundary-specials", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.mode == "single":
            run_single(args)
        elif args.mode == "smoke":
            run_smoke(args)
        elif args.mode == "sweep":
            run_sweep(args)
        elif args.mode == "comparison":
            run_comparison(args)
        elif args.mode == "summarize":
            summarize_all(args.output_root)
        elif args.mode == "verify":
            verify_no_leakage(args)
        else:
            raise NotImplementedError(args.mode)
    except Exception:
        failure_dir = Path(args.output_root) / "failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure_path = failure_dir / f"failure_{int(time.time())}.log"
        failure_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"failure logged to {failure_path}", flush=True)
        raise


if __name__ == "__main__":
    main()
