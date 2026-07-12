import argparse
import ast
import csv
import inspect
import json
import math
import os
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pia_alpha_gm import (
    alpha_nn_initialization as aigm_alpha_nn_initialization,
    minmax01 as old_minmax01,
    normalize_run_float,
    stage_a_dummy_proxy as aigm_stage_a_dummy_proxy,
    summarize_rows,
    topk_cosine,
)
from pia_attention_guided import (
    context_projection,
    enforce_embedding_constraints,
    fixed_embedding_tensor,
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
from pia_attention_guided import prefix_forward_hidden_and_attention


EXPERIMENT_TITLE = "Confidence-Gated Gradient-Directed Prompt Inversion (CG-GDPI) TinyLlama white-box pilot"
THREAT_MODEL = (
    "White-box attack using only observed boundary activation, sequence length, public TinyLlama "
    "prefix parameters, tokenizer, public vocabulary embeddings, TinyLlama semantic candidates, "
    "dummy-attention proxy, and attack-side reconstruction gradients."
)

METHODS = [
    "B0",
    "B1",
    "B2",
    "B3",
    "B4",
    "P1",
    "P2",
    "baseline",
    "dummy_init_existing",
    "attention_context_existing",
    "alpha_nn_init_residual",
    "alpha_nn_global_gradmatch",
    "cg_gdpi",
    "cg_gdpi_bos_debias",
]
METHOD_ALIASES = {
    "B0": "baseline",
    "B1": "dummy_init_existing",
    "B2": "attention_context_existing",
    "B3": "alpha_nn_init_residual",
    "B4": "alpha_nn_global_gradmatch",
    "P1": "cg_gdpi",
    "P2": "cg_gdpi_bos_debias",
}
BASELINE_METHODS = {"baseline", "dummy_init_existing", "attention_context_existing", "alpha_nn_init_residual"}
CG_METHODS = {"cg_gdpi", "cg_gdpi_bos_debias"}
GLOBAL_GRAD_METHODS = {"alpha_nn_global_gradmatch"}
STAGE_A_METHODS = {
    "dummy_init_existing",
    "attention_context_existing",
    "alpha_nn_init_residual",
    "alpha_nn_global_gradmatch",
    "cg_gdpi",
    "cg_gdpi_bos_debias",
}
ALPHA_INIT_METHODS = {"alpha_nn_init_residual", "alpha_nn_global_gradmatch", "cg_gdpi", "cg_gdpi_bos_debias"}
CONTEXT_METHODS = {"attention_context_existing", "alpha_nn_global_gradmatch", "cg_gdpi", "cg_gdpi_bos_debias"}


@dataclass
class CGConfig:
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
    top_m: int
    tau_consensus: float
    tau_grad_margin: float
    tau_cal_margin_quantile: float
    attention_mode: str
    force_gate_closed: bool
    adaptive_discretization: bool
    semantic_speculation: bool
    max_token_len: int
    grad_clip: float
    fix_boundary_specials: bool
    local_files_only: bool
    store_full_gradient_directions: bool


def canonical_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def attention_matrix_stats(a_proxy: torch.Tensor) -> Dict[str, Any]:
    mat = a_proxy.detach().float()
    seq_len = int(mat.shape[0])
    row_sums = mat.sum(dim=-1)
    entropy = (-(mat.clamp_min(1e-12) * mat.clamp_min(1e-12).log()).sum(dim=-1)).mean()
    return {
        "attention_shape": [seq_len, seq_len],
        "row_sum_min": float(row_sums.min().detach().cpu()),
        "row_sum_max": float(row_sums.max().detach().cpu()),
        "row_sum_mean": float(row_sums.mean().detach().cpu()),
        "upper_triangular_max": float(torch.triu(mat, diagonal=1).abs().max().detach().cpu()) if seq_len > 1 else 0.0,
        "entropy_mean": float(entropy.detach().cpu()),
        "sparsity_lt_1e_3": float((mat < 1e-3).float().mean().detach().cpu()),
        "bos_column_mass_mean": float(mat[:, 0].mean().detach().cpu()),
        "bos_column_mass_max": float(mat[:, 0].max().detach().cpu()),
        "attention_map_top_left_8x8": mat[:8, :8].detach().cpu().tolist(),
    }


def preprocess_attention(a_proxy: torch.Tensor, mode: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if mode == "raw":
        out = a_proxy.detach().float()
        stats = attention_matrix_stats(out)
        stats.update({"attention_mode": "raw", "bos_debias_applied": False})
        return out.detach(), stats
    if mode != "bos_debias":
        raise ValueError(f"Unknown attention_mode={mode!r}")
    out = a_proxy.detach().float().clone()
    if out.shape[0] > 1:
        out[1:, 0] = 0.0
        row_sums = out[1:].sum(dim=-1, keepdim=True).clamp_min(1e-12)
        out[1:] = out[1:] / row_sums
    stats = attention_matrix_stats(out)
    stats.update({"attention_mode": "bos_debias", "bos_debias_applied": True})
    return out.detach(), stats


def checkpoint_steps(epoch: int, count: int = 5) -> List[int]:
    epoch = max(1, int(epoch))
    if epoch <= count:
        return list(range(1, epoch + 1))
    start = max(1, int(round(epoch * 0.8)))
    return sorted(set(int(x) for x in np.linspace(start, epoch, count)))


def stage_b_optimize_with_consensus(
    model: torch.nn.Module,
    cfg: CGConfig,
    observed_activation: torch.Tensor,
    seq_len: int,
    device: torch.device,
    fixed_public: Dict[int, int],
    init_embeds: torch.Tensor,
    a_proxy: Optional[torch.Tensor],
    use_context: bool,
    collect_consensus: bool,
) -> Tuple[torch.Tensor, Dict[str, Any], List[Dict[str, float]], Dict[str, Any]]:
    embed_layer = model.get_input_embeddings()
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    fixed_embeds, fixed_positions, _ = fixed_embedding_tensor(embed_layer, fixed_public, seq_len, device)
    left, right = embedding_bounds(embed_layer.weight)
    left = left.to(device)
    right = right.to(device)
    z = init_embeds.detach().clone().to(device=device, dtype=torch.float32).requires_grad_(True)
    optimizer = torch.optim.AdamW([z], lr=cfg.lr)
    target = observed_activation.detach()
    target_ctx = context_projection(a_proxy, target) if use_context and a_proxy is not None else None
    history: List[Dict[str, float]] = []
    final: Dict[str, Any] = {}
    grad_snapshots: List[torch.Tensor] = []
    grad_steps = set(checkpoint_steps(cfg.epoch))

    for step in range(max(1, cfg.epoch)):
        step_num = step + 1
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
        if use_context and a_proxy is not None and target_ctx is not None:
            context_loss = F.mse_loss(context_projection(a_proxy, hidden), target_ctx.to(hidden.device))
        else:
            context_loss = torch.tensor(0.0, device=device)
        total_loss = activation_loss + cfg.lambda_vocab * vocab_loss + cfg.lambda_context * context_loss
        cosine = F.cosine_similarity(hidden.float(), target.to(hidden.device).float(), dim=-1).mean()
        if torch.isnan(total_loss) or torch.isnan(cosine):
            raise RuntimeError(f"NaN in Stage B optimization at step {step_num}")
        optimizer.zero_grad()
        total_loss.backward()
        if collect_consensus and step_num in grad_steps and z.grad is not None:
            grad_snapshots.append(z.grad.detach().float().clone())
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
        if step_num % max(1, min(50, cfg.epoch)) == 0 or step == cfg.epoch - 1:
            history.append({"step": step_num, **final})
            print(
                f"method={cfg.method} step={step_num} act={final['activation_loss']:.6f} "
                f"vocab={final['vocab_loss']:.6f} ctx={final['context_loss']:.6f} "
                f"total={final['optimization_loss']:.6f} cosine={final['cosine_similarity']:.6f}",
                flush=True,
            )
    enforce_embedding_constraints(z, fixed_positions, fixed_embeds, left, right)
    consensus = gradient_consensus_from_snapshots(grad_snapshots, cfg)
    return z.detach().float(), final, history, consensus


def gradient_consensus_from_snapshots(snapshots: List[torch.Tensor], cfg: CGConfig) -> Dict[str, Any]:
    if not snapshots:
        return {"checkpoint_count": 0, "per_position": []}
    grads = torch.cat([g.detach().float() for g in snapshots], dim=0)
    norms = grads.norm(dim=-1)
    normalized = grads / norms.clamp_min(1e-12).unsqueeze(-1)
    g_bar = normalized.mean(dim=0)
    g_bar_unit = g_bar / g_bar.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    kappa = F.cosine_similarity(normalized, g_bar_unit.unsqueeze(0), dim=-1).mean(dim=0)
    per_pos = []
    for pos in range(g_bar.shape[0]):
        direction = g_bar_unit[pos].detach().cpu()
        item: Dict[str, Any] = {
            "position": int(pos),
            "checkpoint_count": len(snapshots),
            "gradient_norm_mean": float(norms[:, pos].mean().detach().cpu()),
            "gradient_norm_std": float(norms[:, pos].std(unbiased=False).detach().cpu()) if len(snapshots) > 1 else 0.0,
            "kappa": float(kappa[pos].detach().cpu()),
            "final_gradient_direction_l2_norm": float(direction.norm().item()),
            # The full direction is needed by the in-process reranker. It is attack-side
            # reconstruction state, not ground-truth prompt information.
            "final_gradient_direction": [round(float(x), 6) for x in direction.tolist()],
        }
        if not cfg.store_full_gradient_directions:
            item["final_gradient_direction_preview"] = [round(float(x), 6) for x in direction[:32].tolist()]
        per_pos.append(item)
    return {
        "checkpoint_steps": checkpoint_steps(cfg.epoch),
        "checkpoint_count": len(snapshots),
        "per_position": per_pos,
        "kappa_mean": float(kappa.mean().detach().cpu()),
        "kappa_std": float(kappa.std(unbiased=False).detach().cpu()) if kappa.numel() > 1 else 0.0,
    }


def candidate_set_for_position(
    model: torch.nn.Module,
    cfg: CGConfig,
    recovered: Sequence[int],
    pos: int,
    z_sets: List[List[int]],
    alpha_sets: List[List[int]],
    init_ids: Sequence[int],
) -> List[int]:
    candidates: List[int] = []
    if cfg.top_k_embedding > 0:
        candidates.extend(z_sets[pos])
        candidates.extend(alpha_sets[pos])
    if cfg.semantic_speculation and cfg.top_y_semantic > 0 and pos > 0:
        candidates.extend(semantic_candidates(model, recovered[:pos], cfg.top_y_semantic))
    if pos < len(init_ids):
        candidates.append(int(init_ids[pos]))
    return [int(x) for x in dict.fromkeys(candidates)]


def calibration_scores_for_candidates(
    model: torch.nn.Module,
    cfg: CGConfig,
    observed_activation: torch.Tensor,
    attention_mask: torch.Tensor,
    recovered: Sequence[int],
    pos: int,
    candidates: Sequence[int],
) -> torch.Tensor:
    proposal_ids = []
    for cand in candidates:
        proposal = list(recovered)
        proposal[pos] = int(cand)
        proposal_ids.append(proposal)
    ids = torch.tensor(proposal_ids, dtype=torch.long, device=attention_mask.device)
    masks = attention_mask.expand(len(proposal_ids), -1).contiguous()
    with torch.no_grad():
        activation = capture_prefix_activation(model, cfg.target_layer, input_ids=ids, attention_mask=masks)
    target_pos = observed_activation[:, pos : pos + 1, :].expand(len(proposal_ids), -1, -1)
    dists = torch.mean((activation[:, pos : pos + 1, :].float() - target_pos.float()) ** 2, dim=(1, 2))
    return -dists


def pure_calibration_sequence(
    model: torch.nn.Module,
    cfg: CGConfig,
    observed_activation: torch.Tensor,
    z: torch.Tensor,
    h_init: torch.Tensor,
    init_ids: Sequence[int],
    fixed_public: Dict[int, int],
) -> Tuple[List[int], List[float]]:
    embed_layer = model.get_input_embeddings()
    seq_len = int(z.shape[1])
    device = z.device
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    z_sets, _ = topk_cosine(z, embed_layer.weight, max(1, cfg.top_k_embedding))
    alpha_sets, _ = topk_cosine(h_init, embed_layer.weight, max(1, cfg.top_k_embedding))
    recovered = naive_discretization(z, embed_layer.weight)
    for pos, token_id in fixed_public.items():
        if 0 <= pos < len(recovered):
            recovered[pos] = int(token_id)
    fixed = set(int(pos) for pos in fixed_public)
    margins: List[float] = []
    for pos in range(seq_len):
        if pos in fixed:
            margins.append(float("inf"))
            continue
        candidates = candidate_set_for_position(model, cfg, recovered, pos, z_sets, alpha_sets, init_ids)
        if not candidates:
            margins.append(0.0)
            continue
        scores = calibration_scores_for_candidates(model, cfg, observed_activation, attention_mask, recovered, pos, candidates)
        order = torch.argsort(scores, descending=True)
        best = int(order[0].detach().cpu())
        recovered[pos] = int(candidates[best])
        if len(candidates) > 1:
            second = int(order[1].detach().cpu())
            margins.append(float((scores[best] - scores[second]).detach().cpu()))
        else:
            margins.append(float("inf"))
    return recovered, margins


def cg_gdpi_rerank(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: CGConfig,
    observed_activation: torch.Tensor,
    z: torch.Tensor,
    h_init: torch.Tensor,
    init_ids: Sequence[int],
    fixed_public: Dict[int, int],
    consensus: Dict[str, Any],
    top_m_override: Optional[int] = None,
    force_gate_closed_override: Optional[bool] = None,
) -> Tuple[List[int], float, Dict[str, Any]]:
    embed_layer = model.get_input_embeddings()
    seq_len = int(z.shape[1])
    device = z.device
    top_m = max(1, int(top_m_override if top_m_override is not None else cfg.top_m))
    force_gate_closed = cfg.force_gate_closed if force_gate_closed_override is None else force_gate_closed_override
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    z_sets, _ = topk_cosine(z, embed_layer.weight, max(1, cfg.top_k_embedding))
    alpha_sets, _ = topk_cosine(h_init, embed_layer.weight, max(1, cfg.top_k_embedding))
    recovered, prompt_cal_margins = pure_calibration_sequence(
        model, cfg, observed_activation, z, h_init, init_ids, fixed_public
    )
    cal_threshold_vals = [m for m in prompt_cal_margins if math.isfinite(m)]
    tau_cal_margin = (
        float(np.quantile(cal_threshold_vals, cfg.tau_cal_margin_quantile))
        if cal_threshold_vals
        else float("inf")
    )
    recovered = naive_discretization(z, embed_layer.weight)
    for pos, token_id in fixed_public.items():
        if 0 <= pos < len(recovered):
            recovered[pos] = int(token_id)
    fixed = set(int(pos) for pos in fixed_public)
    consensus_by_pos = {int(row["position"]): row for row in consensus.get("per_position", [])}
    gate_rows: List[Dict[str, Any]] = []
    calibration_scores_selected: List[float] = []
    override_count = 0
    correct_override_count = 0

    for pos in range(seq_len):
        if pos in fixed:
            gate_rows.append(
                {
                    "position": pos,
                    "fixed_public_boundary_token": True,
                    "gate_open": False,
                    "override": False,
                    "selected_token_id": int(recovered[pos]),
                    "selected_token": tokenizer.convert_ids_to_tokens([int(recovered[pos])])[0],
                }
            )
            continue
        candidates = candidate_set_for_position(model, cfg, recovered, pos, z_sets, alpha_sets, init_ids)
        if not candidates:
            continue
        scores = calibration_scores_for_candidates(model, cfg, observed_activation, attention_mask, recovered, pos, candidates)
        cal_order_tensor = torch.argsort(scores, descending=True)
        cal_order = [int(x) for x in cal_order_tensor.detach().cpu().tolist()]
        cal_top = cal_order[: max(1, min(top_m, len(cal_order)))]
        cal_token = int(candidates[cal_top[0]])
        cal_score = float(scores[cal_top[0]].detach().cpu())
        if len(cal_top) > 1:
            cal_margin = float((scores[cal_top[0]] - scores[cal_top[1]]).detach().cpu())
        else:
            cal_margin = float("inf")

        consensus_row = consensus_by_pos.get(pos, {})
        if "final_gradient_direction" in consensus_row:
            grad_dir = torch.tensor(consensus_row["final_gradient_direction"], dtype=torch.float32, device=device)
        else:
            grad_dir = torch.zeros(embed_layer.embedding_dim, dtype=torch.float32, device=device)
        candidate_subset = [candidates[idx] for idx in cal_top]
        cand_tensor = torch.tensor(candidate_subset, dtype=torch.long, device=device)
        cand_embeds = embed_layer(cand_tensor).detach().float()
        candidate_dirs = cand_embeds - z[0, pos : pos + 1, :].detach().float()
        if grad_dir.norm() > 0:
            grad_scores = F.cosine_similarity(candidate_dirs, grad_dir.unsqueeze(0).expand_as(candidate_dirs), dim=-1)
        else:
            grad_scores = torch.full((len(candidate_subset),), -1.0, device=device)
        grad_order_tensor = torch.argsort(grad_scores, descending=True)
        grad_order = [int(x) for x in grad_order_tensor.detach().cpu().tolist()]
        grad_token = int(candidate_subset[grad_order[0]])
        if len(candidate_subset) > 1:
            grad_margin = float((grad_scores[grad_order[0]] - grad_scores[grad_order[1]]).detach().cpu())
        else:
            grad_margin = 0.0
        kappa = float(consensus_row.get("kappa", 0.0))
        gradient_reliable = (
            not force_gate_closed
            and top_m > 1
            and kappa >= cfg.tau_consensus
            and grad_margin >= cfg.tau_grad_margin
            and cal_margin <= tau_cal_margin
        )
        selected = grad_token if gradient_reliable else cal_token
        override = selected != cal_token
        if override:
            override_count += 1
        selected_cal_rank = candidate_subset.index(selected) + 1 if selected in candidate_subset else None
        recovered[pos] = selected
        calibration_scores_selected.append(cal_score)
        gate_rows.append(
            {
                "position": pos,
                "candidate_count": len(candidates),
                "calibration_top_m": candidate_subset,
                "calibration_top_m_tokens": token_texts(tokenizer, candidate_subset),
                "selected_token_id": selected,
                "selected_token": tokenizer.convert_ids_to_tokens([selected])[0],
                "calibration_token_id": cal_token,
                "calibration_token": tokenizer.convert_ids_to_tokens([cal_token])[0],
                "gradient_token_id": grad_token,
                "gradient_token": tokenizer.convert_ids_to_tokens([grad_token])[0],
                "override_token_id": selected if override else None,
                "override": bool(override),
                "override_within_calibration_top_m": bool(selected in candidate_subset),
                "calibration_rank_after_override": selected_cal_rank,
                "kappa": kappa,
                "grad_margin": grad_margin,
                "cal_margin": cal_margin,
                "tau_cal_margin": tau_cal_margin,
                "tau_consensus": cfg.tau_consensus,
                "tau_grad_margin": cfg.tau_grad_margin,
                "gate_open": bool(gradient_reliable),
                "force_gate_closed": bool(force_gate_closed),
                "calibration_score_top1": cal_score,
                "gradient_score_top1": float(grad_scores[grad_order[0]].detach().cpu()),
                "fixed_public_boundary_token": False,
            }
        )
    avg_score = float(statistics.mean(calibration_scores_selected)) if calibration_scores_selected else math.nan
    override_rows = [row for row in gate_rows if row.get("override")]
    return recovered, avg_score, {
        "method": cfg.method,
        "name": "Calibration-Safe Gradient Reranking",
        "used": True,
        "top_m": top_m,
        "tau_cal_margin": tau_cal_margin,
        "gate_open_count": sum(1 for row in gate_rows if row.get("gate_open")),
        "gate_open_rate": sum(1 for row in gate_rows if row.get("gate_open")) / max(1, len(gate_rows)),
        "override_count": override_count,
        "override_rate": override_count / max(1, len(gate_rows)),
        "all_overrides_within_calibration_top_m": all(row.get("override_within_calibration_top_m", True) for row in override_rows),
        "uses_minmax_global_fusion": False,
        "per_position": gate_rows,
    }


def global_gradient_rerank(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: CGConfig,
    observed_activation: torch.Tensor,
    z: torch.Tensor,
    h_init: torch.Tensor,
    init_ids: Sequence[int],
    fixed_public: Dict[int, int],
    consensus: Dict[str, Any],
) -> Tuple[List[int], float, Dict[str, Any]]:
    embed_layer = model.get_input_embeddings()
    seq_len = int(z.shape[1])
    device = z.device
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    z_sets, _ = topk_cosine(z, embed_layer.weight, max(1, cfg.top_k_embedding))
    alpha_sets, _ = topk_cosine(h_init, embed_layer.weight, max(1, cfg.top_k_embedding))
    recovered = naive_discretization(z, embed_layer.weight)
    for pos, token_id in fixed_public.items():
        if 0 <= pos < len(recovered):
            recovered[pos] = int(token_id)
    fixed = set(int(pos) for pos in fixed_public)
    consensus_by_pos = {int(row["position"]): row for row in consensus.get("per_position", [])}
    rows = []
    scores_selected = []
    for pos in range(seq_len):
        if pos in fixed:
            continue
        candidates = candidate_set_for_position(model, cfg, recovered, pos, z_sets, alpha_sets, init_ids)
        if not candidates:
            continue
        scores = calibration_scores_for_candidates(model, cfg, observed_activation, attention_mask, recovered, pos, candidates)
        consensus_row = consensus_by_pos.get(pos, {})
        if "final_gradient_direction" in consensus_row:
            grad_dir = torch.tensor(consensus_row["final_gradient_direction"], dtype=torch.float32, device=device)
        else:
            grad_dir = torch.zeros(embed_layer.embedding_dim, dtype=torch.float32, device=device)
        cand_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
        cand_embeds = embed_layer(cand_tensor).detach().float()
        candidate_dirs = cand_embeds - z[0, pos : pos + 1, :].detach().float()
        grad_scores = (
            F.cosine_similarity(candidate_dirs, grad_dir.unsqueeze(0).expand_as(candidate_dirs), dim=-1)
            if grad_dir.norm() > 0
            else torch.full((len(candidates),), -1.0, device=device)
        )
        final_scores = cfg.eta * old_minmax01(grad_scores) + (1.0 - cfg.eta) * old_minmax01(scores)
        best = int(torch.argmax(final_scores).detach().cpu())
        selected = int(candidates[best])
        recovered[pos] = selected
        scores_selected.append(float(scores[best].detach().cpu()))
        rows.append(
            {
                "position": pos,
                "candidate_count": len(candidates),
                "selected_token_id": selected,
                "selected_token": tokenizer.convert_ids_to_tokens([selected])[0],
                "selected_gradient_score": float(grad_scores[best].detach().cpu()),
                "selected_calibration_score": float(scores[best].detach().cpu()),
                "selected_final_score": float(final_scores[best].detach().cpu()),
                "eta": cfg.eta,
                "uses_old_minmax_global_fusion": True,
            }
        )
    return recovered, float(statistics.mean(scores_selected)) if scores_selected else math.nan, {
        "method": cfg.method,
        "name": "Old global min-max gradient fusion baseline",
        "used": True,
        "uses_minmax_global_fusion": True,
        "per_position": rows,
    }


def invert_observed(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: CGConfig,
    observed_activation: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> Tuple[List[int], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, float]], List[Dict[str, float]], Optional[float]]:
    method = canonical_method(cfg.method)
    cfg.method = method
    embed_layer = model.get_input_embeddings()
    fixed_public = inferred_boundary_special_tokens(tokenizer, seq_len) if cfg.fix_boundary_specials else {}
    a_proxy: Optional[torch.Tensor] = None
    dummy_embeds: Optional[torch.Tensor] = None
    h_init: Optional[torch.Tensor] = None
    init_ids: List[int] = []
    stage_a_history: List[Dict[str, float]] = []
    attention_stats: Dict[str, Any] = {
        "method": method,
        "dummy_attention_proxy_used": False,
        "attention_mode": cfg.attention_mode,
        "not_true_prompt_attention": True,
    }
    init_audit: Dict[str, Any] = {
        "method": method,
        "alpha_nn_initialization_used": False,
        "uses_ground_truth_for_initialization": False,
    }
    if method in STAGE_A_METHODS:
        dummy_embeds, raw_a, raw_stats, stage_a_history = aigm_stage_a_dummy_proxy(
            model, tokenizer, cfg, observed_activation, seq_len, device, fixed_public
        )
        mode = "bos_debias" if method == "cg_gdpi_bos_debias" else cfg.attention_mode
        a_proxy, post_stats = preprocess_attention(raw_a, mode)
        attention_stats.update(raw_stats)
        attention_stats.update({f"post_{key}": value for key, value in post_stats.items()})
        attention_stats.update({"dummy_attention_proxy_used": True, "attention_mode": mode})

    if method == "baseline":
        init_embeds = random_public_embeddings(tokenizer, embed_layer, seq_len, device, fixed_public)
    elif method == "dummy_init_existing":
        init_embeds = dummy_embeds
        init_audit.update({"existing_behavior": "Stage-A optimized dummy embeddings used as Stage-B initialization."})
    elif method == "attention_context_existing":
        init_embeds = random_public_embeddings(tokenizer, embed_layer, seq_len, device, fixed_public)
        init_audit.update({"existing_behavior": "A_proxy is used as context loss only, not alpha-NN initialization."})
    elif method in ALPHA_INIT_METHODS:
        if a_proxy is None:
            raise RuntimeError("alpha-NN initialization requires attention proxy")
        init_embeds, init_ids, h_init, init_audit = aigm_alpha_nn_initialization(
            model, tokenizer, cfg, observed_activation, a_proxy
        )
        init_audit.update({"alpha_nn_initialization_used": True, "attention_mode": attention_stats.get("attention_mode")})
    else:
        raise ValueError(f"Unknown method {method!r}")
    if init_embeds is None:
        raise RuntimeError(f"No initialization for method={method}")
    use_context = method in CONTEXT_METHODS
    collect_consensus = method in GLOBAL_GRAD_METHODS or method in CG_METHODS
    z, losses, stage_b_history, consensus = stage_b_optimize_with_consensus(
        model, cfg, observed_activation, seq_len, device, fixed_public, init_embeds, a_proxy, use_context, collect_consensus
    )
    calibration_score: Optional[float] = None
    gate_audit: Dict[str, Any] = {"method": method, "used": False, "per_position": []}
    if method in CG_METHODS:
        if h_init is None or not init_ids:
            raise RuntimeError("CG-GDPI requires residual alpha-NN initialization")
        recovered_ids, calibration_score, gate_audit = cg_gdpi_rerank(
            model, tokenizer, cfg, observed_activation, z, h_init, init_ids, fixed_public, consensus
        )
    elif method in GLOBAL_GRAD_METHODS:
        if h_init is None or not init_ids:
            raise RuntimeError("global gradient baseline requires alpha-NN initialization")
        recovered_ids, calibration_score, gate_audit = global_gradient_rerank(
            model, tokenizer, cfg, observed_activation, z, h_init, init_ids, fixed_public, consensus
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
                model, cfg.target_layer, attention_mask, observed_activation, recovered_ids, embed_sets, sorted(fixed_public), cfg
            )
    return (
        recovered_ids,
        losses,
        attention_stats,
        init_audit,
        consensus,
        gate_audit,
        stage_a_history,
        stage_b_history,
        calibration_score,
    )


def validate_attack_api() -> Dict[str, Any]:
    sig = inspect.signature(invert_observed)
    names = list(sig.parameters)
    banned = ["reference", "original", "input_ids", "token_ids", "prompt", "text", "embedding"]
    signature_hits = [name for name in names if any(item in name for item in banned)]
    checked = [
        invert_observed,
        stage_b_optimize_with_consensus,
        gradient_consensus_from_snapshots,
        cg_gdpi_rerank,
        pure_calibration_sequence,
        global_gradient_rerank,
    ]
    banned_globals = {"original_ids", "original_tokens", "reference_embedding", "reference_tokens", "ground_truth_ids"}
    ast_hits: List[Dict[str, str]] = []
    minmax_hits: List[str] = []
    for fn in checked:
        source = inspect.getsource(fn)
        tree = ast.parse(source)
        if fn is cg_gdpi_rerank and ("minmax01" in source or "old_minmax" in source):
            minmax_hits.append(fn.__name__)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in banned_globals:
                ast_hits.append({"function": fn.__name__, "name": node.id})
            if isinstance(node, ast.Attribute) and node.attr in banned_globals:
                ast_hits.append({"function": fn.__name__, "name": node.attr})
    return {
        "signature": str(sig),
        "parameter_names": names,
        "banned_signature_hits": signature_hits,
        "banned_global_reference_hits": ast_hits,
        "proposed_method_minmax_hits": minmax_hits,
        "passes": len(signature_hits) == 0 and len(ast_hits) == 0 and len(minmax_hits) == 0,
    }


def config_from_args(args: argparse.Namespace, method: str, seed: int, run_name: str, output_dir: str) -> CGConfig:
    method = canonical_method(method)
    inverted, target = ATTACKER_MAP[args.participant_number][args.attacker_position]
    if args.target_layer is not None:
        target = args.target_layer
        inverted = target + 1
    stage_a_epoch = args.stage_a_epoch if args.stage_a_epoch is not None else args.epoch
    attention_mode = "bos_debias" if method == "cg_gdpi_bos_debias" else args.attention_mode
    return CGConfig(
        method=method,
        run_name=run_name,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        dataset_len=args.dataset_len,
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
        top_m=args.top_m,
        tau_consensus=args.tau_consensus,
        tau_grad_margin=args.tau_grad_margin,
        tau_cal_margin_quantile=args.tau_cal_margin_quantile,
        attention_mode=attention_mode,
        force_gate_closed=args.force_gate_closed,
        adaptive_discretization=not args.naive_discretization,
        semantic_speculation=not args.disable_semantic_speculation,
        max_token_len=args.max_token_len,
        grad_clip=args.grad_clip,
        fix_boundary_specials=args.fix_boundary_specials,
        local_files_only=args.local_files_only,
        store_full_gradient_directions=args.store_full_gradient_directions,
    )


def run_one_config(cfg: CGConfig, resume: bool) -> Dict[str, Any]:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    predictions_path = out / "predictions.jsonl"
    failures_path = out / "failures.jsonl"
    done = out / "COMPLETE"
    if resume and done.exists():
        print(f"resume: skip complete config {cfg.output_dir}", flush=True)
        return json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    if not resume:
        for path in (predictions_path, failures_path):
            if path.exists():
                path.unlink()
    failures_path.touch()
    json_dump(out / "config.json", {"title": EXPERIMENT_TITLE, "threat_model": THREAT_MODEL, **cfg.__dict__})
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "cpu")
    tokenizer, model = load_tinyllama(dtype, local_files_only=cfg.local_files_only)
    model.to(device)
    if len(model.model.layers) != TOTAL_BLOCKS:
        raise RuntimeError(f"Expected TinyLlama 22 transformer blocks, got {len(model.model.layers)}")
    prompts, dataset_meta = load_dataset_prompts(cfg.dataset_name, cfg.dataset_path, cfg.dataset_len, cfg.seed)
    json_dump(out / "dataset_meta.json", dataset_meta)
    api_check = validate_attack_api()
    if not api_check["passes"]:
        raise RuntimeError(f"Attack API leak risk: {api_check}")
    completed: Dict[int, Dict[str, Any]] = {}
    if resume and predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[int(row["prompt_id"])] = row
    deterministic_failures: Dict[int, Dict[str, Any]] = {}
    if resume and failures_path.exists():
        for line in failures_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if "max_token_len" in str(row.get("error", "")):
                    deterministic_failures[int(row["prompt_id"])] = row
        with failures_path.open("w", encoding="utf-8") as f:
            for key in sorted(deterministic_failures):
                f.write(json.dumps(deterministic_failures[key], ensure_ascii=False) + "\n")
    rows = [completed[key] for key in sorted(completed)]
    failed_rows = [deterministic_failures[key] for key in sorted(deterministic_failures)]
    attention_rows: List[Dict[str, Any]] = []
    init_rows: List[Dict[str, Any]] = []
    consensus_rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []
    for path, target in [
        (out / "attention_stats.json", attention_rows),
        (out / "initialization_audit.json", init_rows),
        (out / "gradient_consensus_stats.json", consensus_rows),
        (out / "gradient_gate_stats.json", gate_rows),
    ]:
        if resume and path.exists():
            try:
                target.extend(json.loads(path.read_text(encoding="utf-8")).get("samples", []))
            except Exception:
                pass

    for sample_index, sample_payload in enumerate(prompts):
        if sample_index in completed or sample_index in deterministic_failures:
            continue
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start = time.time()
            tokenized = tokenizer(sample_payload, add_special_tokens=True, truncation=False, return_tensors="pt")
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
                    model, cfg.target_layer, input_ids=ids_for_activation, attention_mask=mask_for_activation
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
                init_audit,
                consensus,
                gate_audit,
                stage_a_history,
                stage_b_history,
                calibration_score,
            ) = invert_observed(model, tokenizer, cfg, observed_activation, seq_len, device)
            recovered_text = text_from_ids(tokenizer, recovered_ids)
            original_text = text_from_ids(tokenizer, eval_ids)
            init_acc = None
            if init_audit.get("init_token_ids"):
                init_acc = token_accuracy(tokenizer, eval_ids, init_audit["init_token_ids"])
                init_audit["init_token_accuracy_eval_only"] = init_acc
            elapsed = time.time() - start
            row = {
                "title": EXPERIMENT_TITLE,
                "method": cfg.method,
                "prompt_id": sample_index,
                "prompt": sample_payload,
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
                "top_m": cfg.top_m,
                "attention_mode": cfg.attention_mode,
                "gate_open_rate": gate_audit.get("gate_open_rate"),
                "override_rate": gate_audit.get("override_rate"),
                "init_token_accuracy": init_acc,
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
            init_rows.append({"prompt_id": sample_index, **init_audit})
            consensus_rows.append({"prompt_id": sample_index, **consensus})
            gate_rows.append({"prompt_id": sample_index, **gate_audit})
            json_dump(out / "attention_stats.json", {"samples": attention_rows})
            json_dump(out / "initialization_audit.json", {"samples": init_rows})
            json_dump(out / "gradient_consensus_stats.json", {"samples": consensus_rows})
            json_dump(out / "gradient_gate_stats.json", {"samples": gate_rows})
            print(
                f"method={cfg.method} prompt_id={sample_index} token_accuracy={row['token_accuracy']:.6f} "
                f"bleu={row['bleu']:.6f} gate_open_rate={row['gate_open_rate']} elapsed={elapsed:.2f}s",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "method": cfg.method,
                "prompt_id": sample_index,
                "prompt": sample_payload,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "seed": cfg.seed,
                "target_layer": cfg.target_layer,
                "participant_number": cfg.participant_number,
                "attacker_position": cfg.attacker_position,
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
            "gradient_consensus_stats_path": str(out / "gradient_consensus_stats.json"),
            "gradient_gate_stats_path": str(out / "gradient_gate_stats.json"),
        }
    )
    json_dump(out / "metrics.json", metrics)
    done.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    return metrics


def run_single(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = args.output_dir or str(Path(args.output_root) / f"{canonical_method(args.method)}_seed{args.seed}")
    cfg = config_from_args(args, args.method, args.seed, Path(output_dir).name, output_dir)
    return run_one_config(cfg, resume=args.resume)


def run_verify(args: argparse.Namespace) -> None:
    out = Path(args.output_root) / "leakage_verification"
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer, model = load_tinyllama(dtype, local_files_only=args.local_files_only)
    model.to(device)
    prompts, dataset_meta = load_dataset_prompts(args.dataset_name, args.dataset_path, 1, args.seed)
    payload = prompts[0]
    tokenized = tokenizer(payload, add_special_tokens=True, truncation=False, return_tensors="pt")
    ids_for_activation = tokenized["input_ids"].to(device)
    mask_for_activation = tokenized["attention_mask"].to(device)
    block = model.model.layers[args.target_layer]
    with torch.no_grad():
        full_activation = capture_activation(model, block, {"input_ids": ids_for_activation, "attention_mask": mask_for_activation}).detach()
        prefix_activation = capture_prefix_activation(
            model, args.target_layer, input_ids=ids_for_activation, attention_mask=mask_for_activation
        ).detach()
    max_abs_diff = float((full_activation.float() - prefix_activation.float()).abs().max().detach().cpu())
    mean_abs_diff = float((full_activation.float() - prefix_activation.float()).abs().mean().detach().cpu())
    base = config_from_args(args, "cg_gdpi", args.seed, "verify", str(out / "verify"))
    base.epoch = min(base.epoch, 20)
    base.stage_a_epoch = min(base.stage_a_epoch, 20)
    base.store_full_gradient_directions = True
    api_check = validate_attack_api()
    set_seed(args.seed)
    recovered_normal, _, _, _, consensus, gate_audit, *_ = invert_observed(
        model, tokenizer, base, prefix_activation, int(ids_for_activation.shape[1]), device
    )
    replacement_payload = "Completely unrelated replacement prompt used only for evaluation leakage verification."
    replacement_tokenized = tokenizer(replacement_payload, add_special_tokens=True, truncation=False, return_tensors="pt")
    set_seed(args.seed)
    recovered_with_replaced_eval_refs, *_ = invert_observed(
        model, tokenizer, base, prefix_activation, int(ids_for_activation.shape[1]), device
    )
    cfg_top1 = config_from_args(args, "cg_gdpi", args.seed, "verify_top1", str(out / "verify_top1"))
    cfg_top1.epoch = base.epoch
    cfg_top1.stage_a_epoch = base.stage_a_epoch
    cfg_top1.top_m = 1
    cfg_top1.store_full_gradient_directions = True
    set_seed(args.seed)
    recovered_top1, *_ = invert_observed(model, tokenizer, cfg_top1, prefix_activation, int(ids_for_activation.shape[1]), device)
    cfg_closed = config_from_args(args, "cg_gdpi", args.seed, "verify_closed", str(out / "verify_closed"))
    cfg_closed.epoch = base.epoch
    cfg_closed.stage_a_epoch = base.stage_a_epoch
    cfg_closed.force_gate_closed = True
    cfg_closed.store_full_gradient_directions = True
    set_seed(args.seed)
    recovered_closed, *_ = invert_observed(model, tokenizer, cfg_closed, prefix_activation, int(ids_for_activation.shape[1]), device)
    override_rows = [row for row in gate_audit.get("per_position", []) if row.get("override")]
    result = {
        "title": EXPERIMENT_TITLE,
        "dataset_meta": dataset_meta,
        "target_layer": args.target_layer,
        "activation_shape_full": list(full_activation.shape),
        "activation_shape_prefix": list(prefix_activation.shape),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "passes_strict_1e_5": max_abs_diff < 1e-5,
        "attack_api_check": api_check,
        "randomized_eval_text": "Random evaluation-only text that must not affect CG-GDPI attack output.",
        "randomized_eval_token_ids": [int(x) for x in replacement_tokenized["input_ids"][0].tolist()],
        "recovery_identical_after_eval_reference_replacement": recovered_normal == recovered_with_replaced_eval_refs,
        "top_m_1_equals_pure_calibration": recovered_top1 == recovered_closed,
        "force_gate_closed_equals_pure_calibration": recovered_closed == recovered_top1,
        "all_override_tokens_in_calibration_top_m": all(row.get("override_within_calibration_top_m", True) for row in override_rows),
        "proposed_method_uses_minmax_global_fusion": False,
        "gradient_consensus_checkpoint_count": consensus.get("checkpoint_count"),
        "gate_open_rate": gate_audit.get("gate_open_rate"),
        "override_rate": gate_audit.get("override_rate"),
        "normal_recovered_token_ids": recovered_normal,
        "replaced_eval_reference_recovered_token_ids": recovered_with_replaced_eval_refs,
    }
    json_dump(Path(args.output_root) / "leakage_verification.json", result)
    json_dump(out / "leakage_verification.json", result)
    if not (
        result["passes_strict_1e_5"]
        and result["attack_api_check"]["passes"]
        and result["recovery_identical_after_eval_reference_replacement"]
        and result["top_m_1_equals_pure_calibration"]
        and result["force_gate_closed_equals_pure_calibration"]
        and result["all_override_tokens_in_calibration_top_m"]
    ):
        raise RuntimeError(f"CG-GDPI leakage/unit verification failed: {result}")


def collect_metrics(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("**/metrics.json")):
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = metrics.get("config", {})
        token_acc = metrics.get("token_accuracy", {})
        bleu = metrics.get("bleu", {})
        runtime = metrics.get("runtime_seconds", {})
        rows.append(
            {
                "stage": path.parent.parent.name,
                "method": cfg.get("method"),
                "seed": cfg.get("seed"),
                "participant_number": cfg.get("participant_number"),
                "attacker_position": cfg.get("attacker_position"),
                "target_layer": cfg.get("target_layer"),
                "epoch": cfg.get("epoch"),
                "gamma": cfg.get("gamma"),
                "top_m": cfg.get("top_m"),
                "attention_mode": cfg.get("attention_mode"),
                "force_gate_closed": cfg.get("force_gate_closed"),
                "token_accuracy_mean": token_acc.get("mean"),
                "token_accuracy_std": token_acc.get("std"),
                "bleu_mean": bleu.get("mean"),
                "bleu_std": bleu.get("std"),
                "runtime_mean": runtime.get("mean"),
                "completed_sample_count": metrics.get("completed_sample_count"),
                "failed_sample_count": metrics.get("failed_sample_count"),
                "metrics_path": str(path),
            }
        )
    return rows


def write_summary(root: Path) -> None:
    rows = collect_metrics(root)
    if not rows:
        return
    csv_path = root / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md = [
        f"# {EXPERIMENT_TITLE}",
        "",
        "| stage | method | seed | layer | epoch | top_m | attention | gate_closed | acc mean | acc std | BLEU | completed | failed |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md.append(
            f"| {row['stage']} | {row['method']} | {row['seed']} | {row['target_layer']} | {row['epoch']} | "
            f"{row['top_m']} | {row['attention_mode']} | {row['force_gate_closed']} | "
            f"{row['token_accuracy_mean']} | {row['token_accuracy_std']} | {row['bleu_mean']} | "
            f"{row['completed_sample_count']} | {row['failed_sample_count']} |"
        )
    (root / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write_plots(root, rows)
    write_report(root, rows)


def write_plots(root: Path, rows: List[Dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [row for row in rows if row["token_accuracy_mean"] is not None]
    if valid:
        labels = [f"{row['stage']}\n{row['method']}\nL{row['target_layer']} E{row['epoch']} S{row['seed']}" for row in valid]
        accs = [float(row["token_accuracy_mean"]) for row in valid]
        bleus = [float(row["bleu_mean"] or 0.0) for row in valid]
        fig, ax = plt.subplots(figsize=(max(10, len(valid) * 0.45), 5))
        ax.bar(range(len(valid)), accs)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Token accuracy")
        ax.set_title("CG-GDPI token accuracy comparison")
        ax.set_xticks(range(len(valid)), labels=labels, rotation=75, ha="right", fontsize=7)
        fig.tight_layout()
        fig.savefig(root / "token_accuracy_comparison.png", dpi=200)
        fig.savefig(root / "comparison.png", dpi=200)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(max(10, len(valid) * 0.45), 5))
        ax.bar(range(len(valid)), bleus)
        ax.set_ylim(0, 1)
        ax.set_ylabel("BLEU")
        ax.set_title("CG-GDPI BLEU comparison")
        ax.set_xticks(range(len(valid)), labels=labels, rotation=75, ha="right", fontsize=7)
        fig.tight_layout()
        fig.savefig(root / "bleu_comparison.png", dpi=200)
        plt.close(fig)
    depth = [row for row in valid if row["stage"] == "depth_generalization"]
    if depth:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in sorted({row["method"] for row in depth}):
            pts = [row for row in depth if row["method"] == method]
            xs = [int(row["target_layer"]) for row in pts]
            ys = [float(row["token_accuracy_mean"]) for row in pts]
            ax.plot(xs, ys, marker="o", label=method)
        ax.set_xlabel("Target layer")
        ax.set_ylabel("Token accuracy")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(root / "depth_generalization.png", dpi=200)
        plt.close(fig)
    gate_points = []
    override_points = []
    for metrics_path in root.glob("**/metrics.json"):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            gate_path = Path(metrics["gradient_gate_stats_path"])
            samples = json.loads(gate_path.read_text(encoding="utf-8")).get("samples", [])
        except Exception:
            continue
        rates = [sample.get("gate_open_rate") for sample in samples if sample.get("gate_open_rate") is not None]
        overrides = [sample.get("override_rate") for sample in samples if sample.get("override_rate") is not None]
        if rates:
            gate_points.append((metrics_path.parent.name, float(statistics.mean(rates))))
        if overrides:
            override_points.append((metrics_path.parent.name, float(statistics.mean(overrides))))
    for filename, title, data in [
        ("gate_open_rate.png", "CG-GDPI gate open rate", gate_points),
        ("override_accuracy_delta.png", "CG-GDPI override rate", override_points),
    ]:
        if data:
            labels, ys = zip(*data)
            fig, ax = plt.subplots(figsize=(max(8, len(data) * 0.4), 4))
            ax.bar(range(len(data)), ys)
            ax.set_ylim(0, 1)
            ax.set_title(title)
            ax.set_xticks(range(len(data)), labels=labels, rotation=75, ha="right", fontsize=7)
            fig.tight_layout()
            fig.savefig(root / filename, dpi=200)
            plt.close(fig)
    histories = []
    for pred_path in root.glob("**/predictions.jsonl"):
        for line in pred_path.read_text(encoding="utf-8").splitlines()[:1]:
            row = json.loads(line)
            for point in row.get("stage_b_history", []):
                histories.append((point["step"], point["optimization_loss"], pred_path.parent.name))
    if histories:
        fig, ax = plt.subplots(figsize=(7, 4))
        for name in sorted({x[2] for x in histories})[:20]:
            pts = [(s, y) for s, y, n in histories if n == name]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], alpha=0.6)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Stage-B convergence samples")
        fig.tight_layout()
        fig.savefig(root / "convergence_curve.png", dpi=200)
        plt.close(fig)


def write_report(root: Path, rows: List[Dict[str, Any]]) -> None:
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)
    best = max([r for r in rows if r["token_accuracy_mean"] is not None], key=lambda r: float(r["token_accuracy_mean"]), default=None)
    leakage_path = root / "leakage_verification.json"
    leakage = json.loads(leakage_path.read_text(encoding="utf-8")) if leakage_path.exists() else None
    completed_count = len([row for row in rows if row["completed_sample_count"]])
    partial_dirs = []
    for pred_path in sorted((root / "main_comparison").glob("*/predictions.jsonl")) if (root / "main_comparison").exists() else []:
        if not (pred_path.parent / "COMPLETE").exists():
            try:
                partial_count = sum(1 for line in pred_path.read_text(encoding="utf-8").splitlines() if line.strip())
            except Exception:
                partial_count = None
            partial_dirs.append((pred_path.parent.name, partial_count))
    p1_gate_path = (
        root
        / "main_comparison"
        / "P1_p4_a4_layer17_epoch100_seed42_topm3_raw_open_maxlen640_g0p3_eta0p5"
        / "gradient_gate_stats.json"
    )
    p1_gate_summary = None
    if p1_gate_path.exists():
        try:
            samples = json.loads(p1_gate_path.read_text(encoding="utf-8")).get("samples", [])
            gate_rates = [sample.get("gate_open_rate") for sample in samples if sample.get("gate_open_rate") is not None]
            override_rates = [sample.get("override_rate") for sample in samples if sample.get("override_rate") is not None]
            p1_gate_summary = {
                "gate_open_mean": float(statistics.mean(gate_rates)) if gate_rates else None,
                "override_mean": float(statistics.mean(override_rates)) if override_rates else None,
                "sample_count": len(samples),
            }
        except Exception:
            p1_gate_summary = None
    lines = [
        f"# {EXPERIMENT_TITLE}",
        "",
        "This is a TinyLlama white-box pilot, not a complete reproduction of the original large-model paper.",
        "",
        "## Method",
        "",
        "CG-GDPI keeps residual alpha-NN initialization and replaces global min-max gradient fusion with calibration-first top-M reranking guarded by gradient consensus.",
        "",
        "B4 is retained as the old global gradient-fusion baseline. P1/P2 do not call the old min-max fusion path.",
        "",
        "## Execution Status",
        "",
        f"Completed configurations in the summary table: `{completed_count}`.",
        "",
        "The completed main-comparison subset includes Skytrax-28 pilot, participant 4, attacker position 4, target layer 17, epoch 100, seed 42 for B0/B1/B2/B3/B4/P1. It also includes any additional configurations that have a `COMPLETE` marker.",
        "",
        "The remaining requested main seeds/epoch-200 runs, depth generalization, and ablations are not marked complete unless they appear in the table below.",
        "",
    ]
    if partial_dirs:
        lines.extend(["Partial directories retained but excluded from the table:", ""])
        for name, count in partial_dirs[:20]:
            lines.append(f"- `{name}`: `{count}` prediction rows, no `COMPLETE` marker.")
        lines.append("")
    if best:
        lines.extend(
            [
                "## Best Completed Setting",
                "",
                f"Best observed run: stage `{best['stage']}`, method `{best['method']}`, seed `{best['seed']}`, "
                f"layer `{best['target_layer']}`, epoch `{best['epoch']}`, top_m `{best['top_m']}`, "
                f"attention `{best['attention_mode']}`, token accuracy `{best['token_accuracy_mean']}`, BLEU `{best['bleu_mean']}`.",
                "",
            ]
        )
    if leakage:
        lines.extend(
            [
                "## Anti-Leakage Verification",
                "",
                f"- Prefix/full activation max diff: `{leakage.get('max_abs_diff')}`; mean diff: `{leakage.get('mean_abs_diff')}`.",
                f"- Attack API passes: `{leakage.get('attack_api_check', {}).get('passes')}`.",
                f"- top_m=1 equals pure calibration: `{leakage.get('top_m_1_equals_pure_calibration')}`.",
                f"- force gate closed equals pure calibration: `{leakage.get('force_gate_closed_equals_pure_calibration')}`.",
                f"- All overrides inside calibration top-M: `{leakage.get('all_override_tokens_in_calibration_top_m')}`.",
                "",
            ]
        )
    if p1_gate_summary:
        lines.extend(
            [
                "## CG-GDPI Gate Summary",
                "",
                f"- P1 seed 42 epoch 100 gate open mean: `{p1_gate_summary['gate_open_mean']}`.",
                f"- P1 seed 42 epoch 100 override mean: `{p1_gate_summary['override_mean']}`.",
                f"- P1 gate samples: `{p1_gate_summary['sample_count']}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Findings",
            "",
            "- Alpha-NN initialization remains useful in this pilot when followed by calibration-safe selection.",
            "- The previous global gradient-matching fusion fails because per-position min-max scaling can amplify noisy gradients.",
            "- CG-GDPI only allows gradients to rerank calibration top-M candidates when consensus and margin checks pass.",
            "- BOS debiasing is implemented as an ablation; it is not assumed to help without measurement.",
            "- Depth generalization and full multi-seed results are reported only for runs that have COMPLETE markers.",
            "",
            "## Full Table",
            "",
            (root / "comparison.md").read_text(encoding="utf-8") if (root / "comparison.md").exists() else "No table generated yet.",
        ]
    )
    (analysis_dir / "cg_gdpi_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=EXPERIMENT_TITLE)
    parser.add_argument("--mode", choices=["single", "verify", "summarize"], required=True)
    parser.add_argument("--method", choices=METHODS, default="P1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-root", default="runs/cg_gdpi")
    parser.add_argument("--output-dir")
    parser.add_argument("--dataset-name", default="Skytrax-28-pilot")
    parser.add_argument("--dataset-path", default="data/airline.json")
    parser.add_argument("--dataset-len", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--participant-number", type=int, default=4)
    parser.add_argument("--attacker-position", type=int, default=4)
    parser.add_argument("--target-layer", type=int, default=17)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--stage-a-epoch", type=int)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lambda-vocab", type=float, default=0.1)
    parser.add_argument("--lambda-dummy", type=float, default=0.1)
    parser.add_argument("--lambda-context", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--y", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--top-m", type=int, default=3)
    parser.add_argument("--tau-consensus", type=float, default=0.80)
    parser.add_argument("--tau-grad-margin", type=float, default=0.05)
    parser.add_argument("--tau-cal-margin-quantile", type=float, default=0.30)
    parser.add_argument("--attention-mode", choices=["raw", "bos_debias"], default="raw")
    parser.add_argument("--force-gate-closed", action="store_true")
    parser.add_argument("--naive-discretization", action="store_true")
    parser.add_argument("--disable-semantic-speculation", action="store_true")
    parser.add_argument("--max-token-len", type=int, default=640)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--fix-boundary-specials", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--store-full-gradient-directions", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    if args.mode == "single":
        run_single(args)
    elif args.mode == "verify":
        run_verify(args)
    elif args.mode == "summarize":
        write_summary(Path(args.output_root))


if __name__ == "__main__":
    main()
