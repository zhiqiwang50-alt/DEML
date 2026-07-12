import argparse
import ast
import csv
import inspect
import json
import math
import os
import random
import re
import statistics
import time
import traceback
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask

from pia_alpha_gm import (
    alpha_nn_initialization,
    normalize_run_float,
    stage_a_dummy_proxy,
    summarize_rows,
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
    set_seed,
    text_from_ids,
    token_accuracy,
    token_texts,
)


EXPERIMENT_TITLE = "Server-Attention-Weighted Prompt Inversion (SAW-PIA) TinyLlama white-box pilot"
METHODS = ["B0", "B1", "B2", "B3", "P1", "P2", "P3", "original_pia_baseline", "dummy_init_existing", "alpha_nn_init_only", "attention_context_existing", "server_attn_weighted_last", "server_attn_weighted_mean", "alpha_init_plus_server_attn"]
METHOD_ALIASES = {
    "B0": "original_pia_baseline",
    "B1": "dummy_init_existing",
    "B2": "alpha_nn_init_only",
    "B3": "attention_context_existing",
    "P1": "server_attn_weighted_last",
    "P2": "server_attn_weighted_mean",
    "P3": "alpha_init_plus_server_attn",
}
SERVER_ATTN_METHODS = {"server_attn_weighted_last", "server_attn_weighted_mean", "alpha_init_plus_server_attn"}
DUMMY_METHODS = {"dummy_init_existing", "alpha_nn_init_only", "attention_context_existing", "alpha_init_plus_server_attn"}
ALPHA_METHODS = {"alpha_nn_init_only", "alpha_init_plus_server_attn"}


@dataclass
class SAWConfig:
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
    max_token_len: int
    grad_clip: float
    weight_source: str
    weight_floor: float
    residual_rollout: bool
    server_rollout_depth: str
    adaptive_discretization: bool
    semantic_speculation: bool
    local_files_only: bool
    fix_boundary_specials: bool = True


def canonical_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def row_normalize(mat: torch.Tensor) -> torch.Tensor:
    denom = mat.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return mat / denom


def residual_attention(attn: torch.Tensor, residual: bool = True) -> torch.Tensor:
    if attn.dim() != 2 or attn.shape[0] != attn.shape[1]:
        raise RuntimeError(f"attention must be [seq, seq], got {list(attn.shape)}")
    out = attn.float().clamp_min(0.0)
    if residual:
        out = out + torch.eye(out.shape[0], dtype=out.dtype, device=out.device)
    return row_normalize(out)


def rollout_from_attentions(attentions: Sequence[torch.Tensor], residual: bool = True) -> torch.Tensor:
    if not attentions:
        raise RuntimeError("server attention rollout needs at least one server layer")
    seq = attentions[0].shape[-1]
    rollout = torch.eye(seq, dtype=torch.float32, device=attentions[0].device)
    for attn in attentions:
        rollout = residual_attention(attn, residual=residual) @ rollout
    return row_normalize(rollout)


def token_weights_from_rollout(
    rollout: torch.Tensor,
    attention_mask: torch.Tensor,
    source: str,
    weight_floor: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if rollout.dim() != 2:
        raise RuntimeError(f"rollout must be [seq, seq], got {list(rollout.shape)}")
    valid = attention_mask[0].detach().bool()
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise RuntimeError("attention mask has no valid tokens")
    if source == "last_query":
        raw = rollout[int(valid_indices[-1].detach().cpu())].clone()
    elif source == "mean_query":
        raw = rollout[valid].mean(dim=0)
    elif source == "uniform":
        raw = torch.ones(rollout.shape[0], dtype=torch.float32, device=rollout.device)
    else:
        raise ValueError(f"unknown weight_source={source!r}")
    raw = raw.float().clamp_min(0.0)
    raw = raw * valid.float()
    raw_sum = raw.sum().clamp_min(1e-12)
    weights = raw.shape[0] * raw / raw_sum
    if weight_floor > 0:
        weights = torch.where(valid, torch.clamp(weights, min=float(weight_floor)), torch.zeros_like(weights))
        weights = weights.shape[0] * weights / weights.sum().clamp_min(1e-12)
    weights = weights * valid.float()
    stats = {
        "source": source,
        "weight_floor": float(weight_floor),
        "valid_token_count": int(valid.sum().detach().cpu()),
        "raw_sum": float(raw_sum.detach().cpu()),
        "mean": float(weights[valid].mean().detach().cpu()),
        "std": float(weights[valid].std(unbiased=False).detach().cpu()) if int(valid.sum()) > 1 else 0.0,
        "min": float(weights[valid].min().detach().cpu()),
        "max": float(weights[valid].max().detach().cpu()),
        "top_positions": [
            {"position": int(i), "weight": float(weights[i].detach().cpu())}
            for i in torch.topk(weights, min(10, weights.numel())).indices.detach().cpu().tolist()
        ],
    }
    return weights.detach(), stats


def weighted_activation_loss(pred: torch.Tensor, target: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    per_token = torch.mean((pred.float() - target.to(pred.device).float()) ** 2, dim=-1)
    if weights is None:
        return per_token.mean()
    w = weights.to(pred.device).float().unsqueeze(0)
    return (per_token * w).sum() / w.sum().clamp_min(1e-12)


def tensor_attention_stats(attn: torch.Tensor, layer_index: int) -> Dict[str, Any]:
    row_sums = attn.sum(dim=-1)
    upper = torch.triu(attn, diagonal=1).abs().max().item() if attn.shape[0] > 1 else 0.0
    return {
        "layer_index": int(layer_index),
        "shape": [int(attn.shape[0]), int(attn.shape[1])],
        "row_sum_min": float(row_sums.min().detach().cpu()),
        "row_sum_max": float(row_sums.max().detach().cpu()),
        "row_sum_mean": float(row_sums.mean().detach().cpu()),
        "upper_triangular_max": float(upper),
        "nonnegative_min": float(attn.min().detach().cpu()),
        "entropy_mean": float((-(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(dim=-1)).mean().detach().cpu()),
        "top_left_8x8": attn[:8, :8].detach().cpu().tolist(),
    }


def server_forward_with_attention(
    model: torch.nn.Module,
    start_layer: int,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    output_attentions: bool,
    rollout_depth: str = "all",
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, torch.Tensor]]]:
    batch, seq_len, _ = hidden_states.shape
    device = hidden_states.device
    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
    causal_mask = _prepare_4d_causal_attention_mask(
        attention_mask,
        (batch, seq_len),
        hidden_states,
        past_key_values_length=0,
    )
    layers = list(range(start_layer, len(model.model.layers)))
    if rollout_depth == "last2":
        attention_layers = set(layers[-2:])
    elif rollout_depth == "all":
        attention_layers = set(layers)
    else:
        raise ValueError(f"unknown server_rollout_depth={rollout_depth!r}")
    collected: List[Tuple[int, torch.Tensor]] = []
    x = hidden_states
    for layer_index in layers:
        want = output_attentions and layer_index in attention_layers
        layer_outputs = model.model.layers[layer_index](
            x,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=want,
            use_cache=False,
        )
        x = layer_outputs[0]
        if want:
            attn = layer_outputs[1]
            if attn is None:
                raise RuntimeError(f"server layer {layer_index} did not return attention")
            if attn.dim() != 4:
                raise RuntimeError(f"expected attention [batch, heads, seq, seq], got {list(attn.shape)}")
            collected.append((layer_index, attn.detach().float().mean(dim=1)[0]))
    final_hidden = model.model.norm(x)
    logits = model.lm_head(final_hidden)
    return final_hidden, logits, collected


def server_attention_bundle(
    model: torch.nn.Module,
    observed_activation: torch.Tensor,
    attention_mask: torch.Tensor,
    cfg: SAWConfig,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if cfg.weight_source == "uniform":
        seq = observed_activation.shape[1]
        rollout = torch.eye(seq, dtype=torch.float32, device=observed_activation.device)
        weights, weight_stats = token_weights_from_rollout(rollout, attention_mask, "uniform", 0.0)
        return weights, {"used": False, "reason": "uniform weight mode"}, {"rollout_shape": [seq, seq]}, weight_stats
    start = cfg.target_layer + 1
    if start >= len(model.model.layers):
        raise RuntimeError(f"target_layer={cfg.target_layer} leaves no server-side layers")
    with torch.no_grad():
        _hidden, _logits, collected = server_forward_with_attention(
            model,
            start,
            observed_activation.to(dtype=next(model.parameters()).dtype),
            attention_mask,
            output_attentions=True,
            rollout_depth=cfg.server_rollout_depth,
        )
    attentions = [attn for _idx, attn in collected]
    rollout = rollout_from_attentions(attentions, residual=cfg.residual_rollout)
    weights, weight_stats = token_weights_from_rollout(rollout, attention_mask, cfg.weight_source, cfg.weight_floor)
    attention_stats = {
        "used": True,
        "target_layer_output_state": f"H^({cfg.target_layer}) is output of 0-based block {cfg.target_layer}",
        "server_start_layer": start,
        "server_layers": [int(idx) for idx, _attn in collected],
        "rollout_depth": cfg.server_rollout_depth,
        "residual_rollout": bool(cfg.residual_rollout),
        "layer_stats": [tensor_attention_stats(attn, idx) for idx, attn in collected],
        "dummy_attention_used": False,
    }
    rollout_stats = {
        "rollout_shape": [int(rollout.shape[0]), int(rollout.shape[1])],
        "row_sum_min": float(rollout.sum(dim=-1).min().detach().cpu()),
        "row_sum_max": float(rollout.sum(dim=-1).max().detach().cpu()),
        "row_sum_mean": float(rollout.sum(dim=-1).mean().detach().cpu()),
        "upper_triangular_max": float(torch.triu(rollout, diagonal=1).abs().max().detach().cpu()) if rollout.shape[0] > 1 else 0.0,
        "nonnegative_min": float(rollout.min().detach().cpu()),
        "top_left_16x16": rollout[:16, :16].detach().cpu().tolist(),
    }
    if abs(rollout_stats["row_sum_min"] - 1.0) > 2e-3 or abs(rollout_stats["row_sum_max"] - 1.0) > 2e-3:
        raise RuntimeError(f"rollout rows are not normalized: {rollout_stats}")
    if rollout_stats["nonnegative_min"] < -1e-6:
        raise RuntimeError(f"rollout contains negative values: {rollout_stats}")
    if rollout_stats["upper_triangular_max"] > 2e-3:
        raise RuntimeError(f"rollout violates causal mask: {rollout_stats}")
    return weights, attention_stats, rollout_stats, weight_stats


def stage_b_optimize(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: SAWConfig,
    observed_activation: torch.Tensor,
    seq_len: int,
    device: torch.device,
    fixed_public: Dict[int, int],
    init_embeds: torch.Tensor,
    server_weights: Optional[torch.Tensor],
    a_context: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Any], List[Dict[str, float]]]:
    embed_layer = model.get_input_embeddings()
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    fixed_embeds, fixed_positions, _ = fixed_embedding_tensor(embed_layer, fixed_public, seq_len, device)
    left, right = embedding_bounds(embed_layer.weight)
    left = left.to(device)
    right = right.to(device)
    z = init_embeds.detach().clone().to(device=device, dtype=torch.float32).requires_grad_(True)
    optimizer = torch.optim.AdamW([z], lr=cfg.lr)
    target = observed_activation.detach()
    target_ctx = context_projection(a_context, target) if a_context is not None and cfg.lambda_context > 0 else None
    history: List[Dict[str, float]] = []
    final: Dict[str, Any] = {}
    for step in range(max(1, cfg.epoch)):
        enforce_embedding_constraints(z, fixed_positions, fixed_embeds, left, right)
        hidden = capture_prefix_activation(
            model,
            cfg.target_layer,
            inputs_embeds=z.to(dtype=embed_layer.weight.dtype),
            attention_mask=attention_mask,
        )
        act_loss = weighted_activation_loss(hidden, target, server_weights)
        unweighted = weighted_activation_loss(hidden, target, None)
        vocab_loss = nearest_embedding_loss(variable_view(z, fixed_positions), embed_layer.weight)
        if target_ctx is not None:
            ctx_loss = F.mse_loss(context_projection(a_context, hidden), target_ctx.to(hidden.device))
        else:
            ctx_loss = torch.tensor(0.0, device=device)
        total = act_loss + cfg.lambda_vocab * vocab_loss + cfg.lambda_context * ctx_loss
        cosine = F.cosine_similarity(hidden.float(), target.to(hidden.device).float(), dim=-1).mean()
        if torch.isnan(total) or torch.isnan(cosine):
            raise RuntimeError(f"NaN in Stage B at step {step + 1}")
        optimizer.zero_grad()
        total.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([z], cfg.grad_clip)
        optimizer.step()
        final = {
            "activation_loss": float(act_loss.detach().cpu()),
            "unweighted_activation_loss": float(unweighted.detach().cpu()),
            "vocab_loss": float(vocab_loss.detach().cpu()),
            "context_loss": float(ctx_loss.detach().cpu()),
            "optimization_loss": float(total.detach().cpu()),
            "cosine_similarity": float(cosine.detach().cpu()),
        }
        if (step + 1) % max(1, min(50, cfg.epoch)) == 0 or step == cfg.epoch - 1:
            history.append({"step": step + 1, **final})
            print(
                f"method={cfg.method} step={step + 1} act={final['activation_loss']:.6f} "
                f"base_act={final['unweighted_activation_loss']:.6f} vocab={final['vocab_loss']:.6f} "
                f"ctx={final['context_loss']:.6f} total={final['optimization_loss']:.6f} "
                f"cosine={final['cosine_similarity']:.6f}",
                flush=True,
            )
    enforce_embedding_constraints(z, fixed_positions, fixed_embeds, left, right)
    return z.detach().float(), final, history


def invert_observed(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: SAWConfig,
    observed_activation: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> Tuple[List[int], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, float]], List[Dict[str, float]], Optional[float]]:
    method = canonical_method(cfg.method)
    cfg.method = method
    embed_layer = model.get_input_embeddings()
    fixed_public = inferred_boundary_special_tokens(tokenizer, seq_len) if cfg.fix_boundary_specials else {}
    stage_a_history: List[Dict[str, float]] = []
    attention_stats: Dict[str, Any] = {"method": method, "server_attention_used": False, "dummy_attention_used": False}
    rollout_stats: Dict[str, Any] = {}
    weight_stats: Dict[str, Any] = {}
    init_audit: Dict[str, Any] = {"method": method, "uses_ground_truth_for_initialization": False}
    dummy_embeds: Optional[torch.Tensor] = None
    a_dummy: Optional[torch.Tensor] = None
    if method in DUMMY_METHODS:
        dummy_embeds, a_dummy, dummy_stats, stage_a_history = stage_a_dummy_proxy(
            model, tokenizer, cfg, observed_activation, seq_len, device, fixed_public
        )
        init_audit["dummy_stage_a_available"] = True
        init_audit["dummy_attention_proxy_used"] = method in {"alpha_nn_init_only", "attention_context_existing", "alpha_init_plus_server_attn"}
        attention_stats["dummy_proxy_stats"] = dummy_stats
    if method == "original_pia_baseline" or method in SERVER_ATTN_METHODS:
        init_embeds = random_public_embeddings(tokenizer, embed_layer, seq_len, device, fixed_public)
        init_audit["init"] = "random_public_embeddings"
    elif method == "dummy_init_existing":
        init_embeds = dummy_embeds
        init_audit["init"] = "stage_a_dummy_embeddings"
    elif method in ALPHA_METHODS:
        if a_dummy is None:
            raise RuntimeError("alpha initialization requires dummy attention proxy")
        init_embeds, _init_ids, _h_alpha, init_audit = alpha_nn_initialization(
            model, tokenizer, cfg, observed_activation, a_dummy
        )
        init_audit["init"] = "alpha_nn_from_dummy_attention_proxy"
    elif method == "attention_context_existing":
        init_embeds = random_public_embeddings(tokenizer, embed_layer, seq_len, device, fixed_public)
        init_audit["init"] = "random_public_embeddings_with_dummy_context_loss"
    else:
        raise ValueError(f"unknown method={method!r}")
    if init_embeds is None:
        raise RuntimeError(f"no initialization for method={method}")
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    server_weights: Optional[torch.Tensor] = None
    if method in SERVER_ATTN_METHODS:
        source = "mean_query" if method == "server_attn_weighted_mean" else "last_query"
        cfg.weight_source = source
        server_weights, server_stats, rollout_stats, weight_stats = server_attention_bundle(
            model, observed_activation, attention_mask, cfg
        )
        attention_stats.update(server_stats)
    use_dummy_context = method == "attention_context_existing"
    z, losses, stage_b_history = stage_b_optimize(
        model,
        tokenizer,
        cfg,
        observed_activation,
        seq_len,
        device,
        fixed_public,
        init_embeds,
        server_weights,
        a_context=a_dummy if use_dummy_context else None,
    )
    embed_sets = embedding_candidates(z, embed_layer.weight, max(1, cfg.top_k_embedding))
    recovered_ids = naive_discretization(z, embed_layer.weight)
    for pos, token_id in fixed_public.items():
        if 0 <= pos < len(recovered_ids):
            recovered_ids[pos] = int(token_id)
    calibration_score = None
    if cfg.adaptive_discretization:
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
    return recovered_ids, losses, attention_stats, rollout_stats, weight_stats, init_audit, stage_a_history, stage_b_history, calibration_score


def validate_attack_api() -> Dict[str, Any]:
    sig = inspect.signature(invert_observed)
    names = list(sig.parameters)
    banned = ["reference", "original", "input_ids", "token_ids", "prompt", "text", "ground"]
    signature_hits = [name for name in names if any(item in name for item in banned)]
    checked = [invert_observed, server_attention_bundle, server_forward_with_attention, stage_b_optimize]
    banned_globals = {"original_ids", "original_tokens", "reference_embedding", "reference_tokens", "ground_truth_ids", "dummy_embeds_for_server_attention"}
    ast_hits: List[Dict[str, str]] = []
    dummy_hits: List[str] = []
    for fn in checked:
        source = inspect.getsource(fn)
        tree = ast.parse(source)
        if fn in {server_attention_bundle, server_forward_with_attention} and ("stage_a_dummy" in source or "a_dummy" in source):
            dummy_hits.append(fn.__name__)
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
        "server_attention_dummy_proxy_hits": dummy_hits,
        "passes": len(signature_hits) == 0 and len(ast_hits) == 0 and len(dummy_hits) == 0,
    }


def load_model_and_data(cfg: SAWConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer, model = load_tinyllama(dtype, local_files_only=cfg.local_files_only)
    model.to(device)
    if len(model.model.layers) != TOTAL_BLOCKS:
        raise RuntimeError(f"Expected {TOTAL_BLOCKS} TinyLlama blocks, got {len(model.model.layers)}")
    return tokenizer, model, device


def run_one_config(cfg: SAWConfig, resume: bool) -> Dict[str, Any]:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "COMPLETE"
    if resume and done.exists():
        return json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    json_dump(out / "config.json", {"title": EXPERIMENT_TITLE, **cfg.__dict__})
    tokenizer, model, device = load_model_and_data(cfg)
    prompts, dataset_meta = load_dataset_prompts(cfg.dataset_name, cfg.dataset_path, cfg.dataset_len, cfg.seed)
    json_dump(out / "dataset_meta.json", dataset_meta)
    predictions_path = out / "predictions.jsonl"
    failures_path = out / "failures.jsonl"
    if not resume:
        for path in [predictions_path, failures_path]:
            if path.exists():
                path.unlink()
    completed = set()
    rows: List[Dict[str, Any]] = []
    if resume and predictions_path.exists():
        by_id = {}
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed.add(int(row["prompt_id"]))
                by_id[int(row["prompt_id"])] = row
        rows = [by_id[k] for k in sorted(by_id)]
    failures: List[Dict[str, Any]] = []
    if resume and failures_path.exists():
        for line in failures_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                failures.append(json.loads(line))
    all_attention_stats = []
    all_rollout_stats = []
    all_weight_stats = []
    loss_breakdown = []
    for prompt_id, prompt in enumerate(prompts):
        if prompt_id in completed:
            continue
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start = time.time()
            tokenized = tokenizer(prompt, add_special_tokens=True, truncation=False, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            if input_ids.shape[1] > cfg.max_token_len:
                raise RuntimeError(f"prompt_id={prompt_id} has {input_ids.shape[1]} tokens > max_token_len={cfg.max_token_len}")
            original_ids = [int(x) for x in input_ids[0].detach().cpu().tolist()]
            with torch.no_grad():
                observed = capture_prefix_activation(
                    model,
                    cfg.target_layer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).detach()
            preflight = {
                "model_name": MODEL_NAME,
                "block_count": len(model.model.layers),
                "target_layer_index": cfg.target_layer,
                "h_obs_semantics": f"output of 0-based TinyLlama block {cfg.target_layer}",
                "server_start_layer": cfg.target_layer + 1,
                "server_layer_indices": list(range(cfg.target_layer + 1, len(model.model.layers))),
                "inverted_block_count": cfg.inverted_block_count,
                "observed_activation_shape": list(observed.shape),
                "sequence_length": int(input_ids.shape[1]),
                "top_k_embedding": cfg.top_k_embedding,
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "cpu"),
                "attack_receives_ground_truth": False,
            }
            print(f"preflight={json.dumps(preflight, ensure_ascii=True)}", flush=True)
            recovered_ids, losses, attn_stats, rollout_stats, weight_stats, init_audit, stage_a_history, stage_b_history, calibration_score = invert_observed(
                model, tokenizer, cfg, observed, int(input_ids.shape[1]), device
            )
            recovered_text = text_from_ids(tokenizer, recovered_ids)
            original_text = text_from_ids(tokenizer, original_ids)
            elapsed = time.time() - start
            row = {
                "title": EXPERIMENT_TITLE,
                "method": cfg.method,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "original_text": original_text,
                "recovered_text": recovered_text,
                "original_token_ids": original_ids,
                "recovered_token_ids": recovered_ids,
                "original_tokens": token_texts(tokenizer, original_ids),
                "recovered_tokens": token_texts(tokenizer, recovered_ids),
                "token_accuracy": token_accuracy(tokenizer, original_ids, recovered_ids),
                "bleu": bleu_score(tokenizer, original_ids, recovered_ids),
                "nerr": optional_nerr(original_text, recovered_text),
                "optimization_loss": losses["optimization_loss"],
                "activation_loss": losses["activation_loss"],
                "unweighted_activation_loss": losses["unweighted_activation_loss"],
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
                "prompt_token_count": len(original_ids),
                "peak_gpu_memory_mb": peak_memory_mb(),
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "cpu"),
                "top_k_embedding": cfg.top_k_embedding,
                "top_y_semantic": cfg.top_y_semantic,
                "preflight": preflight,
                "init_audit": init_audit,
                "stage_a_history": stage_a_history,
                "stage_b_history": stage_b_history,
                "recovery_uses_ground_truth_tokens": False,
            }
            jsonl_append(predictions_path, row)
            rows.append(row)
            all_attention_stats.append({"prompt_id": prompt_id, **attn_stats})
            all_rollout_stats.append({"prompt_id": prompt_id, **rollout_stats})
            all_weight_stats.append({"prompt_id": prompt_id, **weight_stats})
            loss_breakdown.append({"prompt_id": prompt_id, "stage_b_history": stage_b_history, "final": losses})
            print(
                f"method={cfg.method} prompt_id={prompt_id} token_accuracy={row['token_accuracy']:.6f} "
                f"bleu={row['bleu']:.6f} elapsed={elapsed:.2f}s",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "method": cfg.method,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "seed": cfg.seed,
                "target_layer": cfg.target_layer,
            }
            jsonl_append(failures_path, failure)
            failures.append(failure)
            print(f"failure={json.dumps(failure, ensure_ascii=True)}", flush=True)
    metrics = summarize_rows(rows)
    metrics.update(
        {
            "title": EXPERIMENT_TITLE,
            "method": cfg.method,
            "seed": cfg.seed,
            "target_layer": cfg.target_layer,
            "completed_sample_count": len(rows),
            "failed_sample_count": len(failures),
            "dataset_meta": dataset_meta,
            "top_k_embedding": cfg.top_k_embedding,
            "top_y_semantic": cfg.top_y_semantic,
            "output_dir": cfg.output_dir,
        }
    )
    json_dump(out / "metrics.json", metrics)
    json_dump(out / "server_attention_stats.json", {"samples": all_attention_stats})
    json_dump(out / "attention_rollout.json", {"samples": all_rollout_stats})
    json_dump(out / "token_weight_stats.json", {"samples": all_weight_stats})
    json_dump(out / "loss_breakdown.json", {"samples": loss_breakdown})
    done.write_text("complete\n", encoding="utf-8")
    return metrics


def verify_no_leakage(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = config_from_args(args, args.method, args.seed, "leakage_verification", str(Path(args.output_root) / "leakage_verification" / "sample"))
    cfg.dataset_len = 1
    tokenizer, model, device = load_model_and_data(cfg)
    prompts, dataset_meta = load_dataset_prompts(cfg.dataset_name, cfg.dataset_path, 1, cfg.seed)
    prompt = prompts[0]
    tokenized = tokenizer(prompt, add_special_tokens=True, truncation=False, return_tensors="pt")
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    with torch.no_grad():
        observed = capture_prefix_activation(model, cfg.target_layer, input_ids=input_ids, attention_mask=attention_mask).detach()
        full = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        full_boundary = full.hidden_states[cfg.target_layer + 1].detach()
        server_final, server_logits, collected = server_forward_with_attention(
            model,
            cfg.target_layer + 1,
            observed.to(dtype=next(model.parameters()).dtype),
            attention_mask,
            output_attentions=True,
            rollout_depth=cfg.server_rollout_depth,
        )
    boundary_diff = float((observed.float() - full_boundary.float()).abs().max().detach().cpu())
    final_diff = float((server_final.float() - full.hidden_states[-1].float()).abs().max().detach().cpu())
    logits_diff = float((server_logits.float() - full.logits.float()).abs().max().detach().cpu())
    weights, attn_stats, rollout_stats, weight_stats = server_attention_bundle(model, observed, attention_mask, cfg)
    uniform = torch.ones_like(weights)
    pred = observed + torch.randn_like(observed) * 0.01
    uniform_loss = float(weighted_activation_loss(pred, observed, uniform).detach().cpu())
    base_loss = float(weighted_activation_loss(pred, observed, None).detach().cpu())
    api_check = validate_attack_api()
    result = {
        "title": EXPERIMENT_TITLE,
        "dataset_meta": dataset_meta,
        "target_layer": cfg.target_layer,
        "h_obs_semantics": f"H_obs is output of 0-based block {cfg.target_layer}; server starts at block {cfg.target_layer + 1}",
        "server_layers": [idx for idx, _attn in collected],
        "activation_shape": list(observed.shape),
        "boundary_vs_full_hidden_max_abs_diff": boundary_diff,
        "server_final_hidden_vs_full_max_abs_diff": final_diff,
        "server_logits_vs_full_max_abs_diff": logits_diff,
        "manual_server_forward_matches_full": final_diff < 5e-3 and logits_diff < 5e-3,
        "attack_api_check": api_check,
        "attention_stats": attn_stats,
        "rollout_stats": rollout_stats,
        "weight_stats": weight_stats,
        "uniform_weight_loss": uniform_loss,
        "unweighted_loss": base_loss,
        "uniform_weight_equals_unweighted_loss": abs(uniform_loss - base_loss) < 1e-8,
        "p1_p2_do_not_use_dummy_attention": True,
    }
    out = Path(args.output_root) / "leakage_verification"
    out.mkdir(parents=True, exist_ok=True)
    json_dump(out / "leakage_verification.json", result)
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return result


def download_skytrax_150(args: argparse.Namespace) -> None:
    out = Path(args.dataset_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force_download:
        print(f"dataset exists: {out}", flush=True)
        return
    slugs = [
        "british-airways",
        "emirates",
        "qatar-airways",
        "singapore-airlines",
        "lufthansa",
        "air-france",
        "klm-royal-dutch-airlines",
        "turkish-airlines",
        "american-airlines",
        "united-airlines",
        "delta-air-lines",
        "ryanair",
        "easyjet",
    ]
    reviews: List[str] = []
    seen = set()
    pattern = re.compile(r'<div[^>]+class="[^"]*text_content[^"]*"[^>]*>(.*?)</div>', re.S | re.I)
    tag_re = re.compile(r"<[^>]+>")
    for slug in slugs:
        for page in range(1, 8):
            if len(reviews) >= args.dataset_size:
                break
            url = f"https://www.airlinequality.com/airline-reviews/{slug}/page/{page}/?sortby=post_date%3ADesc&pagesize=100"
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urlopen(req, timeout=20) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")
            except Exception as exc:
                print(f"download warning {url}: {exc}", flush=True)
                continue
            for match in pattern.findall(html):
                text = unescape(tag_re.sub(" ", match))
                text = re.sub(r"\s+", " ", text).strip()
                text = re.sub(r"^(✅ Trip Verified|Not Verified|Trip Verified)\s*\|\s*", "", text).strip()
                if len(text) < 80 or not text.isascii():
                    continue
                key = text.lower()[:200]
                if key in seen:
                    continue
                seen.add(key)
                reviews.append(text)
                if len(reviews) >= args.dataset_size:
                    break
        if len(reviews) >= args.dataset_size:
            break
    if len(reviews) < args.dataset_size:
        raise RuntimeError(f"Only downloaded {len(reviews)} Skytrax reviews, requested {args.dataset_size}")
    json_dump(out, reviews[: args.dataset_size])
    meta = {
        "source": "airlinequality.com airline review pages (Skytrax-branded public review pages)",
        "sample_count": len(reviews[: args.dataset_size]),
        "slugs": slugs,
        "ascii_filter": True,
        "min_length": 80,
    }
    json_dump(out.with_suffix(".meta.json"), meta)
    print(json.dumps({"dataset_out": str(out), **meta}, ensure_ascii=True), flush=True)


def config_from_args(args: argparse.Namespace, method: str, seed: int, run_name: str, output_dir: str) -> SAWConfig:
    method = canonical_method(method)
    inverted, target = ATTACKER_MAP[args.participant_number][args.attacker_position]
    if args.target_layer is not None:
        target = args.target_layer
        inverted = target + 1
    return SAWConfig(
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
        stage_a_epoch=args.stage_a_epoch if args.stage_a_epoch is not None else args.epoch,
        lr=args.lr,
        lambda_vocab=args.lambda_vocab,
        lambda_dummy=args.lambda_dummy,
        lambda_context=args.lambda_context,
        top_k_embedding=args.k,
        top_y_semantic=args.y,
        gamma=args.gamma,
        max_token_len=args.max_token_len,
        grad_clip=args.grad_clip,
        weight_source=args.weight_source,
        weight_floor=args.weight_floor,
        residual_rollout=not args.no_residual_rollout,
        server_rollout_depth=args.server_rollout_depth,
        adaptive_discretization=not args.naive_discretization,
        semantic_speculation=not args.disable_semantic_speculation,
        local_files_only=args.local_files_only,
    )


def run_single(args: argparse.Namespace) -> Dict[str, Any]:
    method = canonical_method(args.method)
    out = args.output_dir or str(Path(args.output_root) / method / f"{method}_seed{args.seed}_layer{args.target_layer}_k{args.k}")
    cfg = config_from_args(args, method, args.seed, Path(out).name, out)
    return run_one_config(cfg, resume=args.resume)


def run_multi_layer(args: argparse.Namespace) -> None:
    for layer in args.target_layers:
        for method in args.methods:
            method_name = canonical_method(method)
            run_name = f"{method_name}_seed{args.seed}_layer{layer}_epoch{args.epoch}_k{args.k}"
            out = Path(args.output_root) / "multi_layer" / run_name
            layer_args = argparse.Namespace(**vars(args))
            layer_args.target_layer = layer
            cfg = config_from_args(layer_args, method_name, args.seed, run_name, str(out))
            run_one_config(cfg, resume=args.resume)
    write_summary(Path(args.output_root))


def write_summary(root: Path) -> None:
    rows = []
    for metrics_path in sorted(root.glob("**/metrics.json")):
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        config = {}
        config_path = metrics_path.with_name("config.json")
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                config = {}
        rows.append(
            {
                "stage": metrics_path.relative_to(root).parts[0] if len(metrics_path.relative_to(root).parts) > 1 else "",
                "method": m.get("method"),
                "seed": m.get("seed"),
                "target_layer": m.get("target_layer"),
                "epoch": m.get("epoch", config.get("epoch")),
                "max_token_len": config.get("max_token_len"),
                "top_k_embedding": m.get("top_k_embedding"),
                "top_y_semantic": m.get("top_y_semantic"),
                "token_accuracy_mean": m.get("token_accuracy", {}).get("mean"),
                "token_accuracy_std": m.get("token_accuracy", {}).get("std"),
                "bleu_mean": m.get("bleu", {}).get("mean"),
                "bleu_std": m.get("bleu", {}).get("std"),
                "completed_sample_count": m.get("completed_sample_count"),
                "failed_sample_count": m.get("failed_sample_count"),
                "metrics_path": str(metrics_path),
            }
        )
    if not rows:
        return
    fields = list(rows[0].keys())
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (root / "comparison.md").open("w", encoding="utf-8") as f:
        f.write("| method | seed | layer | epoch | max_len | k | token_acc | bleu | n | fail |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['method']} | {r['seed']} | {r['target_layer']} | {r['epoch']} | {r['max_token_len']} | {r['top_k_embedding']} | "
                f"{r['token_accuracy_mean']} | {r['bleu_mean']} | {r['completed_sample_count']} | {r['failed_sample_count']} |\n"
            )
    write_report(root, rows)


def write_report(root: Path, rows: List[Dict[str, Any]]) -> None:
    scored_rows = [r for r in rows if r["token_accuracy_mean"] is not None]
    best = max(scored_rows, key=lambda r: float(r["token_accuracy_mean"]), default=None)
    baseline_by_layer = {
        int(r["target_layer"]): r
        for r in scored_rows
        if r.get("method") == "original_pia_baseline" and r.get("target_layer") is not None
    }

    def as_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def fmt(value: Any, digits: int = 4) -> str:
        number = as_float(value)
        if number is None:
            return "NA"
        return f"{number:.{digits}f}"

    result_lines = [
        "| method | layer | token_acc | delta_vs_B0 | BLEU | n | fail |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(scored_rows, key=lambda x: (int(x.get("target_layer") or -1), str(x.get("method")))):
        layer = int(r["target_layer"])
        baseline_acc = as_float(baseline_by_layer.get(layer, {}).get("token_accuracy_mean"))
        acc = as_float(r.get("token_accuracy_mean"))
        delta = None if acc is None or baseline_acc is None else acc - baseline_acc
        result_lines.append(
            f"| {r['method']} | {layer} | {fmt(acc)} | {fmt(delta, 4)} | "
            f"{fmt(r.get('bleu_mean'))} | {r.get('completed_sample_count')} | {r.get('failed_sample_count')} |"
        )

    failed_total = sum(int(r.get("failed_sample_count") or 0) for r in rows)
    completed_total = sum(int(r.get("completed_sample_count") or 0) for r in rows)
    epochs = sorted({str(r.get("epoch")) for r in rows if r.get("epoch") is not None})
    max_lens = sorted({str(r.get("max_token_len")) for r in rows if r.get("max_token_len") is not None})
    lines = [
        "# Server Attention PIA Report",
        "",
        "This is a TinyLlama white-box pilot for server-side attention rollout weighted activation matching.",
        "",
        "## Layer Mapping",
        "",
        "`capture_prefix_activation(target_layer=L)` returns the output of 0-based TinyLlama block `L`. Therefore `target_layer=17` corresponds to `H^(17)`, and server-side layers start from block `18`.",
        "",
        "## Top-k Setting",
        "",
        "All requested runs use embedding candidate `--k 1` (top-1). Semantic candidates are controlled separately by `--y`.",
        "",
        "## Experimental Setup",
        "",
        f"- Output root: `{root}`",
        "- Dataset: `data/skytrax_150.json` (150 Skytrax review prompts)",
        "- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`",
        "- Layers: 11, 17, 19",
        "- Methods: B0 original baseline, P1 last-query server rollout, P2 mean-query server rollout, P3 alpha init plus server rollout",
        f"- Epochs: {', '.join(epochs) if epochs else 'NA'}",
        f"- Max token length: {', '.join(max_lens) if max_lens else 'NA'}",
        f"- Completed prompt-runs: {completed_total}; failed prompt-runs: {failed_total}",
        "",
        "## Results",
        "",
        *result_lines,
        "",
        "## Main Observation",
        "",
        "On the full Skytrax-150 top-1 pilot, the original PIA baseline remains the best row. Server-attention weighting does not produce a stable improvement over B0.",
        "",
        "P2 (`server_attn_weighted_mean`) is the least damaging server-attention variant, especially at layer 19, but it is still below the same-layer baseline. P1 (`last_query`) is strongly hurt at layers 17 and 19. P3 shows that adding alpha initialization to the server-attention loss does not rescue the weighted objective in this run.",
        "",
        "## Best Observed Row",
        "",
        json.dumps(best, indent=2, ensure_ascii=False) if best else "No completed rows.",
        "",
        "## Caution",
        "",
        "These are single-seed epoch-20 pilot results. They support a negative or inconclusive conclusion for the current SAW-PIA weighting design, not a stable improvement claim.",
    ]
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "server_attn_pia_report.md").write_text("\n".join(lines), encoding="utf-8")
    Path("analysis").mkdir(exist_ok=True)
    Path("analysis/server_attn_pia_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_design() -> None:
    Path("analysis").mkdir(exist_ok=True)
    design = """# Server-Attention-Weighted Prompt Inversion Design

## Layer mapping

TinyLlama uses 0-based block indices. `capture_prefix_activation(model, target_layer=L, ...)` manually executes blocks `0..L` and returns the output of block `L`. Thus `target_layer=17` is `H^(17)`, and server-side layers are `18..21`.

## Method

SAW-PIA feeds `H_obs` through public server-side layers, extracts each server layer attention, averages heads, applies `RowNorm(I + A)`, and computes rollout in forward order:

`R = A_tilde_last @ ... @ A_tilde_first`

For causal TinyLlama, token importance defaults to the last valid query row: `w_i = R[last_valid_position, i]`, normalized to mean 1 with optional floor.

The activation loss becomes a weighted per-token loss:

`L = sum_i w_i mean_h((F_prefix(z)_i - H_obs_i)^2) / sum_i w_i + lambda_vocab L_vocab`

P1/P2 do not use dummy embeddings or dummy attention. P3 is an explicit contrast that combines alpha initialization with server attention loss.
"""
    Path("analysis/server_attn_pia_design.md").write_text(design, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["download-skytrax", "verify", "single", "multi-layer", "summarize", "write-design"], default="single")
    parser.add_argument("--method", choices=METHODS, default="P1")
    parser.add_argument("--methods", choices=METHODS, nargs="*", default=["B0", "P1", "P3"])
    parser.add_argument("--output-root", default="runs/server_attn_pia")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-name", default="Skytrax-150")
    parser.add_argument("--dataset-path", default="data/skytrax_150.json")
    parser.add_argument("--dataset-len", type=int, default=150)
    parser.add_argument("--dataset-out", default="data/skytrax_150.json")
    parser.add_argument("--dataset-size", type=int, default=150)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    parser.add_argument("--participant-number", type=int, default=4)
    parser.add_argument("--attacker-position", type=int, default=4)
    parser.add_argument("--target-layer", type=int, default=17)
    parser.add_argument("--target-layers", type=int, nargs="*", default=[11, 17, 19])
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--stage-a-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lambda-vocab", type=float, default=0.1)
    parser.add_argument("--lambda-dummy", type=float, default=0.1)
    parser.add_argument("--lambda-context", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--y", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--weight-source", choices=["last_query", "mean_query", "uniform"], default="last_query")
    parser.add_argument("--weight-floor", type=float, default=0.05)
    parser.add_argument("--no-residual-rollout", action="store_true")
    parser.add_argument("--server-rollout-depth", choices=["all", "last2"], default="all")
    parser.add_argument("--max-token-len", type=int, default=896)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--naive-discretization", action="store_true")
    parser.add_argument("--disable-semantic-speculation", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "download-skytrax":
        download_skytrax_150(args)
    elif args.mode == "verify":
        write_design()
        verify_no_leakage(args)
    elif args.mode == "single":
        write_design()
        run_single(args)
        write_summary(Path(args.output_root))
    elif args.mode == "multi-layer":
        write_design()
        run_multi_layer(args)
    elif args.mode == "summarize":
        write_design()
        write_summary(Path(args.output_root))
    elif args.mode == "write-design":
        write_design()


if __name__ == "__main__":
    main()
