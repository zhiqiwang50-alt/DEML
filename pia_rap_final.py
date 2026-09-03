import argparse
import ast
import csv
import hashlib
import inspect
import json
import math
import os
import random
import re
import statistics
import time
import textwrap
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("TRANSFORMERS_CACHE", "/data/zhiqi/hf_cache/transformers")

import torch
import torch.nn.functional as F
try:
    from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask
except ImportError:  # Local lightweight test environments may use a newer transformers API.
    def _prepare_4d_causal_attention_mask(*_args: Any, **_kwargs: Any) -> torch.Tensor:
        raise RuntimeError("_prepare_4d_causal_attention_mask is unavailable in this transformers build")

from pia_attention_guided import enforce_embedding_constraints, fixed_embedding_tensor, random_public_embeddings, variable_view
from pia_shared_masking import build_variable_mask, uniform_all_token_activation_loss, variable_mask_audit_payload, weighted_variable_activation_loss
from pia_tinyllama import (
    ATTACKER_MAP,
    MODEL_NAME,
    TOTAL_BLOCKS,
    bleu_score,
    capture_prefix_activation,
    embedding_bounds,
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


EXPERIMENT_TITLE = "RAP-FINAL: Top-1 Relational Sparse Repair with Discrete Acceptance"
OUTPUT_ROOT_DEFAULT = "runs/rap_final"
METHOD_ALIASES = {
    "B0": "original_pia_baseline",
    "B0V": "variable_only_uniform",
    "SPARSE_FIXED": "sparse_fixed",
    "REL_ORDER": "rel_order",
    "RAP_CONT_GATE": "rap_cont_gate",
    "RAP_FINAL": "rap_final",
    "SHUFFLED_REL": "shuffled_rel",
    "PIA_FULL_REFERENCE": "pia_full_reference",
    # Historical names are kept only for appendix/reproducibility; final RAP does
    # not use relational min-cut partitioning.
    "SPARSE_PATCH": "sparse_fixed",
    "REL_ORDER_ONLY": "rel_order",
    "RAP_V2": "rap_cont_gate",
}
METHODS = [*METHOD_ALIASES.keys(), *sorted(set(METHOD_ALIASES.values()))]
BASELINE_METHODS = {"original_pia_baseline", "variable_only_uniform"}
SPARSE_METHODS = {"sparse_fixed", "rel_order", "rap_cont_gate", "rap_final", "shuffled_rel"}
RELATIONAL_ORDER_METHODS = {"rel_order", "rap_cont_gate", "rap_final", "shuffled_rel"}
CONTINUOUS_REL_GATE_METHODS = {"rap_cont_gate"}
DISCRETE_REL_GATE_METHODS = {"rap_final", "shuffled_rel"}
DELTA_CUT_ONLY_METHODS = {"sparse_fixed", "rel_order"}
FINAL_EVAL_METHODS = ["B0", "B0V", "SPARSE_FIXED", "RAP_FINAL", "SHUFFLED_REL"]


@dataclass
class RAPFinalConfig:
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
    lr: float
    lambda_vocab: float
    top_k_embedding: int
    top_y_semantic: int
    max_token_len: int
    grad_clip: float
    uncertainty_fraction: float
    patch_size: int
    min_patch_size: int
    max_patch_size: int
    top_r: int
    max_repair_passes: int
    eps_cut: float
    eps_rel: float
    eps_rel_ratio: Optional[float]
    uncertainty_detector: str
    adaptive_discretization: bool
    semantic_speculation: bool
    local_files_only: bool
    fix_boundary_specials: bool = True


RAPV2Config = RAPFinalConfig


def canonical_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def require_strict_top1(cfg: RAPFinalConfig) -> None:
    if int(cfg.top_k_embedding) != 1:
        raise ValueError("RAP-FINAL requires strict Top-1: K must be 1")
    if int(cfg.top_y_semantic) != 0:
        raise ValueError("RAP-FINAL requires strict Top-1: Y must be 0")
    if bool(cfg.semantic_speculation):
        raise ValueError("RAP-FINAL requires semantic_speculation=False")
    if bool(cfg.adaptive_discretization):
        raise ValueError("RAP-FINAL requires activation calibration disabled")
    if cfg.uncertainty_detector not in {"margin", "joint"}:
        raise ValueError("uncertainty_detector must be margin or joint")


def special_token_semantic_note() -> str:
    return (
        "fixed_public positions such as position 0 are public framing positions. "
        "Without token ids in the attack API, token_id_based_special_mask_available=false. "
        "last_valid_position is only a public sequence boundary, not a token-id EOS judgment."
    )


def _ast_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _ast_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def validate_method_integrity(checked: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    checked = list(checked or [invert_observed, run_sparse_repair, optimize_sparse_candidate, build_relational_graph, top_r_relational_destinations])
    dummy_terms = {"dummy", "dummy_attention", "alpha_init"}
    scalar_call_terms = {
        "weighted_activation_loss",
        "build_mts_token_weights",
        "build_attention_scale_alpha",
        "attention_weighted_residual_loss",
        "attention_scaled_activation_loss",
    }
    dummy_hits: List[Dict[str, str]] = []
    scalar_hits: List[Dict[str, str]] = []
    for fn in checked:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Name, ast.Attribute)):
                name = _ast_name(node)
                leaf = name.split(".")[-1]
                if leaf in dummy_terms:
                    dummy_hits.append({"function": fn.__name__, "name": name})
            if isinstance(node, ast.Call):
                call_name = _ast_name(node.func)
                leaf = call_name.split(".")[-1]
                if leaf in scalar_call_terms:
                    scalar_hits.append({"function": fn.__name__, "call": call_name})
    return {
        "checked_functions": [fn.__name__ for fn in checked],
        "dummy_reference_hits": dummy_hits,
        "scalar_attention_call_hits": scalar_hits,
        "passes": not dummy_hits and not scalar_hits,
    }


def method_integrity_audit() -> Dict[str, Any]:
    check = validate_method_integrity()
    return {
        "uses_dummy_attention": bool(check["dummy_reference_hits"]),
        "uses_scalar_attention_weighted_activation_loss": bool(check["scalar_attention_call_hits"]),
        "method_integrity_check": check,
        "attention_role": "relational routing, shuffled control, and discrete repair validation",
    }


def has_nan(value: Any) -> bool:
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    if isinstance(value, dict):
        return any(has_nan(v) for v in value.values())
    if isinstance(value, list):
        return any(has_nan(v) for v in value)
    return False


def mean_std(values: Sequence[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"mean": None, "std": None}
    return {"mean": float(statistics.mean(vals)), "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0}


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "token_accuracy": mean_std([row["token_accuracy"] for row in rows]),
        "bleu": mean_std([row["bleu"] for row in rows]),
        "nerr": mean_std([row["nerr"] for row in rows if row.get("nerr") is not None]),
        "prompt_exact_match": mean_std([row["prompt_exact_match"] for row in rows]),
        "runtime_seconds": mean_std([row["runtime_seconds"] for row in rows]),
        "peak_gpu_memory_mb": mean_std([row["peak_gpu_memory_mb"] for row in rows if row.get("peak_gpu_memory_mb") is not None]),
        "sample_count": len(rows),
    }


def stage1_b0v_cut_loss(pred: torch.Tensor, target: torch.Tensor, variable_mask: torch.Tensor) -> torch.Tensor:
    return weighted_variable_activation_loss(pred, target, variable_mask, None)


def contribution_matrix(attention: torch.Tensor, value_norm: torch.Tensor) -> torch.Tensor:
    if attention.dim() != 3:
        raise RuntimeError(f"attention must be [heads, seq, seq], got {list(attention.shape)}")
    if value_norm.dim() != 2:
        raise RuntimeError(f"value_norm must be [heads, seq], got {list(value_norm.shape)}")
    if attention.shape[0] != value_norm.shape[0] or attention.shape[-1] != value_norm.shape[-1]:
        raise RuntimeError("attention and value_norm are not aligned")
    return (attention.float().clamp_min(0.0) * value_norm.float().clamp_min(0.0).unsqueeze(1)).mean(dim=0)


def legal_variable_causal_mask(score: torch.Tensor, variable_mask: torch.Tensor) -> torch.Tensor:
    variable = variable_mask.to(score.device).bool()
    return torch.tril(torch.ones_like(score, dtype=torch.bool), diagonal=-1) & variable[:, None] & variable[None, :]


def normalize_layer_score(score: torch.Tensor, variable_mask: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, Dict[str, Any]]:
    legal = legal_variable_causal_mask(score, variable_mask)
    legal_values = score.detach().float()[legal]
    if int(legal_values.numel()) == 0:
        scale = torch.tensor(1.0, dtype=torch.float32, device=score.device)
        normalized = score.detach().float()
        raw_mean = 0.0
        raw_max = 0.0
        normalized_mean = 0.0
    else:
        mu = legal_values.mean()
        scale = mu.clamp_min(float(eps))
        normalized = score.detach().float() / scale
        raw_mean = float(mu.detach().cpu())
        raw_max = float(legal_values.max().detach().cpu())
        normalized_mean = float(normalized[legal].mean().detach().cpu())
    return normalized.detach(), {
        "raw_mean": raw_mean,
        "normalized_mean": normalized_mean,
        "raw_max": raw_max,
        "scale_factor": float(scale.detach().cpu()),
        "legal_variable_causal_edge_count": int(legal.sum().detach().cpu()),
    }


def shuffle_relational_score(score: torch.Tensor, variable_mask: torch.Tensor, seed: int) -> torch.Tensor:
    legal = legal_variable_causal_mask(score, variable_mask)
    values = score.detach().float()[legal]
    out = score.detach().float().clone()
    if int(values.numel()) <= 1:
        return out
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(int(values.numel()), generator=generator)
    out[legal] = values.detach().cpu()[perm].to(out.device)
    return out


def shuffle_layer_scores(
    layer_scores: Dict[int, torch.Tensor],
    aggregate: torch.Tensor,
    variable_mask: torch.Tensor,
    seed: int,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, Dict[str, Any]]:
    shuffled: Dict[int, torch.Tensor] = {}
    for offset, (layer, score) in enumerate(sorted(layer_scores.items())):
        shuffled[int(layer)] = shuffle_relational_score(score, variable_mask, int(seed) + 1009 * offset)
    shuffled_aggregate = torch.stack(list(shuffled.values()), dim=0).mean(dim=0) if shuffled else shuffle_relational_score(aggregate, variable_mask, seed)
    legal = legal_variable_causal_mask(aggregate, variable_mask)
    return shuffled, shuffled_aggregate.detach(), {
        "shuffled_relational_control": True,
        "legal_edge_count": int(legal.sum().detach().cpu()),
        "preserves_legal_causal_variable_edge_count": True,
        "preserves_compute_budget": True,
    }


def rank01(values: torch.Tensor, descending: bool = False) -> torch.Tensor:
    flat = values.detach().float()
    finite = torch.isfinite(flat)
    out = torch.zeros_like(flat, dtype=torch.float32)
    idx = torch.nonzero(finite, as_tuple=False).flatten()
    if int(idx.numel()) == 0:
        return out
    order = idx[torch.argsort(flat[idx], descending=descending)]
    if int(order.numel()) == 1:
        out[order[0]] = 1.0
        return out
    ranks = torch.arange(int(order.numel()), dtype=torch.float32, device=flat.device)
    out[order] = 1.0 - ranks / max(1, int(order.numel()) - 1)
    return out


def activation_residual_by_position(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = pred.detach().float() - target.to(pred.device).detach().float()
    return torch.mean(diff.pow(2), dim=-1).squeeze(0)


def build_uncertainty_score(
    margins: torch.Tensor,
    variable_mask: torch.Tensor,
    fraction: float,
    activation_residual: Optional[torch.Tensor] = None,
    detector: str = "margin",
) -> Tuple[torch.Tensor, List[int], Dict[str, Any]]:
    variable = variable_mask.to(margins.device).bool()
    positions = torch.nonzero(variable, as_tuple=False).flatten()
    scores = torch.zeros_like(margins, dtype=torch.float32)
    if positions.numel() == 0:
        return scores, [], {"uncertain_fraction": float(fraction), "uncertainty_detector": detector, "selected_uncertain_positions": []}
    vals = margins[positions].float()
    if detector == "margin":
        vmax = vals.max()
        vmin = vals.min()
        if float((vmax - vmin).detach().cpu()) <= 1e-12:
            scores[positions] = 1.0
        else:
            scores[positions] = (vmax - vals) / (vmax - vmin).clamp_min(1e-12)
        order = torch.argsort(margins[positions], descending=False)
    elif detector == "joint":
        if activation_residual is None:
            raise ValueError("joint uncertainty detector needs activation_residual")
        inverse_margin_rank = rank01(1.0 / (margins.float() + 1e-8), descending=True)
        residual_rank = rank01(activation_residual.to(margins.device).float(), descending=True)
        scores = torch.sqrt(torch.clamp(inverse_margin_rank, min=0.0) * torch.clamp(residual_rank, min=0.0))
        scores = torch.where(variable, scores, torch.zeros_like(scores))
        order = torch.argsort(scores[positions], descending=True)
    else:
        raise ValueError(f"unknown uncertainty detector: {detector}")
    count = max(1, min(int(positions.numel()), int(math.ceil(float(fraction) * int(positions.numel())))))
    chosen = positions[order[:count]]
    return scores.detach(), [int(x) for x in chosen.detach().cpu().tolist()], {
        "uncertain_fraction": float(fraction),
        "uncertainty_detector": detector,
        "variable_count": int(positions.numel()),
        "uncertain_token_count": count,
        "selected_uncertain_positions": [int(x) for x in chosen.detach().cpu().tolist()],
        "margin_mean": float(vals.mean().detach().cpu()),
        "margin_min": float(vals.min().detach().cpu()),
        "margin_max": float(vals.max().detach().cpu()),
        "uncertainty_score_mean": float(scores[positions].mean().detach().cpu()),
    }


def token_margins(z: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
    weight = embed_weight.detach().float().to(z.device)
    flat = z.squeeze(0).detach().float()
    distances = torch.cdist(flat, weight, p=2.0)
    top2 = torch.topk(distances, k=2, largest=False, dim=-1).values
    return (top2[:, 1] - top2[:, 0]).detach()


def patch_update_positions(patch_positions: Sequence[int], uncertain_positions: Set[int]) -> List[int]:
    return [int(pos) for pos in patch_positions if int(pos) in uncertain_positions]


def sparse_update_audit(
    updated_positions: Sequence[int],
    patch_positions: Sequence[int],
    uncertain_positions: Set[int],
    fixed_public: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    fixed = set(int(x) for x in (fixed_public or {}))
    updated = [int(pos) for pos in updated_positions if int(pos) not in fixed]
    frozen = [int(pos) for pos in patch_positions if int(pos) not in set(updated)]
    return {
        "updated_position_count": len(updated),
        "frozen_position_count": len(frozen),
        "updated_positions": updated,
        "frozen_positions": frozen,
        "updated_positions_subset_of_uncertain": set(updated).issubset(set(int(x) for x in uncertain_positions)),
    }


def apply_sparse_update(base: torch.Tensor, updated_positions: Sequence[int], updated_values: torch.Tensor) -> torch.Tensor:
    out = base.detach().clone()
    if updated_positions:
        out[:, list(updated_positions), :] = updated_values.to(out.device).unsqueeze(0)
    return out


def patch_cut_cost(score: torch.Tensor, t: int) -> float:
    if t <= 0 or t >= score.shape[0]:
        return float("inf")
    return float(score[t:, :t].sum().detach().cpu())


def partition_patches(seq_len: int, variable_mask: torch.Tensor, aggregate_score: torch.Tensor, patch_size: int, min_patch: int, max_patch: int) -> List[List[int]]:
    variable = variable_mask.detach().cpu().bool()
    patches: List[List[int]] = []
    start = 0
    while start < seq_len:
        lo = min(seq_len, start + min_patch)
        hi = min(seq_len, start + max_patch)
        if lo >= seq_len:
            end = seq_len
        else:
            candidates = list(range(lo, hi + 1))
            end = min(candidates, key=lambda t: patch_cut_cost(aggregate_score, t))
        patch = [pos for pos in range(start, end) if bool(variable[pos])]
        if patch:
            patches.append(patch)
        start = max(end, start + 1)
    return patches


def fixed_size_patches(seq_len: int, variable_mask: torch.Tensor, patch_size: int) -> List[List[int]]:
    variable = variable_mask.detach().cpu().bool()
    size = max(1, int(patch_size))
    patches: List[List[int]] = []
    for start in range(0, int(seq_len), size):
        end = min(int(seq_len), start + size)
        patch = [pos for pos in range(start, end) if bool(variable[pos])]
        if patch:
            patches.append(patch)
    return patches


def identity_aggregate_score(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.eye(int(seq_len), dtype=torch.float32, device=device)


def build_patch_plan(
    method: str,
    seq_len: int,
    variable_mask: torch.Tensor,
    relational_aggregate_score: torch.Tensor,
    uncertain_positions: Set[int],
    uncertainty_scores: torch.Tensor,
    patch_size: int,
    min_patch: int,
    max_patch: int,
) -> Dict[str, Any]:
    method = canonical_method(method)
    identity = identity_aggregate_score(seq_len, relational_aggregate_score.device)
    priority_score = relational_aggregate_score if method in RELATIONAL_ORDER_METHODS else identity
    patches = fixed_size_patches(seq_len, variable_mask, patch_size)
    priority_rows = compute_patch_priorities(patches, uncertain_positions, uncertainty_scores, priority_score, variable_mask)
    ordered_rows = sorted(priority_rows, key=lambda row: row["priority"], reverse=True) if method in RELATIONAL_ORDER_METHODS else list(priority_rows)
    return {
        "method": method,
        "patches": patches,
        "priority_rows": priority_rows,
        "ordered_rows": ordered_rows,
        "patching_rule": f"fixed_size_{int(patch_size)}_from_sequence_start_with_tail_leftover",
        "partition_uses_relational_graph": False,
        "ordering_uses_relational_graph": method in RELATIONAL_ORDER_METHODS,
        "historical_relational_min_cut_partition_used": False,
    }


def top_r_relational_destinations(
    layer_scores: Dict[int, torch.Tensor],
    uncertain_positions: Sequence[int],
    variable_mask: torch.Tensor,
    top_r: int,
) -> Dict[int, List[int]]:
    variable = variable_mask.detach().bool()
    result: Dict[int, List[int]] = {}
    for k in [int(x) for x in uncertain_positions]:
        selected: List[int] = []
        for score in layer_scores.values():
            col = score[:, k].detach().float().clone()
            legal = variable.to(col.device).clone()
            legal[: k + 1] = False
            col = torch.where(legal, col, torch.full_like(col, -float("inf")))
            finite = torch.isfinite(col)
            if int(finite.sum().detach().cpu()) == 0:
                continue
            vals, idx = torch.topk(col, k=min(int(top_r), int(finite.sum().detach().cpu())))
            for q, v in zip(idx.detach().cpu().tolist(), vals.detach().cpu().tolist()):
                if math.isfinite(float(v)) and int(q) > k and int(q) not in selected:
                    selected.append(int(q))
        result[k] = sorted(selected)
    return result


def compute_patch_priorities(
    patches: Sequence[Sequence[int]],
    uncertain_positions: Set[int],
    uncertainty_scores: torch.Tensor,
    aggregate_score: torch.Tensor,
    variable_mask: torch.Tensor,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    variable = variable_mask.to(aggregate_score.device).bool()
    for patch_id, patch in enumerate(patches):
        patch = [int(pos) for pos in patch]
        up = patch_update_positions(patch, uncertain_positions)
        if not up:
            rows.append({
                "patch_id": patch_id,
                "positions": patch,
                "uncertain_positions": [],
                "uncertainty_score": 0.0,
                "relation_score": 0.0,
                "priority": 0.0,
                "processed": False,
                "skipped": True,
            })
            continue
        uncertainty = float(uncertainty_scores[up].mean().detach().cpu())
        relation = float(aggregate_score[:, up][variable].mean().detach().cpu()) if int(variable.sum().detach().cpu()) else 0.0
        rows.append({
            "patch_id": patch_id,
            "positions": patch,
            "uncertain_positions": up,
            "uncertainty_score": uncertainty,
            "relation_score": relation,
            "priority": uncertainty * relation,
            "processed": False,
            "skipped": False,
        })
    max_relation = max([row["relation_score"] for row in rows], default=0.0)
    if max_relation > 0:
        for row in rows:
            row["normalized_relation_score"] = row["relation_score"] / max_relation
            row["priority"] = row["uncertainty_score"] * row["normalized_relation_score"]
    else:
        for row in rows:
            row["normalized_relation_score"] = 0.0
    return rows


def accept_candidate(delta_cut: float, delta_rel: float, eps_cut: float, eps_rel: float) -> Tuple[bool, str]:
    if not math.isfinite(delta_cut) or not math.isfinite(delta_rel):
        return False, "nan_or_inf_delta"
    if delta_cut >= -float(eps_cut):
        return False, "cut_not_improved"
    if delta_rel > float(eps_rel):
        return False, "relational_worse"
    return True, "accepted"


def rollback_if_rejected(before: torch.Tensor, candidate: torch.Tensor, accepted: bool) -> torch.Tensor:
    return candidate.detach().clone() if accepted else before.detach().clone()


def should_early_stop(accepted_this_pass: int) -> bool:
    return int(accepted_this_pass) == 0


def early_stop_summary(accepted_by_pass: Sequence[int], accepted_total: int) -> Dict[str, Any]:
    triggered = False
    triggered_pass = None
    for index, count in enumerate(accepted_by_pass, start=1):
        if should_early_stop(int(count)):
            triggered = True
            triggered_pass = index
            break
    return {
        "early_stop": int(accepted_total) == 0,
        "early_stop_triggered": triggered,
        "early_stop_pass": triggered_pass,
        "accepted_total": int(accepted_total),
        "accepted_by_pass": [int(x) for x in accepted_by_pass],
    }


def prompt_exact_match_for_variable(original_ids: Sequence[int], recovered_ids: Sequence[int], variable_rows: Sequence[Dict[str, Any]]) -> float:
    positions = [int(row["position"]) for row in variable_rows if row.get("variable_mask")]
    if not positions:
        return 0.0
    return 1.0 if all(pos < len(original_ids) and pos < len(recovered_ids) and int(original_ids[pos]) == int(recovered_ids[pos]) for pos in positions) else 0.0


def value_norms_for_layer(layer: torch.nn.Module, hidden_states: torch.Tensor, attn_heads: int) -> torch.Tensor:
    value_states = layer.self_attn.v_proj(hidden_states)
    batch, seq_len, hidden_dim = value_states.shape
    kv_heads = int(getattr(layer.self_attn, "num_key_value_heads", getattr(layer.self_attn, "num_heads", attn_heads)))
    head_dim = hidden_dim // kv_heads
    value_states = value_states.view(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
    norms = torch.linalg.vector_norm(value_states.float(), dim=-1)[0]
    if norms.shape[0] != attn_heads:
        if attn_heads % norms.shape[0] != 0:
            raise RuntimeError("cannot align value heads with attention heads")
        norms = norms.repeat_interleave(attn_heads // norms.shape[0], dim=0)
    return norms


def server_forward_evidence(
    model: torch.nn.Module,
    start_layer: int,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    collect_attention: bool,
    detach: bool,
) -> Tuple[List[Tuple[int, torch.Tensor]], List[Tuple[int, torch.Tensor]], List[Tuple[int, torch.Tensor]]]:
    batch, seq_len, _ = hidden_states.shape
    device = hidden_states.device
    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
    causal_mask = _prepare_4d_causal_attention_mask(attention_mask, (batch, seq_len), hidden_states, past_key_values_length=0)
    x = hidden_states
    states: List[Tuple[int, torch.Tensor]] = []
    attentions: List[Tuple[int, torch.Tensor]] = []
    value_norms: List[Tuple[int, torch.Tensor]] = []
    for layer_index in range(start_layer, len(model.model.layers)):
        layer = model.model.layers[layer_index]
        if collect_attention:
            value_norms.append((layer_index, value_norms_for_layer(layer, x, int(layer.self_attn.num_heads))))
        layer_outputs = layer(
            x,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=collect_attention,
            use_cache=False,
        )
        x = layer_outputs[0]
        states.append((layer_index, x.detach().float() if detach else x.float()))
        if collect_attention:
            attn = layer_outputs[1]
            if attn is None or attn.dim() != 4:
                raise RuntimeError(f"server layer {layer_index} did not return attention")
            attentions.append((layer_index, attn[0].detach().float()))
    return states, attentions, value_norms


def build_relational_graph(
    attentions: Sequence[Tuple[int, torch.Tensor]],
    value_norms: Sequence[Tuple[int, torch.Tensor]],
    variable_mask: torch.Tensor,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, Dict[str, Any]]:
    values = {int(idx): val for idx, val in value_norms}
    layer_scores: Dict[int, torch.Tensor] = {}
    stats = []
    for layer_index, attn in attentions:
        raw_score = contribution_matrix(attn, values[int(layer_index)])
        score, scale_stats = normalize_layer_score(raw_score, variable_mask)
        layer_scores[int(layer_index)] = score.detach()
        stats.append({
            "layer": int(layer_index),
            "shape": [int(score.shape[0]), int(score.shape[1])],
            "upper_triangular_max": float(torch.triu(score, diagonal=1).abs().max().detach().cpu()) if score.shape[0] > 1 else 0.0,
            "mean": float(score.mean().detach().cpu()),
            "max": float(score.max().detach().cpu()),
            **scale_stats,
        })
    if not layer_scores:
        raise RuntimeError("relational graph needs at least one server layer")
    aggregate = torch.stack(list(layer_scores.values()), dim=0).mean(dim=0)
    return layer_scores, aggregate.detach(), {
        "formula": "S_l[q,k] = mean_h A_lh[q,k] * ||V_lh[k]||_2; S_norm_l = S_l / (mean_legal_variable_causal_edges(S_l) + eps); S = mean_l S_norm_l",
        "server_layers": [int(idx) for idx in layer_scores],
        "layer_count": len(layer_scores),
        "layer_stats": stats,
        "uses_layerwise_relational_scale_normalization": True,
        "uses_scalar_attention_weighted_activation_loss": False,
    }


def global_cut_loss(model: torch.nn.Module, cfg: RAPV2Config, z: torch.Tensor, observed: torch.Tensor, attention_mask: torch.Tensor, variable_mask: torch.Tensor) -> torch.Tensor:
    hidden = capture_prefix_activation(model, cfg.target_layer, inputs_embeds=z, attention_mask=attention_mask)
    return stage1_b0v_cut_loss(hidden, observed, variable_mask)


def optimize_stage1(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: RAPV2Config,
    observed: torch.Tensor,
    seq_len: int,
    device: torch.device,
    fixed_public: Dict[int, int],
    variable_mask: torch.Tensor,
    loss_mode: str,
) -> Tuple[torch.Tensor, Dict[str, Any], List[Dict[str, Any]]]:
    embed_layer = model.get_input_embeddings()
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    fixed_embeds, fixed_positions, _ids = fixed_embedding_tensor(embed_layer, fixed_public, seq_len, device)
    left, right = embedding_bounds(embed_layer.weight)
    left = left.to(device)
    right = right.to(device)
    z = random_public_embeddings(tokenizer, embed_layer, seq_len, device, fixed_public).detach().clone().float().requires_grad_(True)
    opt = torch.optim.AdamW([z], lr=float(cfg.lr))
    history: List[Dict[str, Any]] = []
    final: Dict[str, Any] = {}
    for step in range(max(1, int(cfg.epoch))):
        enforce_embedding_constraints(z, fixed_positions, fixed_embeds, left, right)
        hidden = capture_prefix_activation(model, cfg.target_layer, inputs_embeds=z.to(dtype=embed_layer.weight.dtype), attention_mask=attention_mask)
        all_loss = uniform_all_token_activation_loss(hidden, observed, attention_mask)
        var_loss = stage1_b0v_cut_loss(hidden, observed, variable_mask)
        cut = all_loss if loss_mode == "all_valid_uniform" else var_loss
        vocab = nearest_embedding_loss(variable_view(z, fixed_positions), embed_layer.weight)
        total = cut + float(cfg.lambda_vocab) * vocab
        cosine = F.cosine_similarity(hidden.float(), observed.to(hidden.device).float(), dim=-1).mean()
        if torch.isnan(total) or torch.isnan(cosine):
            raise RuntimeError(f"NaN in Stage1 at step {step + 1}")
        opt.zero_grad()
        total.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([z], float(cfg.grad_clip))
        opt.step()
        final = {
            "step": step + 1,
            "activation_loss": float(cut.detach().cpu()),
            "all_token_uniform_loss": float(all_loss.detach().cpu()),
            "variable_uniform_loss": float(var_loss.detach().cpu()),
            "vocab_loss": float(vocab.detach().cpu()),
            "optimization_loss": float(total.detach().cpu()),
            "cosine_similarity": float(cosine.detach().cpu()),
            "loss_mode": loss_mode,
        }
        if (step + 1) % max(1, min(50, int(cfg.epoch))) == 0 or step + 1 == int(cfg.epoch):
            history.append(dict(final))
            print(
                f"method={cfg.method} stage=stage1 step={step+1} act={final['activation_loss']:.6f} "
                f"vocab={final['vocab_loss']:.6f} total={final['optimization_loss']:.6f} cosine={final['cosine_similarity']:.6f}",
                flush=True,
            )
    enforce_embedding_constraints(z, fixed_positions, fixed_embeds, left, right)
    return z.detach().float(), final, history


def relational_discrepancy(
    pred_states: Sequence[Tuple[int, torch.Tensor]],
    obs_states: Sequence[Tuple[int, torch.Tensor]],
    destinations_by_token: Dict[int, List[int]],
    updated_positions: Sequence[int],
) -> torch.Tensor:
    obs = {int(idx): state for idx, state in obs_states}
    terms: List[torch.Tensor] = []
    dests = sorted({q for k in updated_positions for q in destinations_by_token.get(int(k), [])})
    if not dests:
        return pred_states[0][1].sum() * 0.0
    for layer_index, pred in pred_states:
        if int(layer_index) not in obs:
            continue
        valid_dests = [q for q in dests if q < pred.shape[1]]
        if not valid_dests:
            continue
        diff = pred[:, valid_dests, :].float() - obs[int(layer_index)][:, valid_dests, :].to(pred.device).float()
        terms.append(torch.mean(diff ** 2))
    if not terms:
        return pred_states[0][1].sum() * 0.0
    return torch.stack(terms).mean()


def ids_from_embeddings(z: torch.Tensor, embed_weight: torch.Tensor, fixed_public: Dict[int, int]) -> List[int]:
    ids = naive_discretization(z.detach().float(), embed_weight)
    for pos, token_id in fixed_public.items():
        if 0 <= int(pos) < len(ids):
            ids[int(pos)] = int(token_id)
    return [int(x) for x in ids]


def embeddings_from_ids(embed_layer: torch.nn.Module, ids: Sequence[int], device: torch.device) -> torch.Tensor:
    tensor = torch.tensor([int(x) for x in ids], dtype=torch.long, device=device).unsqueeze(0)
    return embed_layer(tensor).detach().float()


def variable_token_sequence_changed(current_ids: Sequence[int], candidate_ids: Sequence[int], variable_mask: torch.Tensor) -> bool:
    variable = variable_mask.detach().cpu().bool()
    for pos, is_variable in enumerate(variable.tolist()):
        if bool(is_variable) and pos < len(current_ids) and pos < len(candidate_ids) and int(current_ids[pos]) != int(candidate_ids[pos]):
            return True
    return False


def discrete_accept_candidate(
    model: torch.nn.Module,
    cfg: RAPFinalConfig,
    current_z: torch.Tensor,
    candidate_z: torch.Tensor,
    observed: torch.Tensor,
    attention_mask: torch.Tensor,
    variable_mask: torch.Tensor,
    fixed_public: Dict[int, int],
    obs_states: Optional[Sequence[Tuple[int, torch.Tensor]]],
    destinations: Dict[int, List[int]],
    update_positions: Sequence[int],
    delta_cut_continuous: float,
) -> Tuple[bool, str, torch.Tensor, Dict[str, Any]]:
    embed_layer = model.get_input_embeddings()
    current_ids = ids_from_embeddings(current_z, embed_layer.weight, fixed_public)
    candidate_ids = ids_from_embeddings(candidate_z, embed_layer.weight, fixed_public)
    if not math.isfinite(float(delta_cut_continuous)) or float(delta_cut_continuous) >= -float(cfg.eps_cut):
        return False, "no_continuous_improvement", current_z.detach().clone(), {
            "discrete_token_changed": variable_token_sequence_changed(current_ids, candidate_ids, variable_mask),
            "delta_cut_continuous": float(delta_cut_continuous),
        }
    if not variable_token_sequence_changed(current_ids, candidate_ids, variable_mask):
        return False, "no_discrete_change", current_z.detach().clone(), {
            "discrete_token_changed": False,
            "delta_cut_continuous": float(delta_cut_continuous),
        }
    device = current_z.device
    current_disc = embeddings_from_ids(embed_layer, current_ids, device)
    candidate_disc = embeddings_from_ids(embed_layer, candidate_ids, device)
    with torch.no_grad():
        current_hidden = capture_prefix_activation(model, cfg.target_layer, inputs_embeds=current_disc.to(dtype=embed_layer.weight.dtype), attention_mask=attention_mask)
        candidate_hidden = capture_prefix_activation(model, cfg.target_layer, inputs_embeds=candidate_disc.to(dtype=embed_layer.weight.dtype), attention_mask=attention_mask)
        current_cut = float(stage1_b0v_cut_loss(current_hidden, observed, variable_mask).detach().cpu())
        candidate_cut = float(stage1_b0v_cut_loss(candidate_hidden, observed, variable_mask).detach().cpu())
        delta_cut_disc = candidate_cut - current_cut
        current_rel = 0.0
        candidate_rel = 0.0
        if destinations:
            current_states, _a, _v = server_forward_evidence(model, cfg.target_layer + 1, current_hidden.to(dtype=embed_layer.weight.dtype), attention_mask, False, False)
            candidate_states, _a, _v = server_forward_evidence(model, cfg.target_layer + 1, candidate_hidden.to(dtype=embed_layer.weight.dtype), attention_mask, False, False)
            current_rel = float(relational_discrepancy(current_states, obs_states or [], destinations, update_positions).detach().cpu())
            candidate_rel = float(relational_discrepancy(candidate_states, obs_states or [], destinations, update_positions).detach().cpu())
        delta_rel_disc = candidate_rel - current_rel
    stats = {
        "discrete_token_changed": True,
        "delta_cut_continuous": float(delta_cut_continuous),
        "discrete_cut_before": current_cut,
        "discrete_cut_after": candidate_cut,
        "delta_cut_disc": delta_cut_disc,
        "discrete_rel_before": current_rel,
        "discrete_rel_after": candidate_rel,
        "delta_rel_disc": delta_rel_disc,
    }
    if not math.isfinite(delta_cut_disc) or not math.isfinite(delta_rel_disc):
        return False, "nan_or_inf_delta", current_z.detach().clone(), stats
    if delta_cut_disc >= 0:
        return False, "discrete_cut_worse", current_z.detach().clone(), stats
    if delta_rel_disc > float(cfg.eps_rel):
        return False, "discrete_relational_worse", current_z.detach().clone(), stats
    return True, "accepted", candidate_disc.detach().clone(), stats


def optimize_sparse_candidate(
    model: torch.nn.Module,
    cfg: RAPV2Config,
    z_current: torch.Tensor,
    observed: torch.Tensor,
    attention_mask: torch.Tensor,
    variable_mask: torch.Tensor,
    fixed_public: Dict[int, int],
    update_positions: Sequence[int],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    device = z_current.device
    embed_layer = model.get_input_embeddings()
    fixed_embeds, fixed_positions, _ids = fixed_embedding_tensor(embed_layer, fixed_public, int(z_current.shape[1]), device)
    left, right = embedding_bounds(embed_layer.weight)
    left = left.to(device)
    right = right.to(device)
    update_positions = [int(pos) for pos in update_positions]
    if not update_positions:
        return z_current.detach(), {"skipped": True, "reason": "empty_uncertain_patch"}
    learnable = z_current[:, update_positions, :].detach().clone().squeeze(0).float().requires_grad_(True)
    opt = torch.optim.AdamW([learnable], lr=float(cfg.lr))
    best_z = z_current.detach().clone()
    best_loss = float(global_cut_loss(model, cfg, z_current.to(dtype=embed_layer.weight.dtype), observed, attention_mask, variable_mask).detach().cpu())
    final: Dict[str, Any] = {"initial_global_cut_loss": best_loss}
    for step in range(max(1, int(cfg.epoch))):
        z = apply_sparse_update(z_current, update_positions, learnable)
        enforce_embedding_constraints(z, fixed_positions, fixed_embeds, left, right)
        hidden = capture_prefix_activation(model, cfg.target_layer, inputs_embeds=z.to(dtype=embed_layer.weight.dtype), attention_mask=attention_mask)
        cut = stage1_b0v_cut_loss(hidden, observed, variable_mask)
        vocab = nearest_embedding_loss(learnable, embed_layer.weight)
        total = cut + float(cfg.lambda_vocab) * vocab
        if torch.isnan(total):
            raise RuntimeError(f"NaN in sparse patch optimization step {step + 1}")
        opt.zero_grad()
        total.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([learnable], float(cfg.grad_clip))
        opt.step()
        cut_value = float(cut.detach().cpu())
        if cut_value < best_loss:
            best_loss = cut_value
            best_z = z.detach().clone()
        final = {
            "step": step + 1,
            "candidate_cut_loss": cut_value,
            "candidate_vocab_loss": float(vocab.detach().cpu()),
            "candidate_total_loss": float(total.detach().cpu()),
            "best_candidate_cut_loss": best_loss,
        }
    return best_z.detach(), final


def run_sparse_repair(
    model: torch.nn.Module,
    cfg: RAPV2Config,
    observed: torch.Tensor,
    z_stage1: torch.Tensor,
    variable_mask: torch.Tensor,
    fixed_public: Dict[int, int],
    layer_scores: Optional[Dict[int, torch.Tensor]],
    aggregate_score: Optional[torch.Tensor],
    obs_states: Optional[Sequence[Tuple[int, torch.Tensor]]],
    uncertain_positions: Sequence[int],
    uncertainty_scores: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    device = z_stage1.device
    embed_layer = model.get_input_embeddings()
    attention_mask = torch.ones((1, z_stage1.shape[1]), dtype=torch.long, device=device)
    if aggregate_score is None:
        aggregate_score = identity_aggregate_score(int(z_stage1.shape[1]), device)
    uncertain_set = set(int(x) for x in uncertain_positions)
    plan = build_patch_plan(
        cfg.method,
        int(z_stage1.shape[1]),
        variable_mask,
        aggregate_score,
        uncertain_set,
        uncertainty_scores,
        cfg.patch_size,
        cfg.min_patch_size,
        cfg.max_patch_size,
    )
    priority_rows = plan["priority_rows"]
    ordered_rows = plan["ordered_rows"]
    destinations = top_r_relational_destinations(layer_scores or {cfg.target_layer + 1: aggregate_score}, uncertain_positions, variable_mask, cfg.top_r) if cfg.method in (CONTINUOUS_REL_GATE_METHODS | DISCRETE_REL_GATE_METHODS) else {}
    z = z_stage1.detach().clone()
    best_z = z.detach().clone()
    best_cut = float(global_cut_loss(model, cfg, z.to(dtype=embed_layer.weight.dtype), observed, attention_mask, variable_mask).detach().cpu())
    events: List[Dict[str, Any]] = []
    accepted_total = 0
    rejected_total = 0
    accepted_by_pass: List[int] = []
    processed_priority: List[Dict[str, Any]] = []
    for repair_pass in range(max(1, int(cfg.max_repair_passes))):
        accepted_this_pass = 0
        for row in ordered_rows:
            patch = [int(x) for x in row["positions"]]
            update_positions = patch_update_positions(patch, uncertain_set)
            pr_row = dict(row)
            pr_row["processed"] = bool(update_positions)
            pr_row["skipped"] = not bool(update_positions)
            if not update_positions:
                events.append({"repair_pass": repair_pass + 1, "patch_id": row["patch_id"], "accepted": False, "rejected_reason": "empty_uncertain_patch"})
                processed_priority.append(pr_row)
                continue
            audit = sparse_update_audit(update_positions, patch, uncertain_set, fixed_public=fixed_public)
            assert audit["updated_positions_subset_of_uncertain"]
            cut_before = float(global_cut_loss(model, cfg, z.to(dtype=embed_layer.weight.dtype), observed, attention_mask, variable_mask).detach().cpu())
            rel_before = 0.0
            if cfg.method in CONTINUOUS_REL_GATE_METHODS:
                with torch.no_grad():
                    h_before = capture_prefix_activation(model, cfg.target_layer, inputs_embeds=z.to(dtype=embed_layer.weight.dtype), attention_mask=attention_mask)
                    states_before, _a, _v = server_forward_evidence(model, cfg.target_layer + 1, h_before.to(dtype=embed_layer.weight.dtype), attention_mask, False, False)
                    rel_before = float(relational_discrepancy(states_before, obs_states or [], destinations, update_positions).detach().cpu())
            candidate, candidate_stats = optimize_sparse_candidate(model, cfg, z, observed, attention_mask, variable_mask, fixed_public, update_positions)
            cut_after = float(global_cut_loss(model, cfg, candidate.to(dtype=embed_layer.weight.dtype), observed, attention_mask, variable_mask).detach().cpu())
            rel_after = rel_before
            if cfg.method in CONTINUOUS_REL_GATE_METHODS:
                with torch.no_grad():
                    h_after = capture_prefix_activation(model, cfg.target_layer, inputs_embeds=candidate.to(dtype=embed_layer.weight.dtype), attention_mask=attention_mask)
                    states_after, _a, _v = server_forward_evidence(model, cfg.target_layer + 1, h_after.to(dtype=embed_layer.weight.dtype), attention_mask, False, False)
                    rel_after = float(relational_discrepancy(states_after, obs_states or [], destinations, update_positions).detach().cpu())
            delta_cut = cut_after - cut_before
            delta_rel = rel_after - rel_before
            eps_rel = float(cfg.eps_rel)
            if cfg.eps_rel_ratio is not None and rel_before > 0:
                eps_rel = max(eps_rel, float(cfg.eps_rel_ratio) * rel_before)
            discrete_stats: Dict[str, Any] = {}
            accepted_z = candidate
            if cfg.method in DISCRETE_REL_GATE_METHODS:
                accepted, reason, accepted_z, discrete_stats = discrete_accept_candidate(
                    model,
                    cfg,
                    z,
                    candidate,
                    observed,
                    attention_mask,
                    variable_mask,
                    fixed_public,
                    obs_states,
                    destinations,
                    update_positions,
                    delta_cut,
                )
            elif cfg.method in DELTA_CUT_ONLY_METHODS:
                accepted, reason = accept_candidate(delta_cut, 0.0, cfg.eps_cut, float("inf"))
            else:
                accepted, reason = accept_candidate(delta_cut, delta_rel, cfg.eps_cut, eps_rel)
            z = rollback_if_rejected(z, accepted_z, accepted)
            if accepted:
                accepted_total += 1
                accepted_this_pass += 1
                accepted_cut = float(global_cut_loss(model, cfg, z.to(dtype=embed_layer.weight.dtype), observed, attention_mask, variable_mask).detach().cpu())
                if accepted_cut < best_cut:
                    best_cut = accepted_cut
                    best_z = z.detach().clone()
            else:
                rejected_total += 1
            event = {
                "repair_pass": repair_pass + 1,
                "patch_id": int(row["patch_id"]),
                "accepted": bool(accepted),
                "rejected_reason": reason,
                "delta_cut": delta_cut,
                "delta_rel": delta_rel,
                "relative_delta_rel": (delta_rel / rel_before) if rel_before > 0 else None,
                "cut_before": cut_before,
                "cut_after": cut_after,
                "rel_before": rel_before,
                "rel_after": rel_after,
                "discrete_acceptance_gate": cfg.method in DISCRETE_REL_GATE_METHODS,
                **audit,
                **candidate_stats,
                **discrete_stats,
            }
            events.append(event)
            pr_row["accepted"] = bool(accepted)
            pr_row["rejected_reason"] = reason
            processed_priority.append(pr_row)
        accepted_by_pass.append(int(accepted_this_pass))
        if should_early_stop(accepted_this_pass):
            break
    early = early_stop_summary(accepted_by_pass, accepted_total)
    repair_stats = {
        "used": True,
        "repair_pass_count": max([e.get("repair_pass", 0) for e in events], default=0),
        "max_repair_passes": int(cfg.max_repair_passes),
        "accepted_patch_count": accepted_total,
        "rejected_patch_count": rejected_total,
        "accept_rate": accepted_total / max(1, accepted_total + rejected_total),
        **early,
        "mean_updated_token_count": statistics.mean([e["updated_position_count"] for e in events if "updated_position_count" in e]) if events else 0.0,
        "mean_frozen_token_count": statistics.mean([e["frozen_position_count"] for e in events if "frozen_position_count" in e]) if events else 0.0,
        "mean_delta_cut_accepted": statistics.mean([e["delta_cut"] for e in events if e.get("accepted") and e.get("delta_cut") is not None]) if any(e.get("accepted") for e in events) else None,
        "mean_delta_rel_accepted": statistics.mean([e["delta_rel"] for e in events if e.get("accepted") and e.get("delta_rel") is not None]) if any(e.get("accepted") for e in events) else None,
        "relational_destination_count": sum(len(v) for v in destinations.values()),
        "updated_positions_subset_of_uncertain": all(e.get("updated_positions_subset_of_uncertain", True) for e in events),
        "best_global_cut_loss": best_cut,
        "final_current_state_cut_loss": float(global_cut_loss(model, cfg, z.to(dtype=embed_layer.weight.dtype), observed, attention_mask, variable_mask).detach().cpu()),
        "returned_state_policy": "final_current_state_not_best_continuous_state",
        "partition_uses_relational_graph": bool(plan["partition_uses_relational_graph"]),
        "ordering_uses_relational_graph": bool(plan["ordering_uses_relational_graph"]),
    }
    return z.detach(), repair_stats, events, processed_priority, {"destinations_by_token": destinations}


def invert_observed(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: RAPV2Config,
    observed_activation: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> Tuple[List[int], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    cfg.method = canonical_method(cfg.method)
    require_strict_top1(cfg)
    fixed_public = inferred_boundary_special_tokens(tokenizer, seq_len) if cfg.fix_boundary_specials else {}
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    variable_audit = build_variable_mask(attention_mask, fixed_public, getattr(tokenizer, "all_special_ids", None), token_ids=None)
    loss_mode = "all_valid_uniform" if cfg.method == "original_pia_baseline" else "variable_uniform"
    z, losses, history = optimize_stage1(model, tokenizer, cfg, observed_activation, seq_len, device, fixed_public, variable_audit.variable_mask, loss_mode)
    uncertainty_stats: Dict[str, Any] = {}
    graph_stats: Dict[str, Any] = {"used": False}
    repair_stats: Dict[str, Any] = {"used": False}
    repair_events: List[Dict[str, Any]] = []
    priority_rows: List[Dict[str, Any]] = []
    rel_extra: Dict[str, Any] = {}
    if cfg.method == "variable_only_uniform":
        embed_layer = model.get_input_embeddings()
        margins = token_margins(z, embed_layer.weight)
        with torch.no_grad():
            stage1_hidden = capture_prefix_activation(model, cfg.target_layer, inputs_embeds=z.to(dtype=embed_layer.weight.dtype), attention_mask=attention_mask)
            residual = activation_residual_by_position(stage1_hidden, observed_activation)
        _scores, _positions, uncertainty_stats = build_uncertainty_score(
            margins,
            variable_audit.variable_mask,
            cfg.uncertainty_fraction,
            activation_residual=residual,
            detector=cfg.uncertainty_detector,
        )
    if cfg.method in SPARSE_METHODS:
        embed_layer = model.get_input_embeddings()
        margins = token_margins(z, embed_layer.weight)
        with torch.no_grad():
            stage1_hidden = capture_prefix_activation(model, cfg.target_layer, inputs_embeds=z.to(dtype=embed_layer.weight.dtype), attention_mask=attention_mask)
            residual = activation_residual_by_position(stage1_hidden, observed_activation)
        uncertainty_scores, uncertain_positions, uncertainty_stats = build_uncertainty_score(
            margins,
            variable_audit.variable_mask,
            cfg.uncertainty_fraction,
            activation_residual=residual,
            detector=cfg.uncertainty_detector,
        )
        layer_scores = None
        aggregate = None
        obs_states = None
        if cfg.method in RELATIONAL_ORDER_METHODS:
            with torch.no_grad():
                obs_states, attentions, value_norms = server_forward_evidence(
                    model,
                    cfg.target_layer + 1,
                    observed_activation.to(dtype=next(model.parameters()).dtype),
                    attention_mask,
                    collect_attention=True,
                    detach=True,
                )
                layer_scores, aggregate, graph_stats = build_relational_graph(attentions, value_norms, variable_audit.variable_mask)
                if cfg.method == "shuffled_rel":
                    layer_scores, aggregate, shuffle_stats = shuffle_layer_scores(layer_scores, aggregate, variable_audit.variable_mask, cfg.seed)
                    graph_stats.update(shuffle_stats)
                graph_stats["used"] = True
        z, repair_stats, repair_events, priority_rows, rel_extra = run_sparse_repair(
            model,
            cfg,
            observed_activation,
            z,
            variable_audit.variable_mask,
            fixed_public,
            layer_scores,
            aggregate,
            obs_states,
            uncertain_positions,
            uncertainty_scores,
        )
    embed_layer = model.get_input_embeddings()
    recovered_ids = naive_discretization(z, embed_layer.weight)
    for pos, token_id in fixed_public.items():
        if 0 <= int(pos) < len(recovered_ids):
            recovered_ids[int(pos)] = int(token_id)
    with torch.no_grad():
        hidden = capture_prefix_activation(model, cfg.target_layer, inputs_embeds=z.to(dtype=embed_layer.weight.dtype), attention_mask=attention_mask)
        all_loss = uniform_all_token_activation_loss(hidden, observed_activation, attention_mask)
        var_loss = stage1_b0v_cut_loss(hidden, observed_activation, variable_audit.variable_mask)
        active = all_loss if cfg.method == "original_pia_baseline" else var_loss
    losses.update({
        "activation_loss": float(active.detach().cpu()),
        "all_token_uniform_loss": float(all_loss.detach().cpu()),
        "variable_uniform_loss": float(var_loss.detach().cpu()),
        "optimization_loss": float(active.detach().cpu()),
        "loss_mode": loss_mode if cfg.method in BASELINE_METHODS else "sparse_patch_repair",
    })
    audits = {
        "variable_mask_audit": variable_mask_audit_payload(variable_audit),
        "uncertainty_stats": uncertainty_stats,
        "relational_graph_stats": graph_stats,
        "patch_priority": {"samples": priority_rows},
        "patch_repair_stats": repair_stats,
        "acceptance_gate_stats": {"events": repair_events, **repair_stats},
        "relational_destinations": rel_extra,
        "strict_top1": {
            "top_k_embedding": cfg.top_k_embedding,
            "top_y_semantic": cfg.top_y_semantic,
            "semantic_speculation": cfg.semantic_speculation,
            "adaptive_discretization": cfg.adaptive_discretization,
        },
        "attack_receives_ground_truth": False,
        **method_integrity_audit(),
    }
    return recovered_ids, losses, audits, history + repair_events, repair_stats


def validate_attack_api() -> Dict[str, Any]:
    checked = [invert_observed, optimize_stage1, run_sparse_repair, optimize_sparse_candidate, build_relational_graph]
    banned = ["reference", "original", "input_ids", "token_ids", "prompt", "text", "ground"]
    sig = inspect.signature(invert_observed)
    signature_hits = [name for name in sig.parameters if any(word in name for word in banned)]
    banned_globals = {"original_ids", "original_tokens", "ground_truth_ids", "prompt", "reference_tokens"}
    ast_hits: List[Dict[str, str]] = []
    for fn in checked:
        tree = ast.parse(inspect.getsource(fn))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in banned_globals:
                ast_hits.append({"function": fn.__name__, "name": node.id})
    return {
        "signature": str(sig),
        "parameter_names": list(sig.parameters),
        "banned_signature_hits": signature_hits,
        "banned_global_reference_hits": ast_hits,
        "passes": not signature_hits and not ast_hits,
    }


def load_model_and_data(cfg: RAPV2Config):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer, model = load_tinyllama(dtype, local_files_only=cfg.local_files_only)
    model.to(device)
    model.eval()
    if len(model.model.layers) != TOTAL_BLOCKS:
        raise RuntimeError(f"Expected {TOTAL_BLOCKS} blocks, got {len(model.model.layers)}")
    return tokenizer, model, device


def ensure_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def run_one_config(cfg: RAPV2Config, resume: bool) -> Dict[str, Any]:
    require_strict_top1(cfg)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if resume and (out / "COMPLETE").exists():
        return json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    json_dump(out / "config.json", {"title": EXPERIMENT_TITLE, **cfg.__dict__})
    tokenizer, model, device = load_model_and_data(cfg)
    prompts, dataset_meta = load_dataset_prompts(cfg.dataset_name, cfg.dataset_path, cfg.dataset_len, cfg.seed)
    json_dump(out / "dataset_meta.json", dataset_meta)
    pred_path = out / "predictions.jsonl"
    fail_path = out / "failures.jsonl"
    if not resume:
        for path in [pred_path, fail_path]:
            if path.exists():
                path.unlink()
    ensure_jsonl(pred_path)
    ensure_jsonl(fail_path)
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    variable_samples = []
    uncertainty_samples = []
    graph_samples = []
    priority_samples = []
    repair_samples = []
    gate_samples = []
    loss_samples = []
    for prompt_id, prompt in enumerate(prompts):
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start = time.time()
            tok = tokenizer(prompt, add_special_tokens=True, truncation=False, return_tensors="pt")
            input_ids = tok["input_ids"].to(device)
            attention_mask = tok["attention_mask"].to(device)
            if input_ids.shape[1] > cfg.max_token_len:
                raise RuntimeError(f"prompt_id={prompt_id} has {input_ids.shape[1]} tokens > max_token_len={cfg.max_token_len}")
            eval_original_ids = [int(x) for x in input_ids[0].detach().cpu().tolist()]
            with torch.no_grad():
                observed = capture_prefix_activation(model, cfg.target_layer, input_ids=input_ids, attention_mask=attention_mask).detach()
            preflight = {
                "model_name": MODEL_NAME,
                "target_layer_index": cfg.target_layer,
                "h_obs_semantics": f"output of 0-based TinyLlama block {cfg.target_layer}",
                "server_start_layer": cfg.target_layer + 1,
                "server_layer_indices": list(range(cfg.target_layer + 1, len(model.model.layers))),
                "sequence_length": int(input_ids.shape[1]),
                "strict_top1": True,
                "top_k_embedding": cfg.top_k_embedding,
                "top_y_semantic": cfg.top_y_semantic,
                "semantic_speculation": cfg.semantic_speculation,
                "adaptive_discretization": cfg.adaptive_discretization,
                "attack_receives_ground_truth": False,
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "cpu"),
            }
            print(f"preflight={json.dumps(preflight, ensure_ascii=True)}", flush=True)
            recovered_ids, losses, audits, history, repair_stats = invert_observed(model, tokenizer, cfg, observed, int(input_ids.shape[1]), device)
            recovered_text = text_from_ids(tokenizer, recovered_ids)
            original_text = text_from_ids(tokenizer, eval_original_ids)
            variable_rows = audits["variable_mask_audit"]["rows"]
            elapsed = time.time() - start
            row = {
                "title": EXPERIMENT_TITLE,
                "method": cfg.method,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "original_text": original_text,
                "recovered_text": recovered_text,
                "original_token_ids": eval_original_ids,
                "recovered_token_ids": recovered_ids,
                "original_tokens": token_texts(tokenizer, eval_original_ids),
                "recovered_tokens": token_texts(tokenizer, recovered_ids),
                "token_accuracy": token_accuracy(tokenizer, eval_original_ids, recovered_ids),
                "bleu": bleu_score(tokenizer, eval_original_ids, recovered_ids),
                "nerr": optional_nerr(original_text, recovered_text),
                "prompt_exact_match": prompt_exact_match_for_variable(eval_original_ids, recovered_ids, variable_rows),
                "runtime_seconds": elapsed,
                "elapsed_time": elapsed,
                "seed": cfg.seed,
                "target_layer": cfg.target_layer,
                "prompt_token_count": len(eval_original_ids),
                "peak_gpu_memory_mb": peak_memory_mb(),
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "cpu"),
                "top_k_embedding": cfg.top_k_embedding,
                "top_y_semantic": cfg.top_y_semantic,
                "preflight": preflight,
                "recovery_uses_ground_truth_tokens": False,
                **losses,
                **repair_stats,
            }
            jsonl_append(pred_path, row)
            rows.append(row)
            variable_samples.append({"prompt_id": prompt_id, **audits["variable_mask_audit"]})
            uncertainty_samples.append({"prompt_id": prompt_id, **audits["uncertainty_stats"]})
            graph_samples.append({"prompt_id": prompt_id, **audits["relational_graph_stats"]})
            priority_samples.append({"prompt_id": prompt_id, **audits["patch_priority"]})
            repair_samples.append({"prompt_id": prompt_id, **audits["patch_repair_stats"]})
            gate_samples.append({"prompt_id": prompt_id, **audits["acceptance_gate_stats"]})
            loss_samples.append({"prompt_id": prompt_id, "final": losses, "history": history})
            print(f"method={cfg.method} prompt_id={prompt_id} token_accuracy={row['token_accuracy']:.6f} bleu={row['bleu']:.6f} elapsed={elapsed:.2f}s", flush=True)
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
            jsonl_append(fail_path, failure)
            failures.append(failure)
            print(f"failure={json.dumps(failure, ensure_ascii=True)}", flush=True)
    metrics = summarize_rows(rows)
    metrics.update({
        "title": EXPERIMENT_TITLE,
        "method": cfg.method,
        "seed": cfg.seed,
        "target_layer": cfg.target_layer,
        "completed_sample_count": len(rows),
        "failed_sample_count": len(failures),
        "dataset_meta": dataset_meta,
        "top_k_embedding": cfg.top_k_embedding,
        "top_y_semantic": cfg.top_y_semantic,
        "semantic_speculation": cfg.semantic_speculation,
        "adaptive_discretization": cfg.adaptive_discretization,
        "has_nan": has_nan(rows) or has_nan(failures),
        "output_dir": cfg.output_dir,
    })
    json_dump(out / "metrics.json", metrics)
    json_dump(out / "variable_mask_audit.json", {"samples": variable_samples})
    json_dump(out / "uncertainty_stats.json", {"samples": uncertainty_samples})
    json_dump(out / "relational_graph_stats.json", {"samples": graph_samples})
    json_dump(out / "patch_priority.json", {"samples": priority_samples})
    json_dump(out / "patch_repair_stats.json", {"samples": repair_samples})
    json_dump(out / "acceptance_gate_stats.json", {"samples": gate_samples})
    json_dump(out / "loss_breakdown.json", {"samples": loss_samples})
    (out / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return metrics


def config_from_args(args: argparse.Namespace, method: str, run_name: str, output_dir: str) -> RAPV2Config:
    method = canonical_method(method)
    inverted, target = ATTACKER_MAP[args.participant_number][args.attacker_position]
    if args.target_layer is not None:
        target = int(args.target_layer)
        inverted = target + 1
    return RAPV2Config(
        method=method,
        run_name=run_name,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        dataset_len=args.dataset_len,
        seed=args.seed,
        participant_number=args.participant_number,
        attacker_position=args.attacker_position,
        inverted_block_count=inverted,
        target_layer=target,
        epoch=args.epoch,
        lr=args.lr,
        lambda_vocab=args.lambda_vocab,
        top_k_embedding=args.k,
        top_y_semantic=args.y,
        max_token_len=args.max_token_len,
        grad_clip=args.grad_clip,
        uncertainty_fraction=args.uncertainty_fraction,
        patch_size=args.patch_size,
        min_patch_size=args.min_patch_size,
        max_patch_size=args.max_patch_size,
        top_r=args.top_r,
        max_repair_passes=args.max_repair_passes,
        eps_cut=args.eps_cut,
        eps_rel=args.eps_rel,
        eps_rel_ratio=args.eps_rel_ratio,
        uncertainty_detector=args.uncertainty_detector,
        adaptive_discretization=not args.no_adaptive_discretization,
        semantic_speculation=not args.disable_semantic_speculation,
        local_files_only=args.local_files_only,
    )


def verify_no_leakage(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = config_from_args(args, "RAP_FINAL", "verify", str(Path(args.output_root) / "leakage_verification" / "sample"))
    cfg.dataset_len = 1
    require_strict_top1(cfg)
    tokenizer, model, device = load_model_and_data(cfg)
    prompts, dataset_meta = load_dataset_prompts(cfg.dataset_name, cfg.dataset_path, 1, cfg.seed)
    tok = tokenizer(prompts[0], add_special_tokens=True, truncation=False, return_tensors="pt")
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)
    with torch.no_grad():
        observed = capture_prefix_activation(model, cfg.target_layer, input_ids=input_ids, attention_mask=attention_mask).detach()
        full = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        full_boundary = full.hidden_states[cfg.target_layer + 1].detach()
        obs_states, attentions, value_norms = server_forward_evidence(model, cfg.target_layer + 1, observed.to(dtype=next(model.parameters()).dtype), attention_mask, True, True)
    fixed_public = inferred_boundary_special_tokens(tokenizer, int(input_ids.shape[1]))
    variable_audit = build_variable_mask(attention_mask, fixed_public, getattr(tokenizer, "all_special_ids", None), token_ids=None)
    layer_scores, aggregate, graph_stats = build_relational_graph(attentions, value_norms, variable_audit.variable_mask)
    margins = torch.linspace(0.0, 1.0, steps=int(input_ids.shape[1]), device=device)
    scores, uncertain, _stats = build_uncertainty_score(margins, variable_audit.variable_mask, cfg.uncertainty_fraction)
    patches = fixed_size_patches(int(input_ids.shape[1]), variable_audit.variable_mask, cfg.patch_size)
    priority = compute_patch_priorities(patches, set(uncertain), scores, aggregate, variable_audit.variable_mask)
    subset_ok = all(sparse_update_audit(row["uncertain_positions"], row["positions"], set(uncertain), fixed_public)["updated_positions_subset_of_uncertain"] for row in priority)
    boundary_diff = float((observed.float() - full_boundary.float()).abs().max().detach().cpu())
    api_check = validate_attack_api()
    integrity = validate_method_integrity()
    result = {
        "title": EXPERIMENT_TITLE,
        "dataset_meta": dataset_meta,
        "target_layer": cfg.target_layer,
        "h_obs_semantics": f"H_obs is output of 0-based block {cfg.target_layer}; server starts at block {cfg.target_layer + 1}",
        "server_start": cfg.target_layer + 1,
        "server_layers": [idx for idx, _state in obs_states],
        "manual_h_obs_boundary_vs_full_max_abs_diff": boundary_diff,
        "manual_boundary_pass": boundary_diff < 5e-3,
        "attack_api_check": api_check,
        "method_integrity_check": integrity,
        "strict_top1": {
            "enabled": True,
            "K": cfg.top_k_embedding,
            "Y": cfg.top_y_semantic,
            "semantic": cfg.semantic_speculation,
            "calibration": cfg.adaptive_discretization,
        },
        "no_ground_truth_in_attack": bool(api_check["passes"]),
        "no_dummy_attention": not bool(integrity["dummy_reference_hits"]),
        "no_scalar_attention_weighting_loss": not bool(integrity["scalar_attention_call_hits"]),
        "updated_positions_subset_of_uncertain": subset_ok,
        "token_id_based_special_mask_available": False,
        "variable_mask_audit": variable_mask_audit_payload(variable_audit),
        "relational_graph_stats": graph_stats,
        "aggregate_shape": list(aggregate.shape),
        "passes": boundary_diff < 5e-3 and api_check["passes"] and integrity["passes"] and subset_ok,
    }
    out = Path(args.output_root) / "leakage_verification"
    out.mkdir(parents=True, exist_ok=True)
    json_dump(out / "leakage_verification.json", result)
    Path("analysis/rap_final_preflight_audit.md").write_text("# RAP-FINAL Preflight Audit\n\n```json\n" + json.dumps(result, indent=2, ensure_ascii=True) + "\n```\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return result


def method_dir_name(method: str) -> str:
    return canonical_method(method)


def run_method_set(args: argparse.Namespace, methods: Sequence[str], subdir: str) -> List[Dict[str, Any]]:
    rows = []
    for method in methods:
        canonical = canonical_method(method)
        out = Path(args.output_root) / subdir / canonical
        cfg = config_from_args(args, method, canonical, str(out))
        metrics = run_one_config(cfg, args.resume)
        rows.append({
            "method": canonical,
            "run_name": canonical,
            "token_accuracy": metrics.get("token_accuracy", {}).get("mean"),
            "bleu": metrics.get("bleu", {}).get("mean"),
            "prompt_exact_match": metrics.get("prompt_exact_match", {}).get("mean"),
            "completed_sample_count": metrics.get("completed_sample_count"),
            "failed_sample_count": metrics.get("failed_sample_count"),
            "output_dir": metrics.get("output_dir"),
        })
    out_csv = Path(args.output_root) / f"{subdir}_comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_prediction_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def paired_deltas(a_rows: List[Dict[str, Any]], b_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    a = {int(row["prompt_id"]): row for row in a_rows}
    b = {int(row["prompt_id"]): row for row in b_rows}
    prompt_ids = sorted(set(a) & set(b))
    acc_d = [float(a[i]["token_accuracy"]) - float(b[i]["token_accuracy"]) for i in prompt_ids]
    bleu_d = [float(a[i]["bleu"]) - float(b[i]["bleu"]) for i in prompt_ids]
    wins = sum(1 for x in acc_d if x > 1e-12)
    ties = sum(1 for x in acc_d if abs(x) <= 1e-12)
    losses = sum(1 for x in acc_d if x < -1e-12)
    return {
        "paired_prompt_count": len(prompt_ids),
        "mean_token_accuracy_delta": statistics.mean(acc_d) if acc_d else None,
        "mean_bleu_delta": statistics.mean(bleu_d) if bleu_d else None,
        "win_count": wins,
        "tie_count": ties,
        "loss_count": losses,
        "net_fix": token_level_net_fix(a, b, prompt_ids),
    }


def token_level_net_fix(a: Dict[int, Dict[str, Any]], b: Dict[int, Dict[str, Any]], prompt_ids: Sequence[int]) -> Dict[str, int]:
    fix = harm = 0
    for pid in prompt_ids:
        ar = a[int(pid)]
        br = b[int(pid)]
        orig = [int(x) for x in ar.get("original_token_ids", [])]
        aa = [int(x) for x in ar.get("recovered_token_ids", [])]
        bb = [int(x) for x in br.get("recovered_token_ids", [])]
        for idx, gold in enumerate(orig):
            a_ok = idx < len(aa) and int(aa[idx]) == gold
            b_ok = idx < len(bb) and int(bb[idx]) == gold
            if a_ok and not b_ok:
                fix += 1
            elif b_ok and not a_ok:
                harm += 1
    return {"fix_count": fix, "harm_count": harm, "net_fix_count": fix - harm}


def bootstrap_ci(values: Sequence[float], seed: int = 123, rounds: int = 10000) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(x) for x in values]
    if not vals:
        return None, None
    rng = random.Random(int(seed))
    means: List[float] = []
    for _ in range(int(rounds)):
        sample = [vals[rng.randrange(len(vals))] for _j in vals]
        means.append(statistics.mean(sample))
    means.sort()
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def summarize_run_root(output_root: str) -> Dict[str, Any]:
    root = Path(output_root)
    method_dirs = sorted(root.glob("heldout_seed*/**/metrics.json"))
    rows: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []
    for metrics_path in method_dirs:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        method = metrics.get("method")
        seed = metrics.get("seed")
        rows.append({
            "seed": seed,
            "method": method,
            "token_accuracy": metrics.get("token_accuracy", {}).get("mean"),
            "bleu": metrics.get("bleu", {}).get("mean"),
            "prompt_exact_match": metrics.get("prompt_exact_match", {}).get("mean"),
            "completed_sample_count": metrics.get("completed_sample_count"),
            "failed_sample_count": metrics.get("failed_sample_count"),
            "has_nan": metrics.get("has_nan"),
            "output_dir": metrics.get("output_dir"),
        })
        runtime_rows.append({
            "seed": seed,
            "method": method,
            "runtime_seconds_mean": metrics.get("runtime_seconds", {}).get("mean"),
            "peak_gpu_memory_mb_mean": metrics.get("peak_gpu_memory_mb", {}).get("mean"),
            "completed_sample_count": metrics.get("completed_sample_count"),
            "failed_sample_count": metrics.get("failed_sample_count"),
        })
    if rows:
        csv_write(root / "heldout_pooled.csv", rows)
        csv_write(root / "runtime_memory_summary.csv", runtime_rows)
    paired_rows: List[Dict[str, Any]] = []
    for seed_dir in sorted(root.glob("heldout_seed*")):
        seed_label = seed_dir.name.replace("heldout_seed", "")
        by_method = {p.parent.name: load_prediction_rows(p) for p in seed_dir.glob("*/predictions.jsonl")}
        for method, baseline in [("sparse_fixed", "variable_only_uniform"), ("rap_final", "variable_only_uniform"), ("shuffled_rel", "variable_only_uniform"), ("rap_final", "shuffled_rel"), ("rap_final", "sparse_fixed")]:
            if method in by_method and baseline in by_method:
                delta = paired_deltas(by_method[method], by_method[baseline])
                paired_rows.append({"seed": seed_label, "comparison": f"{method}_vs_{baseline}", **delta})
    if paired_rows:
        csv_write(root / "heldout_paired_summary.csv", paired_rows)
    make_length_bucket_summary(root)
    return {"summary_rows": rows, "paired_rows": paired_rows}


def make_length_bucket_summary(root: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for pred_path in sorted(root.glob("heldout_seed*/**/predictions.jsonl")):
        method = pred_path.parent.name
        seed = pred_path.parents[1].name.replace("heldout_seed", "")
        for row in load_prediction_rows(pred_path):
            length = int(row.get("prompt_token_count", 0))
            bucket = "short" if length < 128 else "medium" if length < 256 else "long"
            rows.append({"seed": seed, "method": method, "bucket": bucket, "token_accuracy": float(row["token_accuracy"]), "bleu": float(row["bleu"])})
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["seed"], row["method"], row["bucket"]), []).append(row)
    out = []
    for (seed, method, bucket), vals in sorted(grouped.items()):
        out.append({
            "seed": seed,
            "method": method,
            "bucket": bucket,
            "sample_count": len(vals),
            "token_accuracy": statistics.mean([v["token_accuracy"] for v in vals]),
            "bleu": statistics.mean([v["bleu"] for v in vals]),
        })
    if out:
        csv_write(root / "length_bucket_summary.csv", out)


def csv_write(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def normalize_prompt_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def prompt_hash(text: str) -> str:
    return hashlib.sha256(normalize_prompt_for_hash(text).encode("utf-8")).hexdigest()


def read_prompt_list(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"expected list prompts in {path}")
    return [str(x) for x in data]


def prepare_heldout_split(args: argparse.Namespace) -> Dict[str, Any]:
    dev_path = Path("data/airline.json")
    full_path = Path("data/skytrax_150.json")
    dev_prompts = read_prompt_list(dev_path)
    full_prompts = read_prompt_list(full_path)
    dev_hashes = {prompt_hash(p) for p in dev_prompts}
    heldout: List[str] = []
    overlap: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for index, prompt in enumerate(full_prompts):
        h = prompt_hash(prompt)
        if h in dev_hashes:
            overlap.append({"skytrax_150_index": index, "sha256": h, "reason": "overlap_with_airline28"})
            continue
        if h in seen:
            overlap.append({"skytrax_150_index": index, "sha256": h, "reason": "duplicate_within_skytrax_150"})
            continue
        seen.add(h)
        heldout.append(prompt)
    out_path = Path("data/rap_final_heldout.json")
    out_path.write_text(json.dumps(heldout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "created_for": EXPERIMENT_TITLE,
        "development_sets": ["data/airline.json seeds 42/43/44 are development/diagnostic only"],
        "source_dataset": str(full_path),
        "source_prompt_count": len(full_prompts),
        "development_prompt_count": len(dev_prompts),
        "removed_overlap_or_duplicate_count": len(overlap),
        "heldout_prompt_count": len(heldout),
        "heldout_path": str(out_path),
        "heldout_file_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "normalization": "collapse whitespace, strip, lowercase, SHA256",
        "overlap_rows": overlap,
        "no_test_prompt_modified_after_results": True,
    }
    Path("analysis").mkdir(exist_ok=True)
    json_dump(Path("analysis/rap_final_heldout_manifest.json"), manifest)
    Path("analysis/rap_prompt_overlap_audit.md").write_text(
        "# RAP-FINAL Prompt Overlap Audit\n\n"
        f"- Source prompts: {len(full_prompts)}\n"
        f"- Development prompts: {len(dev_prompts)}\n"
        f"- Removed overlaps/duplicates: {len(overlap)}\n"
        f"- Held-out prompts: {len(heldout)}\n"
        f"- Held-out SHA256: `{manifest['heldout_file_sha256']}`\n\n"
        "Seed42/43/44 are treated as development/diagnostic only. The held-out split is fixed before seed45/46/47 evaluation.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=True), flush=True)
    return manifest


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_uncertainty_detector(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.output_root)
    rows: List[Dict[str, Any]] = []
    for pred_path in sorted(root.glob("dev_seed*/variable_only_uniform/predictions.jsonl")):
        seed = int(pred_path.parents[1].name.replace("dev_seed", ""))
        predictions = load_prediction_rows(pred_path)
        uncertainty_path = pred_path.parent / "uncertainty_stats.json"
        var_path = pred_path.parent / "variable_mask_audit.json"
        if not uncertainty_path.exists() or not var_path.exists():
            continue
        uncertainty_samples = {int(s["prompt_id"]): s for s in load_json(uncertainty_path).get("samples", [])}
        variable_samples = {int(s["prompt_id"]): s for s in load_json(var_path).get("samples", [])}
        for row in predictions:
            pid = int(row["prompt_id"])
            selected = set(int(x) for x in uncertainty_samples.get(pid, {}).get("selected_uncertain_positions", []))
            variable_positions = [int(r["position"]) for r in variable_samples.get(pid, {}).get("rows", []) if r.get("variable_mask")]
            original = [int(x) for x in row.get("original_token_ids", [])]
            recovered = [int(x) for x in row.get("recovered_token_ids", [])]
            for pos in variable_positions:
                if pos >= len(original):
                    continue
                correct = pos < len(recovered) and int(original[pos]) == int(recovered[pos])
                rows.append({
                    "seed": seed,
                    "prompt_id": pid,
                    "position": pos,
                    "selected_uncertain": pos in selected,
                    "b0v_correct": correct,
                    "b0v_wrong": not correct,
                })
    selected_rows = [r for r in rows if r["selected_uncertain"]]
    not_rows = [r for r in rows if not r["selected_uncertain"]]
    overall_wrong = statistics.mean([1.0 if r["b0v_wrong"] else 0.0 for r in rows]) if rows else None
    selected_wrong = statistics.mean([1.0 if r["b0v_wrong"] else 0.0 for r in selected_rows]) if selected_rows else None
    not_wrong = statistics.mean([1.0 if r["b0v_wrong"] else 0.0 for r in not_rows]) if not_rows else None
    enrichment = (selected_wrong / overall_wrong) if selected_wrong is not None and overall_wrong and overall_wrong > 0 else None
    decision = "margin" if enrichment is not None and enrichment >= 1.5 else "joint"
    summary = {
        "token_count": len(rows),
        "selected_token_count": len(selected_rows),
        "not_selected_token_count": len(not_rows),
        "p_b0v_wrong_given_selected_uncertain": selected_wrong,
        "p_b0v_wrong_given_not_selected": not_wrong,
        "overall_wrong_rate": overall_wrong,
        "enrichment_ratio": enrichment,
        "recommended_uncertainty_detector": decision,
        "rule": "keep margin-only if pooled enrichment >= 1.5; otherwise use deterministic joint uncertainty without beta search",
    }
    Path("analysis").mkdir(exist_ok=True)
    csv_write(root / "uncertainty_audit_tokens.csv", rows)
    Path("analysis/rap_uncertainty_audit.md").write_text(
        "# RAP-FINAL Uncertainty Audit\n\n"
        "Seed42/43/44 are development/diagnostic only.\n\n"
        f"- P(B0V wrong | selected uncertain): {selected_wrong}\n"
        f"- P(B0V wrong | not selected): {not_wrong}\n"
        f"- Overall wrong rate: {overall_wrong}\n"
        f"- Enrichment ratio: {enrichment}\n"
        f"- Frozen detector for final heldout: `{decision}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EXPERIMENT_TITLE)
    parser.add_argument("--mode", choices=["verify", "single", "smoke", "dev", "prepare-heldout", "summarize", "audit-uncertainty"], default="single")
    parser.add_argument("--method", default="RAP_FINAL", choices=METHODS)
    parser.add_argument("--output-root", default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--subdir", default=None)
    parser.add_argument("--dataset-name", default="Skytrax")
    parser.add_argument("--dataset-path", default="data/airline.json")
    parser.add_argument("--dataset-len", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--participant-number", type=int, default=4)
    parser.add_argument("--attacker-position", type=int, default=4)
    parser.add_argument("--target-layer", type=int, default=17)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--lambda-vocab", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--y", type=int, default=0)
    parser.add_argument("--max-token-len", type=int, default=896)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--uncertainty-fraction", type=float, default=0.20)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--min-patch-size", type=int, default=12)
    parser.add_argument("--max-patch-size", type=int, default=20)
    parser.add_argument("--top-r", type=int, default=4)
    parser.add_argument("--max-repair-passes", type=int, default=2)
    parser.add_argument("--eps-cut", type=float, default=1e-6)
    parser.add_argument("--eps-rel", type=float, default=0.0)
    parser.add_argument("--eps-rel-ratio", type=float, default=None)
    parser.add_argument("--uncertainty-detector", choices=["margin", "joint"], default="margin")
    parser.add_argument("--no-adaptive-discretization", action="store_true", default=True)
    parser.add_argument("--disable-semantic-speculation", action="store_true", default=True)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "verify":
        verify_no_leakage(args)
    elif args.mode == "prepare-heldout":
        prepare_heldout_split(args)
    elif args.mode == "single":
        subdir = args.subdir or "single"
        cfg = config_from_args(args, args.method, canonical_method(args.method), str(Path(args.output_root) / subdir / canonical_method(args.method)))
        run_one_config(cfg, args.resume)
    elif args.mode == "smoke":
        args.dataset_len = 2
        args.epoch = min(int(args.epoch), 20)
        run_method_set(args, ["B0", "B0V", "SPARSE_FIXED", "REL_ORDER", "RAP_CONT_GATE", "RAP_FINAL", "SHUFFLED_REL"], "smoke")
    elif args.mode == "dev":
        run_method_set(args, ["B0", "B0V", "SPARSE_FIXED", "REL_ORDER", "RAP_CONT_GATE", "RAP_FINAL", "SHUFFLED_REL"], f"dev_seed{args.seed}")
        comparison = Path(args.output_root) / f"dev_seed{args.seed}_comparison.csv"
        target = Path(args.output_root) / "dev_component_summary.csv"
        if comparison.exists():
            rows = list(csv.DictReader(comparison.open("r", encoding="utf-8")))
            existing = list(csv.DictReader(target.open("r", encoding="utf-8"))) if target.exists() else []
            csv_write(target, existing + rows)
    elif args.mode == "summarize":
        summarize_run_root(args.output_root)
    elif args.mode == "audit-uncertainty":
        audit_uncertainty_detector(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
