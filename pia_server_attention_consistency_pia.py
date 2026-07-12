import argparse
import ast
import csv
import inspect
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TRANSFORMERS_CACHE", "/data/zhiqi/hf_cache/transformers")

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask

from pia_attention_guided import enforce_embedding_constraints, fixed_embedding_tensor, random_public_embeddings, variable_view
from pia_masked_server_attn_pia import (
    GpuMemory,
    bootstrap_mean_ci,
    build_variable_mask,
    fmt_md,
    metric_mean,
    plan_auto_batch_jobs,
    query_gpu_memory,
    uniform_all_token_activation_loss,
    variable_mask_audit_payload,
    weighted_variable_activation_loss,
)
from pia_tinyllama import (
    MODEL_NAME,
    TOTAL_BLOCKS,
    activation_calibrated_discretization,
    bleu_score,
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


EXPERIMENT_TITLE = "Server-Attention Consistency-Regularized Prompt Inversion (SACR-PIA)"
OUTPUT_ROOT_DEFAULT = "runs/server_attention_consistency_pia"
METHOD_ALIASES = {
    "B0V": "variable_only_uniform",
    "C1": "tail_consistency",
    "C2": "attention_consistency",
    "C3": "tail_attention_consistency",
}
METHODS = ["B0V", "C1", "C2", "C3", *METHOD_ALIASES.values()]
CONSISTENCY_METHODS = {"tail_consistency", "attention_consistency", "tail_attention_consistency"}


@dataclass
class SACRConfig:
    method: str
    run_name: str
    output_dir: str
    dataset_name: str
    dataset_path: str
    dataset_len: int
    prompt_split_seed: int
    split: str
    prompt_limit: int
    init_seed: int
    participant_number: int
    attacker_position: int
    inverted_block_count: int
    target_layer: int
    epoch: int
    lr: float
    lambda_vocab: float
    lambda_tail: float
    lambda_attn: float
    rho_tail: float
    rho_attn: float
    top_k_embedding: int
    top_y_semantic: int
    max_token_len: int
    grad_clip: float
    server_attention_layers: str
    adaptive_discretization: bool
    semantic_speculation: bool
    local_files_only: bool
    fix_boundary_specials: bool = True
    eps: float = 1e-8


@dataclass(frozen=True)
class SACRTask:
    method: str
    split: str
    init_seed: int
    output_subdir: str
    prompt_limit: int
    lambda_tail: float = 0.0
    lambda_attn: float = 0.0
    rho_tail: float = 0.10
    rho_attn: float = 0.10
    server_attention_layers: str = "all"
    run_name: Optional[str] = None
    gpu_index: int = 0


def canonical_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def float_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.float().clamp_min(eps)
    q = q.float().clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * (p * (p / m).log()).sum(dim=-1) + 0.5 * (q * (q / m).log()).sum(dim=-1)


def tail_consistency_loss(pred_tail: torch.Tensor, ref_tail: torch.Tensor, variable_mask: torch.Tensor) -> torch.Tensor:
    variable = variable_mask.to(pred_tail.device).bool()
    if int(variable.sum().detach().cpu()) == 0:
        raise RuntimeError("variable_mask has no variable positions")
    per_token = torch.mean((pred_tail.float() - ref_tail.to(pred_tail.device).float()) ** 2, dim=-1)
    return per_token[:, variable].mean()


def attention_js_consistency_loss(
    ref_attentions: Sequence[Tuple[int, torch.Tensor]],
    pred_attentions: Sequence[Tuple[int, torch.Tensor]],
    variable_mask: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if len(ref_attentions) != len(pred_attentions):
        raise RuntimeError("reference and predicted attention layer counts differ")
    if not ref_attentions:
        raise RuntimeError("attention consistency requires at least one server attention layer")
    device = pred_attentions[0][1].device
    variable = variable_mask.to(device).bool()
    query_positions = torch.nonzero(variable, as_tuple=False).flatten()
    js_terms: List[torch.Tensor] = []
    skipped = 0
    layer_ids: List[int] = []
    for (ref_layer, ref), (pred_layer, pred) in zip(ref_attentions, pred_attentions):
        if int(ref_layer) != int(pred_layer):
            raise RuntimeError("reference and predicted attention layer ids differ")
        layer_ids.append(int(ref_layer))
        ref = ref.to(device).float()
        pred = pred.float()
        if ref.shape != pred.shape:
            raise RuntimeError(f"attention shape mismatch: {list(ref.shape)} vs {list(pred.shape)}")
        for q in query_positions:
            ref_row = ref[int(q)] * variable.float()
            pred_row = pred[int(q)] * variable.float()
            ref_mass = ref_row.sum()
            pred_mass = pred_row.sum()
            if float(ref_mass.detach().cpu()) <= eps or float(pred_mass.detach().cpu()) <= eps:
                skipped += 1
                continue
            ref_dist = (ref_row / ref_mass.clamp_min(eps))[variable]
            pred_dist = (pred_row / pred_mass.clamp_min(eps))[variable]
            js_terms.append(jensen_shannon_divergence(ref_dist.unsqueeze(0), pred_dist.unsqueeze(0), eps=eps)[0])
    if js_terms:
        loss = torch.stack(js_terms).mean()
    else:
        loss = pred_attentions[0][1].sum() * 0.0
    return loss, {
        "server_layers": layer_ids,
        "query_positions": [int(x) for x in query_positions.detach().cpu().tolist()],
        "variable_key_count": int(variable.sum().detach().cpu()),
        "used_rows": len(js_terms),
        "skipped_rows": skipped,
        "js_mean": float(loss.detach().cpu()),
    }


def tensor_norm(tensor: Optional[torch.Tensor]) -> float:
    if tensor is None:
        return 0.0
    return float(torch.linalg.vector_norm(tensor.detach().float()).cpu())


def grad_cosine(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[float]:
    if a is None or b is None:
        return None
    af = a.detach().float().flatten()
    bf = b.detach().float().flatten()
    if float(torch.linalg.vector_norm(af).cpu()) <= 1e-12 or float(torch.linalg.vector_norm(bf).cpu()) <= 1e-12:
        return None
    return float(F.cosine_similarity(af, bf, dim=0).detach().cpu())


def capped_auxiliary_lambda(
    base_grad: Optional[torch.Tensor],
    aux_grad: Optional[torch.Tensor],
    raw_lambda: float,
    rho: float,
    eps: float = 1e-12,
) -> float:
    if raw_lambda <= 0.0:
        return 0.0
    base_norm = tensor_norm(base_grad)
    aux_norm = tensor_norm(aux_grad)
    if base_norm <= eps or aux_norm <= eps:
        return 0.0
    return float(min(float(raw_lambda), float(rho) * base_norm / aux_norm))


def normalized_auxiliary_loss(loss: torch.Tensor, initial: Optional[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    if initial is None:
        return loss
    return loss / (initial.detach().to(loss.device).float() + float(eps))


def sacr_total_loss(
    base_loss: torch.Tensor,
    vocab_loss: torch.Tensor,
    tail_loss: torch.Tensor,
    attn_loss: torch.Tensor,
    tail_initial: Optional[torch.Tensor],
    attn_initial: Optional[torch.Tensor],
    lambda_vocab: float,
    lambda_tail_eff: float,
    lambda_attn_eff: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    total = base_loss + float(lambda_vocab) * vocab_loss
    if lambda_tail_eff > 0.0:
        total = total + float(lambda_tail_eff) * normalized_auxiliary_loss(tail_loss, tail_initial, eps)
    if lambda_attn_eff > 0.0:
        total = total + float(lambda_attn_eff) * normalized_auxiliary_loss(attn_loss, attn_initial, eps)
    return total


def server_forward_consistency(
    model: torch.nn.Module,
    start_layer: int,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    output_attentions: bool,
    attention_layers: str = "all",
    detach_attentions: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, torch.Tensor]]]:
    batch, seq_len, _ = hidden_states.shape
    device = hidden_states.device
    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
    causal_mask = _prepare_4d_causal_attention_mask(attention_mask, (batch, seq_len), hidden_states, past_key_values_length=0)
    layers = list(range(start_layer, len(model.model.layers)))
    if attention_layers == "last2":
        selected = set(layers[-2:])
    elif attention_layers == "all":
        selected = set(layers)
    else:
        raise ValueError(f"unknown server_attention_layers={attention_layers!r}")
    collected: List[Tuple[int, torch.Tensor]] = []
    x = hidden_states
    for layer_index in layers:
        want_attention = output_attentions and layer_index in selected
        outputs = model.model.layers[layer_index](
            x,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=want_attention,
            use_cache=False,
        )
        x = outputs[0]
        if want_attention:
            attn = outputs[1]
            if attn is None:
                raise RuntimeError(f"server layer {layer_index} did not return attention")
            mean_attn = attn.float().mean(dim=1)[0]
            if detach_attentions:
                mean_attn = mean_attn.detach()
            collected.append((int(layer_index), mean_attn))
    final_hidden = model.model.norm(x)
    logits = model.lm_head(final_hidden)
    return final_hidden, logits, collected


def load_model(cfg: SACRConfig):
    set_seed(cfg.init_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer, model = load_tinyllama(dtype, local_files_only=cfg.local_files_only)
    model.to(device)
    if len(model.model.layers) != TOTAL_BLOCKS:
        raise RuntimeError(f"Expected {TOTAL_BLOCKS} TinyLlama blocks, got {len(model.model.layers)}")
    return tokenizer, model, device


def split_path(output_root: Path, prompt_split_seed: int) -> Path:
    return output_root / "splits" / f"prompt_split_{prompt_split_seed}.json"


def ensure_prompt_split(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.output_root)
    path = split_path(root, args.prompt_split_seed)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    prompts, meta = load_dataset_prompts(args.dataset_name, args.dataset_path, args.dataset_len, args.prompt_split_seed)
    ids = list(range(len(prompts)))
    rng = random.Random(args.prompt_split_seed)
    rng.shuffle(ids)
    split = {
        "dataset_name": args.dataset_name,
        "dataset_path": args.dataset_path,
        "dataset_len": args.dataset_len,
        "prompt_split_seed": args.prompt_split_seed,
        "dataset_meta": meta,
        "all_ids": ids,
        "tune_ids": ids[: args.tune_count],
        "holdout_ids": ids[args.tune_count : args.tune_count + args.holdout_count],
        "note": "IDs index the canonical Skytrax-28 prompt list selected with prompt_split_seed; init_seed is separate.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    json_dump(path, split)
    return split


def prompts_for_split(cfg: SACRConfig, output_root: Path) -> Tuple[List[Tuple[int, str]], Dict[str, Any]]:
    split = json.loads(split_path(output_root, cfg.prompt_split_seed).read_text(encoding="utf-8"))
    prompts, meta = load_dataset_prompts(cfg.dataset_name, cfg.dataset_path, cfg.dataset_len, cfg.prompt_split_seed)
    if cfg.split == "tune":
        ids = list(split["tune_ids"])
    elif cfg.split == "holdout":
        ids = list(split["holdout_ids"])
    elif cfg.split == "all":
        ids = list(split["all_ids"])
    else:
        raise ValueError(f"unknown split={cfg.split!r}")
    if cfg.prompt_limit > 0:
        ids = ids[: cfg.prompt_limit]
    return [(int(idx), prompts[int(idx)]) for idx in ids], {"canonical_dataset_meta": meta, "split": split, "selected_ids": ids}


def method_uses_tail(method: str) -> bool:
    return canonical_method(method) in {"tail_consistency", "tail_attention_consistency"}


def method_uses_attention(method: str) -> bool:
    return canonical_method(method) in {"attention_consistency", "tail_attention_consistency"}


def stage_b_optimize(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: SACRConfig,
    observed_activation: torch.Tensor,
    seq_len: int,
    device: torch.device,
    fixed_public: Dict[int, int],
    variable_audit: Any,
    init_embeds: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    method = canonical_method(cfg.method)
    embed_layer = model.get_input_embeddings()
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    start_layer = cfg.target_layer + 1
    if start_layer >= len(model.model.layers) and method in CONSISTENCY_METHODS:
        raise RuntimeError(f"target_layer={cfg.target_layer} leaves no public server-side layers")
    need_tail = method_uses_tail(method)
    need_attn = method_uses_attention(method)
    with torch.no_grad():
        ref_tail, _ref_logits, ref_attentions = server_forward_consistency(
            model,
            start_layer,
            observed_activation.to(dtype=next(model.parameters()).dtype),
            attention_mask,
            output_attentions=need_attn,
            attention_layers=cfg.server_attention_layers,
            detach_attentions=True,
        )
    fixed_embeds, fixed_positions, _ = fixed_embedding_tensor(embed_layer, fixed_public, seq_len, device)
    left, right = embedding_bounds(embed_layer.weight)
    left = left.to(device)
    right = right.to(device)
    z = init_embeds.detach().clone().to(device=device, dtype=torch.float32).requires_grad_(True)
    optimizer = torch.optim.AdamW([z], lr=cfg.lr)
    target = observed_activation.detach()
    history: List[Dict[str, Any]] = []
    grad_history: List[Dict[str, Any]] = []
    tail_initial: Optional[torch.Tensor] = None
    attn_initial: Optional[torch.Tensor] = None
    final: Dict[str, Any] = {}
    final_consistency: Dict[str, Any] = {
        "method": method,
        "server_start_layer": start_layer,
        "server_attention_layers": cfg.server_attention_layers,
        "reference_attention_layers": [idx for idx, _attn in ref_attentions],
    }
    for step in range(max(1, cfg.epoch)):
        enforce_embedding_constraints(z, fixed_positions, fixed_embeds, left, right)
        hidden = capture_prefix_activation(
            model,
            cfg.target_layer,
            inputs_embeds=z.to(dtype=embed_layer.weight.dtype),
            attention_mask=attention_mask,
        )
        base_loss = weighted_variable_activation_loss(hidden, target, variable_audit.variable_mask, None)
        all_valid_loss = uniform_all_token_activation_loss(hidden, target, attention_mask)
        vocab_loss = nearest_embedding_loss(variable_view(z, fixed_positions), embed_layer.weight)
        tail_loss = hidden.sum() * 0.0
        attn_loss = hidden.sum() * 0.0
        attn_stats: Dict[str, Any] = {"used_rows": 0, "skipped_rows": 0, "server_layers": []}
        if need_tail or need_attn:
            pred_tail, _pred_logits, pred_attentions = server_forward_consistency(
                model,
                start_layer,
                hidden.to(dtype=next(model.parameters()).dtype),
                attention_mask,
                output_attentions=need_attn,
                attention_layers=cfg.server_attention_layers,
                detach_attentions=False,
            )
            if need_tail:
                tail_loss = tail_consistency_loss(pred_tail, ref_tail, variable_audit.variable_mask)
            if need_attn:
                attn_loss, attn_stats = attention_js_consistency_loss(ref_attentions, pred_attentions, variable_audit.variable_mask)
        if step == 0:
            tail_initial = tail_loss.detach().float().clamp_min(cfg.eps)
            attn_initial = attn_loss.detach().float().clamp_min(cfg.eps)
            final_consistency["tail_initial"] = float(tail_initial.detach().cpu())
            final_consistency["attention_initial"] = float(attn_initial.detach().cpu())
        norm_tail = normalized_auxiliary_loss(tail_loss, tail_initial, cfg.eps)
        norm_attn = normalized_auxiliary_loss(attn_loss, attn_initial, cfg.eps)
        g_base = torch.autograd.grad(base_loss, z, retain_graph=True, allow_unused=True)[0]
        g_tail = torch.autograd.grad(norm_tail, z, retain_graph=True, allow_unused=True)[0] if need_tail else None
        g_attn = torch.autograd.grad(norm_attn, z, retain_graph=True, allow_unused=True)[0] if need_attn else None
        lambda_tail_eff = capped_auxiliary_lambda(g_base, g_tail, cfg.lambda_tail, cfg.rho_tail, cfg.eps) if need_tail else 0.0
        lambda_attn_eff = capped_auxiliary_lambda(g_base, g_attn, cfg.lambda_attn, cfg.rho_attn, cfg.eps) if need_attn else 0.0
        base_norm = tensor_norm(g_base)
        tail_norm = tensor_norm(g_tail)
        attn_norm = tensor_norm(g_attn)
        tail_cap_violation = lambda_tail_eff * tail_norm > cfg.rho_tail * base_norm + 1e-7
        attn_cap_violation = lambda_attn_eff * attn_norm > cfg.rho_attn * base_norm + 1e-7
        total = sacr_total_loss(
            base_loss,
            vocab_loss,
            tail_loss,
            attn_loss,
            tail_initial,
            attn_initial,
            cfg.lambda_vocab,
            lambda_tail_eff,
            lambda_attn_eff,
            cfg.eps,
        )
        cosine = F.cosine_similarity(hidden.float(), target.to(hidden.device).float(), dim=-1).mean()
        if not torch.isfinite(total) or not torch.isfinite(cosine):
            raise RuntimeError(f"NaN/Inf in SACR optimization at step {step + 1}")
        optimizer.zero_grad()
        total.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([z], cfg.grad_clip)
        optimizer.step()
        grad_row = {
            "step": step + 1,
            "base_grad_norm": base_norm,
            "tail_grad_norm": tail_norm,
            "attention_grad_norm": attn_norm,
            "lambda_tail_eff": float(lambda_tail_eff),
            "lambda_attn_eff": float(lambda_attn_eff),
            "cosine_base_tail": grad_cosine(g_base, g_tail),
            "cosine_base_attn": grad_cosine(g_base, g_attn),
            "tail_cap_violation": bool(tail_cap_violation),
            "attention_cap_violation": bool(attn_cap_violation),
        }
        grad_history.append(grad_row)
        final = {
            "activation_loss": float(base_loss.detach().cpu()),
            "all_token_uniform_loss": float(all_valid_loss.detach().cpu()),
            "variable_uniform_loss": float(base_loss.detach().cpu()),
            "tail_loss": float(tail_loss.detach().cpu()),
            "tail_loss_normalized": float(norm_tail.detach().cpu()),
            "attention_js_loss": float(attn_loss.detach().cpu()),
            "attention_js_loss_normalized": float(norm_attn.detach().cpu()),
            "vocab_loss": float(vocab_loss.detach().cpu()),
            "optimization_loss": float(total.detach().cpu()),
            "cosine_similarity": float(cosine.detach().cpu()),
            **grad_row,
            "attention_used_rows": int(attn_stats.get("used_rows", 0)),
            "attention_skipped_rows": int(attn_stats.get("skipped_rows", 0)),
        }
        history.append(final)
        if (step + 1) % max(1, min(50, cfg.epoch)) == 0 or step == cfg.epoch - 1:
            print(
                f"method={method} step={step + 1} base={final['activation_loss']:.6f} "
                f"tail={final['tail_loss']:.6f} attn={final['attention_js_loss']:.6f} "
                f"lt={lambda_tail_eff:.5f} la={lambda_attn_eff:.5f} total={final['optimization_loss']:.6f} "
                f"cosine={final['cosine_similarity']:.6f}",
                flush=True,
            )
    enforce_embedding_constraints(z, fixed_positions, fixed_embeds, left, right)
    if history:
        final_consistency.update(
            {
                "tail_final": history[-1]["tail_loss"],
                "attention_final": history[-1]["attention_js_loss"],
                "attention_used_rows_final": history[-1]["attention_used_rows"],
                "attention_skipped_rows_final": history[-1]["attention_skipped_rows"],
            }
        )
    return z.detach().float(), final, history, final_consistency, grad_history


def invert_observed(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: SACRConfig,
    observed_activation: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> Tuple[List[int], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Optional[float]]:
    cfg.method = canonical_method(cfg.method)
    set_seed(cfg.init_seed)
    embed_layer = model.get_input_embeddings()
    fixed_public = inferred_boundary_special_tokens(tokenizer, seq_len) if cfg.fix_boundary_specials else {}
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    variable_audit = build_variable_mask(
        attention_mask=attention_mask,
        fixed_public=fixed_public,
        special_token_ids=getattr(tokenizer, "all_special_ids", None),
        token_ids=None,
    )
    init_embeds = random_public_embeddings(tokenizer, embed_layer, seq_len, device, fixed_public)
    z, losses, history, consistency_stats, grad_history = stage_b_optimize(
        model, tokenizer, cfg, observed_activation, seq_len, device, fixed_public, variable_audit, init_embeds
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
    audit_payload = variable_mask_audit_payload(variable_audit)
    return recovered_ids, losses, audit_payload, consistency_stats, history, calibration_score


def validate_attack_api() -> Dict[str, Any]:
    checked = [
        invert_observed,
        stage_b_optimize,
        server_forward_consistency,
        attention_js_consistency_loss,
        tail_consistency_loss,
    ]
    sig = inspect.signature(invert_observed)
    names = list(sig.parameters)
    banned_fragments = ["prompt", "original", "input_ids", "token_ids", "reference", "text", "ground"]
    signature_hits = [name for name in names if any(fragment in name for fragment in banned_fragments)]
    banned_globals = {
        "prompt",
        "original_ids",
        "original_tokens",
        "input_ids",
        "reference_text",
        "ground_truth_embedding",
        "dummy_x",
        "dummy_attention",
        "gradient_token_reranking",
    }
    ast_hits: List[Dict[str, str]] = []
    dummy_hits: List[str] = []
    for fn in checked:
        source = inspect.getsource(fn)
        tree = ast.parse(source)
        lowered = source.lower()
        if "dummy_x" in lowered or "dummy attention" in lowered or "gradient token reranking" in lowered:
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
        "dummy_or_gradient_rerank_hits": dummy_hits,
        "passes": not signature_hits and not ast_hits and not dummy_hits,
    }


def ensure_jsonl_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def mean_std(values: Sequence[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {
        "mean": float(statistics.mean(vals)) if vals else None,
        "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0 if vals else None,
    }


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "token_accuracy": mean_std([row.get("token_accuracy") for row in rows]),
        "bleu": mean_std([row.get("bleu") for row in rows]),
        "runtime_seconds": mean_std([row.get("runtime_seconds") for row in rows]),
        "peak_gpu_memory_mb": mean_std([row.get("peak_gpu_memory_mb") for row in rows if row.get("peak_gpu_memory_mb") is not None]),
        "optimization_loss": mean_std([row.get("optimization_loss") for row in rows]),
        "cosine_similarity": mean_std([row.get("cosine_similarity") for row in rows]),
        "prompt_token_count": mean_std([row.get("prompt_token_count") for row in rows]),
        "sample_count": len(rows),
        "nerr": {"mean": None, "std": None, "status": "NERR unavailable"},
    }


def gradient_summary(gradient_samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for sample in gradient_samples:
        rows.extend(sample.get("history", []))
    return {
        "step_count": len(rows),
        "cap_violation_count": sum(1 for row in rows if row.get("tail_cap_violation") or row.get("attention_cap_violation")),
        "mean_lambda_tail_eff": mean_std([row.get("lambda_tail_eff") for row in rows])["mean"],
        "mean_lambda_attn_eff": mean_std([row.get("lambda_attn_eff") for row in rows])["mean"],
        "mean_base_grad_norm": mean_std([row.get("base_grad_norm") for row in rows])["mean"],
        "mean_tail_grad_norm": mean_std([row.get("tail_grad_norm") for row in rows])["mean"],
        "mean_attention_grad_norm": mean_std([row.get("attention_grad_norm") for row in rows])["mean"],
        "mean_cosine_base_tail": mean_std([row.get("cosine_base_tail") for row in rows if row.get("cosine_base_tail") is not None])["mean"],
        "mean_cosine_base_attn": mean_std([row.get("cosine_base_attn") for row in rows if row.get("cosine_base_attn") is not None])["mean"],
    }


def run_one_config(cfg: SACRConfig, output_root: Path, resume: bool) -> Dict[str, Any]:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "COMPLETE"
    if resume and done.exists():
        return json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    json_dump(out / "config.json", {"title": EXPERIMENT_TITLE, **cfg.__dict__})
    tokenizer, model, device = load_model(cfg)
    prompt_items, dataset_meta = prompts_for_split(cfg, output_root)
    json_dump(out / "dataset_meta.json", dataset_meta)
    predictions_path = out / "predictions.jsonl"
    failures_path = out / "failures.jsonl"
    if not resume:
        for path in [predictions_path, failures_path]:
            if path.exists():
                path.unlink()
    ensure_jsonl_file(predictions_path)
    ensure_jsonl_file(failures_path)
    completed = set()
    rows: List[Dict[str, Any]] = []
    if resume:
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append(row)
                completed.add(int(row["prompt_id"]))
    failures: List[Dict[str, Any]] = []
    loss_breakdown: List[Dict[str, Any]] = []
    grad_samples: List[Dict[str, Any]] = []
    consistency_samples: List[Dict[str, Any]] = []
    mask_audits: List[Dict[str, Any]] = []
    leakage_payload = validate_attack_api()
    for prompt_id, prompt in prompt_items:
        if prompt_id in completed:
            continue
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            tokenized = tokenizer(prompt, add_special_tokens=True, truncation=False, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            if input_ids.shape[1] > cfg.max_token_len:
                raise RuntimeError(f"prompt_id={prompt_id} has {input_ids.shape[1]} tokens > max_token_len={cfg.max_token_len}")
            original_ids = [int(x) for x in input_ids[0].detach().cpu().tolist()]
            with torch.no_grad():
                observed = capture_prefix_activation(model, cfg.target_layer, input_ids=input_ids, attention_mask=attention_mask).detach()
            preflight = {
                "model_name": MODEL_NAME,
                "block_count": len(model.model.layers),
                "target_layer_index": cfg.target_layer,
                "h_obs_semantics": f"output of 0-based TinyLlama block {cfg.target_layer}",
                "server_start_layer": cfg.target_layer + 1,
                "server_layer_indices": list(range(cfg.target_layer + 1, len(model.model.layers))),
                "observed_activation_shape": list(observed.shape),
                "sequence_length": int(input_ids.shape[1]),
                "prompt_id": int(prompt_id),
                "init_seed": cfg.init_seed,
                "attack_receives_ground_truth": False,
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "cpu"),
            }
            print(f"preflight={json.dumps(preflight, ensure_ascii=True)}", flush=True)
            recovered_ids, losses, mask_audit, consistency_stats, history, calibration_score = invert_observed(
                model, tokenizer, cfg, observed, int(input_ids.shape[1]), device
            )
            recovered_text = text_from_ids(tokenizer, recovered_ids)
            original_text = text_from_ids(tokenizer, original_ids)
            elapsed = time.time() - start_time
            row = {
                "title": EXPERIMENT_TITLE,
                "method": cfg.method,
                "prompt_id": int(prompt_id),
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
                "variable_uniform_loss": losses["variable_uniform_loss"],
                "all_token_uniform_loss": losses["all_token_uniform_loss"],
                "tail_loss": losses["tail_loss"],
                "attention_js_loss": losses["attention_js_loss"],
                "vocab_loss": losses["vocab_loss"],
                "cosine_similarity": losses["cosine_similarity"],
                "calibration_score": calibration_score,
                "runtime_seconds": elapsed,
                "elapsed_time": elapsed,
                "init_seed": cfg.init_seed,
                "prompt_split_seed": cfg.prompt_split_seed,
                "target_layer": cfg.target_layer,
                "prompt_token_count": len(original_ids),
                "peak_gpu_memory_mb": peak_memory_mb(),
                "preflight": preflight,
                "recovery_uses_ground_truth_tokens": False,
            }
            jsonl_append(predictions_path, row)
            rows.append(row)
            loss_breakdown.append({"prompt_id": int(prompt_id), "history": history, "final": losses})
            grad_samples.append({"prompt_id": int(prompt_id), "history": history})
            consistency_samples.append({"prompt_id": int(prompt_id), **consistency_stats})
            mask_audits.append({"prompt_id": int(prompt_id), **mask_audit})
            print(
                f"method={cfg.method} prompt_id={prompt_id} token_accuracy={row['token_accuracy']:.6f} "
                f"bleu={row['bleu']:.6f} elapsed={elapsed:.2f}s",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "method": cfg.method,
                "prompt_id": int(prompt_id),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "init_seed": cfg.init_seed,
                "target_layer": cfg.target_layer,
            }
            jsonl_append(failures_path, failure)
            failures.append(failure)
            print(f"failure={json.dumps(failure, ensure_ascii=True)}", flush=True)
    metrics = summarize_rows(rows)
    grad_summary = gradient_summary(grad_samples)
    metrics.update(
        {
            "title": EXPERIMENT_TITLE,
            "method": cfg.method,
            "run_name": cfg.run_name,
            "split": cfg.split,
            "init_seed": cfg.init_seed,
            "prompt_split_seed": cfg.prompt_split_seed,
            "target_layer": cfg.target_layer,
            "completed_sample_count": len(rows),
            "failed_sample_count": len(failures),
            "dataset_meta": dataset_meta,
            "top_k_embedding": cfg.top_k_embedding,
            "top_y_semantic": cfg.top_y_semantic,
            "output_dir": cfg.output_dir,
            "gradient_balance_summary": grad_summary,
        }
    )
    json_dump(out / "metrics.json", metrics)
    json_dump(out / "loss_breakdown.json", {"samples": loss_breakdown})
    json_dump(out / "gradient_balance_stats.json", {"samples": grad_samples, "summary": grad_summary})
    json_dump(out / "server_attention_consistency_stats.json", {"samples": consistency_samples})
    json_dump(out / "variable_mask_audit.json", {"samples": mask_audits})
    json_dump(
        out / "leakage_verification.json",
        {
            "attack_api_check": leakage_payload,
            "attack_api_passes": bool(leakage_payload.get("passes")),
            "manual_server_forward_checked_in_global_verify": True,
            "uses_dummy_attention": False,
            "uses_attention_rollout_as_token_weight": False,
            "uses_gradient_token_reranking": False,
        },
    )
    done.write_text("complete\n", encoding="utf-8")
    return metrics


def config_from_args(args: argparse.Namespace, method: str, run_name: str, output_dir: str) -> SACRConfig:
    canonical = canonical_method(method)
    return SACRConfig(
        method=canonical,
        run_name=run_name,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        dataset_len=args.dataset_len,
        prompt_split_seed=args.prompt_split_seed,
        split=args.split,
        prompt_limit=args.prompt_limit,
        init_seed=args.init_seed,
        participant_number=args.participant_number,
        attacker_position=args.attacker_position,
        inverted_block_count=args.target_layer + 1,
        target_layer=args.target_layer,
        epoch=args.epoch,
        lr=args.lr,
        lambda_vocab=args.lambda_vocab,
        lambda_tail=args.lambda_tail,
        lambda_attn=args.lambda_attn,
        rho_tail=args.rho_tail if args.rho_tail is not None else args.rho,
        rho_attn=args.rho_attn if args.rho_attn is not None else args.rho,
        top_k_embedding=args.k,
        top_y_semantic=args.y,
        max_token_len=args.max_token_len,
        grad_clip=args.grad_clip,
        server_attention_layers=args.server_attention_layers,
        adaptive_discretization=not args.naive_discretization,
        semantic_speculation=not args.disable_semantic_speculation,
        local_files_only=args.local_files_only,
    )


def run_name_for_task(task: SACRTask, args: argparse.Namespace) -> str:
    if task.run_name:
        return task.run_name
    method = canonical_method(task.method)
    pieces = [
        method,
        f"split{task.split}",
        f"init{task.init_seed}",
        f"layer{args.target_layer}",
        f"epoch{args.epoch}",
        f"k{args.k}",
    ]
    if method in {"tail_consistency", "tail_attention_consistency"}:
        pieces.append(f"lt{float_tag(task.lambda_tail)}")
        pieces.append(f"rt{float_tag(task.rho_tail)}")
    if method in {"attention_consistency", "tail_attention_consistency"}:
        pieces.append(f"la{float_tag(task.lambda_attn)}")
        pieces.append(f"ra{float_tag(task.rho_attn)}")
        pieces.append(f"attn{task.server_attention_layers}")
    return "_".join(pieces)


def build_single_command(args: argparse.Namespace, task: SACRTask) -> List[str]:
    run_name = run_name_for_task(task, args)
    output_dir = str(Path(args.output_root) / task.output_subdir / run_name)
    return [
        sys.executable,
        "pia_server_attention_consistency_pia.py",
        "--mode",
        "single",
        "--method",
        task.method,
        "--output-root",
        args.output_root,
        "--output-dir",
        output_dir,
        "--dataset-name",
        args.dataset_name,
        "--dataset-path",
        args.dataset_path,
        "--dataset-len",
        str(args.dataset_len),
        "--prompt-split-seed",
        str(args.prompt_split_seed),
        "--split",
        task.split,
        "--prompt-limit",
        str(task.prompt_limit),
        "--init-seed",
        str(task.init_seed),
        "--participant-number",
        str(args.participant_number),
        "--attacker-position",
        str(args.attacker_position),
        "--target-layer",
        str(args.target_layer),
        "--epoch",
        str(args.epoch),
        "--lr",
        str(args.lr),
        "--lambda-vocab",
        str(args.lambda_vocab),
        "--lambda-tail",
        str(task.lambda_tail),
        "--lambda-attn",
        str(task.lambda_attn),
        "--rho-tail",
        str(task.rho_tail),
        "--rho-attn",
        str(task.rho_attn),
        "--k",
        str(args.k),
        "--y",
        str(args.y),
        "--max-token-len",
        str(args.max_token_len),
        "--grad-clip",
        str(args.grad_clip),
        "--server-attention-layers",
        task.server_attention_layers,
    ] + (["--local-files-only"] if args.local_files_only else []) + (["--resume"] if args.resume else [])


def gpu_slots(args: argparse.Namespace) -> List[int]:
    memories = query_gpu_memory()
    plan = plan_auto_batch_jobs(
        memories,
        min_free_mb_per_job=args.batch_free_mb_per_job,
        reserve_free_mb=args.batch_reserve_free_mb,
        max_jobs_per_gpu=args.max_jobs_per_gpu,
        max_total_jobs=args.max_parallel_jobs,
    )
    slots: List[int] = []
    for gpu_index, count in sorted(plan.jobs_by_gpu.items()):
        slots.extend([int(gpu_index)] * int(count))
    print(json.dumps({"gpu_memories": [m.__dict__ for m in memories], "jobs_by_gpu": plan.jobs_by_gpu}, ensure_ascii=False), flush=True)
    return slots or [0]


def run_task_queue(args: argparse.Namespace, tasks: Sequence[SACRTask]) -> None:
    if not tasks:
        return
    slots = gpu_slots(args)
    assigned = []
    for idx, task in enumerate(tasks):
        assigned.append(SACRTask(**{**task.__dict__, "gpu_index": slots[idx % len(slots)]}))
    if args.dry_run:
        for task in assigned:
            print(f"CUDA_VISIBLE_DEVICES={task.gpu_index}", " ".join(build_single_command(args, task)))
        return
    log_dir = Path(args.output_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    queue = list(assigned)
    active: List[Tuple[subprocess.Popen, SACRTask, Any, Path]] = []
    capacity = {gpu: slots.count(gpu) for gpu in set(slots)}
    while queue or active:
        launched = True
        while queue and launched:
            launched = False
            active_by_gpu: Dict[int, int] = {}
            for _proc, active_task, _file, _path in active:
                active_by_gpu[active_task.gpu_index] = active_by_gpu.get(active_task.gpu_index, 0) + 1
            for idx, task in enumerate(queue):
                if active_by_gpu.get(task.gpu_index, 0) < capacity.get(task.gpu_index, 0):
                    queue.pop(idx)
                    launched = True
                    break
            if not launched:
                break
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(task.gpu_index)
            if args.transformers_cache:
                env.setdefault("TRANSFORMERS_CACHE", args.transformers_cache)
            command = build_single_command(args, task)
            run_name = run_name_for_task(task, args)
            log_path = log_dir / f"{run_name}_gpu{task.gpu_index}.log"
            log_file = log_path.open("w", encoding="utf-8")
            print(json.dumps({"launch": command, "gpu": task.gpu_index, "log": str(log_path)}, ensure_ascii=False), flush=True)
            proc = subprocess.Popen(command, cwd=Path.cwd(), env=env, stdout=log_file, stderr=subprocess.STDOUT)
            active.append((proc, task, log_file, log_path))
        next_active: List[Tuple[subprocess.Popen, SACRTask, Any, Path]] = []
        for proc, task, log_file, log_path in active:
            code = proc.poll()
            if code is None:
                next_active.append((proc, task, log_file, log_path))
                continue
            log_file.close()
            if code != 0:
                raise RuntimeError(f"SACR task failed method={task.method} split={task.split} seed={task.init_seed} log={log_path}")
        active = next_active
        if active:
            time.sleep(5)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_dirs(root: Path, subdir: str) -> List[Path]:
    base = root / subdir
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir() and (p / "metrics.json").exists()])


def summarize_run_dir(path: Path) -> Dict[str, Any]:
    metrics = read_json(path / "metrics.json")
    grad = read_json(path / "gradient_balance_stats.json").get("summary", {})
    cfg = read_json(path / "config.json")
    return {
        "run_name": path.name,
        "method": metrics.get("method"),
        "split": metrics.get("split"),
        "init_seed": metrics.get("init_seed"),
        "target_layer": metrics.get("target_layer"),
        "lambda_tail": cfg.get("lambda_tail"),
        "lambda_attn": cfg.get("lambda_attn"),
        "rho_tail": cfg.get("rho_tail"),
        "rho_attn": cfg.get("rho_attn"),
        "server_attention_layers": cfg.get("server_attention_layers"),
        "token_accuracy_mean": metric_mean(metrics, "token_accuracy"),
        "bleu_mean": metric_mean(metrics, "bleu"),
        "completed_sample_count": metrics.get("completed_sample_count", 0),
        "failed_sample_count": metrics.get("failed_sample_count", 0),
        "cap_violation_count": grad.get("cap_violation_count", 0),
        "mean_lambda_tail_eff": grad.get("mean_lambda_tail_eff"),
        "mean_lambda_attn_eff": grad.get("mean_lambda_attn_eff"),
        "mean_cosine_base_tail": grad.get("mean_cosine_base_tail"),
        "mean_cosine_base_attn": grad.get("mean_cosine_base_attn"),
        "has_nan": has_nan(metrics) or has_nan(grad),
        "output_dir": str(path),
    }


def has_nan(value: Any) -> bool:
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    if isinstance(value, dict):
        return any(has_nan(v) for v in value.values())
    if isinstance(value, list):
        return any(has_nan(v) for v in value)
    return False


def paired_delta(baseline: Sequence[Dict[str, Any]], candidate: Sequence[Dict[str, Any]], bootstrap_iters: int, seed: int) -> Dict[str, Any]:
    base_by_id = {int(row["prompt_id"]): row for row in baseline}
    cand_by_id = {int(row["prompt_id"]): row for row in candidate}
    ids = sorted(set(base_by_id) & set(cand_by_id))
    acc = [float(cand_by_id[i]["token_accuracy"]) - float(base_by_id[i]["token_accuracy"]) for i in ids]
    bleu = [float(cand_by_id[i]["bleu"]) - float(base_by_id[i]["bleu"]) for i in ids]
    wins = sum(1 for value in acc if value > 1e-12)
    losses = sum(1 for value in acc if value < -1e-12)
    ties = len(acc) - wins - losses
    acc_low, acc_high = bootstrap_mean_ci(acc, bootstrap_iters, seed)
    bleu_low, bleu_high = bootstrap_mean_ci(bleu, bootstrap_iters, seed + 19)
    return {
        "pair_count": len(ids),
        "mean_token_accuracy_delta": statistics.mean(acc) if acc else None,
        "mean_bleu_delta": statistics.mean(bleu) if bleu else None,
        "win_count": wins,
        "tie_count": ties,
        "loss_count": losses,
        "token_accuracy_ci_low": acc_low,
        "token_accuracy_ci_high": acc_high,
        "bleu_ci_low": bleu_low,
        "bleu_ci_high": bleu_high,
    }


def write_basic_plot(path: Path, rows: Sequence[Dict[str, Any]], x_key: str, y_key: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt

        labels = [str(row.get(x_key))[:32] for row in rows]
        values = [float(row.get(y_key) or 0.0) for row in rows]
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.7), 4))
        ax.bar(range(len(labels)), values)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylabel(y_key)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
    except Exception:
        path.write_text("plot unavailable\n", encoding="utf-8")


def summarize_smoke(root: Path) -> List[Dict[str, Any]]:
    rows = [summarize_run_dir(path) for path in run_dirs(root, "smoke")]
    b0v = next((row for row in rows if row["method"] == "variable_only_uniform"), None)
    for row in rows:
        if b0v and row["method"] != "variable_only_uniform":
            row["token_accuracy_delta_vs_B0V"] = (row["token_accuracy_mean"] or 0.0) - (b0v["token_accuracy_mean"] or 0.0)
            row["bleu_delta_vs_B0V"] = (row["bleu_mean"] or 0.0) - (b0v["bleu_mean"] or 0.0)
            row["smoke_stop_method"] = row["token_accuracy_delta_vs_B0V"] < -0.03
        else:
            row["token_accuracy_delta_vs_B0V"] = 0.0
            row["bleu_delta_vs_B0V"] = 0.0
            row["smoke_stop_method"] = False
    write_csv(root / "comparison.csv", rows)
    write_csv(root / "reports" / "smoke_summary.csv", rows)
    lines = ["# SACR-PIA Smoke Summary", "", "| method | Acc | BLEU | dAcc vs B0V | cap violations | stop method |", "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(
            f"| {row['method']} | {fmt_md(row['token_accuracy_mean'])} | {fmt_md(row['bleu_mean'])} | {fmt_md(row['token_accuracy_delta_vs_B0V'])} | {row['cap_violation_count']} | {row['smoke_stop_method']} |"
        )
    (root / "reports" / "smoke_summary.md").parent.mkdir(parents=True, exist_ok=True)
    (root / "reports" / "smoke_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def summarize_tune(root: Path, bootstrap_iters: int = 10000) -> List[Dict[str, Any]]:
    dirs = run_dirs(root, "tune")
    rows = [summarize_run_dir(path) for path in dirs]
    preds = {path.name: read_jsonl(path / "predictions.jsonl") for path in dirs}
    b0v_by_seed = {row["init_seed"]: row for row in rows if row["method"] == "variable_only_uniform"}
    b0v_pred_by_seed = {
        summarize_run_dir(path)["init_seed"]: read_jsonl(path / "predictions.jsonl")
        for path in dirs
        if summarize_run_dir(path)["method"] == "variable_only_uniform"
    }
    for row in rows:
        if row["method"] == "variable_only_uniform":
            row.update({"mean_token_accuracy_delta_vs_B0V": 0.0, "mean_bleu_delta_vs_B0V": 0.0, "win_count": 0, "tie_count": row["completed_sample_count"], "loss_count": 0})
            continue
        base = b0v_by_seed.get(row["init_seed"])
        if base:
            pair = paired_delta(b0v_pred_by_seed[row["init_seed"]], preds[row["run_name"]], bootstrap_iters, int(row["init_seed"] or 0))
            row.update({**pair, "mean_token_accuracy_delta_vs_B0V": pair["mean_token_accuracy_delta"], "mean_bleu_delta_vs_B0V": pair["mean_bleu_delta"]})
    write_csv(root / "tune_ablation.csv", rows)
    write_csv(root / "reports" / "tune_ablation.csv", rows)
    candidates = choose_tune_candidates(rows)
    write_csv(root / "reports" / "tune_candidates.csv", candidates)
    write_basic_plot(root / "token_accuracy_comparison.png", rows, "run_name", "token_accuracy_mean", "SACR tune token accuracy")
    write_basic_plot(root / "gradient_norm_ratio.png", rows, "run_name", "mean_lambda_tail_eff", "SACR effective tail lambda")
    write_basic_plot(root / "attention_js_curve.png", rows, "run_name", "mean_lambda_attn_eff", "SACR effective attention lambda")
    write_basic_plot(root / "tail_consistency_curve.png", rows, "run_name", "mean_cosine_base_tail", "SACR base-tail cosine")
    return rows


def choose_tune_candidates(rows: Sequence[Dict[str, Any]], max_candidates: int = 2) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        method = row.get("method")
        if method == "variable_only_uniform":
            continue
        key = (
            method,
            row.get("lambda_tail"),
            row.get("lambda_attn"),
            row.get("rho_tail"),
            row.get("rho_attn"),
            row.get("server_attention_layers"),
        )
        grouped.setdefault(key, []).append(row)
    summary: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        if len({item.get("init_seed") for item in items}) < 2:
            continue
        total_n = sum(int(item.get("pair_count") or item.get("completed_sample_count") or 0) for item in items)
        if total_n <= 0:
            continue
        pooled_acc = sum((item.get("mean_token_accuracy_delta_vs_B0V") or 0.0) * int(item.get("pair_count") or item.get("completed_sample_count") or 0) for item in items) / total_n
        pooled_bleu = sum((item.get("mean_bleu_delta_vs_B0V") or 0.0) * int(item.get("pair_count") or item.get("completed_sample_count") or 0) for item in items) / total_n
        passed = (
            pooled_acc >= 0.0
            and pooled_bleu >= 0.0
            and all(int(item.get("failed_sample_count") or 0) == 0 for item in items)
            and all(int(item.get("cap_violation_count") or 0) == 0 for item in items)
            and not any(bool(item.get("has_nan")) for item in items)
        )
        summary.append(
            {
                "method": key[0],
                "lambda_tail": key[1],
                "lambda_attn": key[2],
                "rho_tail": key[3],
                "rho_attn": key[4],
                "server_attention_layers": key[5],
                "pooled_token_accuracy_delta_vs_B0V": pooled_acc,
                "pooled_bleu_delta_vs_B0V": pooled_bleu,
                "win_count": sum(int(item.get("win_count") or 0) for item in items),
                "tie_count": sum(int(item.get("tie_count") or 0) for item in items),
                "loss_count": sum(int(item.get("loss_count") or 0) for item in items),
                "mean_lambda_eff": statistics.mean(
                    [
                        float(item.get("mean_lambda_tail_eff") or 0.0) + float(item.get("mean_lambda_attn_eff") or 0.0)
                        for item in items
                    ]
                ),
                "passed_tune_filter": passed,
            }
        )
    summary.sort(
        key=lambda row: (
            bool(row["passed_tune_filter"]),
            row["pooled_token_accuracy_delta_vs_B0V"],
            row["pooled_bleu_delta_vs_B0V"],
            -float(row["mean_lambda_eff"] or 0.0),
        ),
        reverse=True,
    )
    return [row for row in summary if row["passed_tune_filter"]][:max_candidates]


def summarize_holdout(root: Path, bootstrap_iters: int = 10000) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    dirs = run_dirs(root, "holdout")
    rows = [summarize_run_dir(path) for path in dirs]
    preds = {path.name: read_jsonl(path / "predictions.jsonl") for path in dirs}
    b0v_pred_by_seed = {
        summarize_run_dir(path)["init_seed"]: read_jsonl(path / "predictions.jsonl")
        for path in dirs
        if summarize_run_dir(path)["method"] == "variable_only_uniform"
    }
    paired_rows: List[Dict[str, Any]] = []
    for path in dirs:
        row = summarize_run_dir(path)
        if row["method"] == "variable_only_uniform":
            continue
        seed = row["init_seed"]
        pair = paired_delta(b0v_pred_by_seed.get(seed, []), preds[path.name], bootstrap_iters, int(seed or 0))
        paired = {**row, **pair, "mean_token_accuracy_delta_vs_B0V": pair["mean_token_accuracy_delta"], "mean_bleu_delta_vs_B0V": pair["mean_bleu_delta"]}
        paired_rows.append(paired)
    write_csv(root / "paired_summary.csv", paired_rows)
    write_csv(root / "reports" / "paired_summary.csv", paired_rows)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in paired_rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    summary_rows: List[Dict[str, Any]] = []
    promising: List[str] = []
    for method, items in grouped.items():
        total_n = sum(int(item.get("pair_count") or 0) for item in items)
        pooled_acc = sum((item.get("mean_token_accuracy_delta_vs_B0V") or 0.0) * int(item.get("pair_count") or 0) for item in items) / total_n if total_n else None
        pooled_bleu = sum((item.get("mean_bleu_delta_vs_B0V") or 0.0) * int(item.get("pair_count") or 0) for item in items) / total_n if total_n else None
        acc_deltas = [item.get("mean_token_accuracy_delta_vs_B0V") or 0.0 for item in items]
        pooled_acc_low, pooled_acc_high = bootstrap_mean_ci(acc_deltas, bootstrap_iters, 7700)
        win = sum(int(item.get("win_count") or 0) for item in items)
        loss = sum(int(item.get("loss_count") or 0) for item in items)
        stop_reasons = []
        if sum(1 for value in acc_deltas if value < 0.0) >= 2:
            stop_reasons.append("two init seeds have negative Token Accuracy delta")
        if pooled_acc is None or pooled_acc <= 0.0:
            stop_reasons.append("pooled Token Accuracy delta <= 0")
        if pooled_bleu is None or pooled_bleu < 0.0:
            stop_reasons.append("pooled BLEU delta < 0")
        if pooled_acc_high is not None and pooled_acc_high <= 0.0:
            stop_reasons.append("pooled Token Accuracy CI upper <= 0")
        if win <= loss:
            stop_reasons.append("win count <= loss count")
        if any(item.get("has_nan") or int(item.get("failed_sample_count") or 0) > 0 or int(item.get("cap_violation_count") or 0) > 0 for item in items):
            stop_reasons.append("NaN, failed sample, or gradient cap violation")
        mean_eff = statistics.mean([float(item.get("mean_lambda_tail_eff") or 0.0) + float(item.get("mean_lambda_attn_eff") or 0.0) for item in items])
        if mean_eff < 1e-8 and method != "variable_only_uniform":
            stop_reasons.append("mean effective lambda is near zero")
        is_promising = (
            pooled_acc is not None
            and pooled_acc > 0.0
            and pooled_bleu is not None
            and pooled_bleu >= 0.0
            and win > loss
            and pooled_acc_low is not None
            and pooled_acc_low > 0.0
            and not stop_reasons
        )
        if is_promising:
            promising.append(method)
        summary_rows.append(
            {
                "method": method,
                "pooled_token_accuracy_delta_vs_B0V": pooled_acc,
                "pooled_bleu_delta_vs_B0V": pooled_bleu,
                "token_accuracy_ci_low": pooled_acc_low,
                "token_accuracy_ci_high": pooled_acc_high,
                "win_count": win,
                "tie_count": sum(int(item.get("tie_count") or 0) for item in items),
                "loss_count": loss,
                "mean_effective_lambda": mean_eff,
                "stop": bool(stop_reasons),
                "stop_reasons": "; ".join(stop_reasons),
                "promising": is_promising,
            }
        )
    write_csv(root / "holdout_summary.csv", summary_rows)
    write_csv(root / "reports" / "holdout_summary.csv", summary_rows)
    decision = {"promising_methods": promising, "allow_multilayer_or_skytrax150": bool(promising)}
    lines = ["# SACR-PIA Holdout Summary", "", "| method | pooled dAcc | pooled dBLEU | CI | win/tie/loss | stop | promising |", "|---|---:|---:|---|---|---|---|"]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {fmt_md(row['pooled_token_accuracy_delta_vs_B0V'])} | {fmt_md(row['pooled_bleu_delta_vs_B0V'])} | [{fmt_md(row['token_accuracy_ci_low'])}, {fmt_md(row['token_accuracy_ci_high'])}] | {row['win_count']}/{row['tie_count']}/{row['loss_count']} | {row['stop']} | {row['promising']} |"
        )
    lines.append("")
    lines.append(f"- allow_multilayer_or_skytrax150: {decision['allow_multilayer_or_skytrax150']}")
    (root / "reports" / "holdout_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_rows, decision


def build_smoke_tasks() -> List[SACRTask]:
    return [
        SACRTask("B0V", "tune", 42, "smoke", 3, run_name="B0V_smoke_tune3_init42"),
        SACRTask("C1", "tune", 42, "smoke", 3, lambda_tail=0.05, rho_tail=0.10, run_name="C1_smoke_tune3_init42"),
        SACRTask("C2", "tune", 42, "smoke", 3, lambda_attn=0.05, rho_attn=0.10, server_attention_layers="all", run_name="C2_smoke_tune3_init42"),
        SACRTask("C3", "tune", 42, "smoke", 3, lambda_tail=0.05, lambda_attn=0.05, rho_tail=0.10, rho_attn=0.10, server_attention_layers="all", run_name="C3_smoke_tune3_init42"),
    ]


def build_tune_tasks(root: Path) -> List[SACRTask]:
    stopped = {
        row["method"]
        for row in summarize_smoke(root)
        if row.get("smoke_stop_method") and row.get("method") in CONSISTENCY_METHODS
    }
    tasks: List[SACRTask] = []
    for seed in [42, 43]:
        tasks.append(SACRTask("B0V", "tune", seed, "tune", 0))
    if "tail_consistency" not in stopped:
        for seed in [42, 43]:
            for lam in [0.01, 0.03, 0.05]:
                for rho in [0.05, 0.10]:
                    tasks.append(SACRTask("C1", "tune", seed, "tune", 0, lambda_tail=lam, rho_tail=rho))
    if "attention_consistency" not in stopped:
        for seed in [42, 43]:
            for lam in [0.01, 0.03, 0.05]:
                for rho in [0.05, 0.10]:
                    for layers in ["all", "last2"]:
                        tasks.append(SACRTask("C2", "tune", seed, "tune", 0, lambda_attn=lam, rho_attn=rho, server_attention_layers=layers))
    return tasks


def build_c3_tune_tasks(root: Path) -> List[SACRTask]:
    rows = summarize_tune(root)
    best = choose_tune_candidates(rows, max_candidates=10)
    best_c1 = next((row for row in best if row["method"] == "tail_consistency"), None)
    best_c2 = next((row for row in best if row["method"] == "attention_consistency"), None)
    if not best_c1 or not best_c2:
        return []
    tasks: List[SACRTask] = []
    for seed in [42, 43]:
        tasks.append(
            SACRTask(
                "C3",
                "tune",
                seed,
                "tune",
                0,
                lambda_tail=float(best_c1["lambda_tail"]),
                lambda_attn=float(best_c2["lambda_attn"]),
                rho_tail=float(best_c1["rho_tail"]),
                rho_attn=float(best_c2["rho_attn"]),
                server_attention_layers=str(best_c2["server_attention_layers"]),
            )
        )
    return tasks


def build_holdout_tasks(root: Path) -> List[SACRTask]:
    rows = summarize_tune(root)
    candidates = choose_tune_candidates(rows, max_candidates=2)
    tasks: List[SACRTask] = []
    for seed in [101, 102, 103]:
        tasks.append(SACRTask("B0V", "holdout", seed, "holdout", 0))
        for cand in candidates:
            method_alias = {
                "tail_consistency": "C1",
                "attention_consistency": "C2",
                "tail_attention_consistency": "C3",
            }[str(cand["method"])]
            tasks.append(
                SACRTask(
                    method_alias,
                    "holdout",
                    seed,
                    "holdout",
                    0,
                    lambda_tail=float(cand.get("lambda_tail") or 0.0),
                    lambda_attn=float(cand.get("lambda_attn") or 0.0),
                    rho_tail=float(cand.get("rho_tail") or 0.10),
                    rho_attn=float(cand.get("rho_attn") or 0.10),
                    server_attention_layers=str(cand.get("server_attention_layers") or "all"),
                )
            )
    return tasks


def verify_no_leakage(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_prompt_split(args)
    cfg = config_from_args(args, args.method, "leakage_verification", str(Path(args.output_root) / "leakage_verification" / "sample"))
    cfg.split = "tune"
    cfg.prompt_limit = 1
    tokenizer, model, device = load_model(cfg)
    prompt_items, dataset_meta = prompts_for_split(cfg, Path(args.output_root))
    _prompt_id, prompt = prompt_items[0]
    tokenized = tokenizer(prompt, add_special_tokens=True, truncation=False, return_tensors="pt")
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    with torch.no_grad():
        observed = capture_prefix_activation(model, cfg.target_layer, input_ids=input_ids, attention_mask=attention_mask).detach()
        full = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        full_boundary = full.hidden_states[cfg.target_layer + 1].detach()
        server_tail, server_logits, collected = server_forward_consistency(
            model,
            cfg.target_layer + 1,
            observed.to(dtype=next(model.parameters()).dtype),
            attention_mask,
            output_attentions=True,
            attention_layers=cfg.server_attention_layers,
            detach_attentions=True,
        )
    boundary_diff = float((observed.float() - full_boundary.float()).abs().max().detach().cpu())
    tail_diff = float((server_tail.float() - full.hidden_states[-1].float()).abs().max().detach().cpu())
    logits_diff = float((server_logits.float() - full.logits.float()).abs().max().detach().cpu())
    api = validate_attack_api()
    result = {
        "title": EXPERIMENT_TITLE,
        "dataset_meta": dataset_meta,
        "target_layer": cfg.target_layer,
        "h_obs_semantics": f"H_obs is output of 0-based TinyLlama block {cfg.target_layer}; server starts at block {cfg.target_layer + 1}",
        "server_layers": [idx for idx, _attn in collected],
        "activation_shape": list(observed.shape),
        "boundary_vs_full_hidden_max_abs_diff": boundary_diff,
        "server_final_hidden_vs_full_max_abs_diff": tail_diff,
        "server_logits_vs_full_max_abs_diff": logits_diff,
        "manual_server_forward_matches_full": tail_diff < 5e-3 and logits_diff < 5e-3 and boundary_diff < 5e-3,
        "attack_api_check": api,
        "uses_attention_rollout_as_token_weight": False,
        "uses_dummy_attention": False,
        "uses_gradient_token_reranking": False,
    }
    out = Path(args.output_root) / "leakage_verification"
    out.mkdir(parents=True, exist_ok=True)
    json_dump(out / "leakage_verification.json", result)
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return result


def write_design() -> None:
    Path("analysis").mkdir(parents=True, exist_ok=True)
    text = """# Server-Attention Consistency PIA Design

SACR-PIA freezes the negative MTS finding: server attention rollout is no longer used as direct token-loss weights. The main reconstruction objective remains B0V variable-only uniform activation matching.

Auxiliary server information is used only as a consistency regularizer:

- C1 adds server tail hidden-state consistency.
- C2 adds server attention Jensen-Shannon consistency over variable query/key positions.
- C3 combines C1 and C2.

The attack API accepts only model, tokenizer, cfg, observed activation H_obs, sequence length, and device. It does not accept prompt text, original ids/tokens, input ids, ground-truth embeddings, dummy_x, dummy attention, or client-side true attention.

Prompt split and optimization initialization are separated by `--prompt-split-seed` and `--init-seed`. The canonical split is saved under `runs/server_attention_consistency_pia/splits/`.

Auxiliary losses are normalized by their detached step-0 value. Per-step gradient caps enforce `lambda_eff * ||g_aux|| <= rho * ||g_base||`; effective lambdas, gradient norms, cosines, and cap violations are written for every run.
"""
    (Path("analysis") / "server_attention_consistency_design.md").write_text(text, encoding="utf-8")


def write_report(root: Path) -> None:
    Path("analysis").mkdir(parents=True, exist_ok=True)
    smoke = read_json(root / "reports" / "smoke_summary.json")
    tune_candidates = []
    cand_path = root / "reports" / "tune_candidates.csv"
    if cand_path.exists():
        with cand_path.open("r", encoding="utf-8", newline="") as handle:
            tune_candidates = list(csv.DictReader(handle))
    holdout = []
    holdout_path = root / "holdout_summary.csv"
    if holdout_path.exists():
        with holdout_path.open("r", encoding="utf-8", newline="") as handle:
            holdout = list(csv.DictReader(handle))
    lines = [
        "# Server-Attention Consistency PIA Report",
        "",
        "MTS direct server-attention token weighting is frozen as a negative/inconclusive path. SACR-PIA instead keeps B0V as the main objective and uses server-side tail/attention consistency as capped auxiliary regularization.",
        "",
        f"- output root: `{root}`",
        "- no automatic multi-layer or Skytrax-150 expansion is allowed unless holdout is promising.",
        "",
        "## Tune Candidates",
        "",
    ]
    if tune_candidates:
        lines.append("| method | pooled dAcc vs B0V | pooled dBLEU vs B0V | lambda_tail | lambda_attn | layers |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for row in tune_candidates:
            lines.append(
                f"| {row.get('method')} | {fmt_md(row.get('pooled_token_accuracy_delta_vs_B0V'))} | {fmt_md(row.get('pooled_bleu_delta_vs_B0V'))} | {row.get('lambda_tail')} | {row.get('lambda_attn')} | {row.get('server_attention_layers')} |"
            )
    else:
        lines.append("No tune candidate passed the non-negative B0V filter.")
    lines += ["", "## Holdout", ""]
    if holdout:
        lines.append("| method | pooled dAcc | pooled dBLEU | win/tie/loss | stop | promising |")
        lines.append("|---|---:|---:|---|---|---|")
        for row in holdout:
            lines.append(
                f"| {row.get('method')} | {fmt_md(row.get('pooled_token_accuracy_delta_vs_B0V'))} | {fmt_md(row.get('pooled_bleu_delta_vs_B0V'))} | {row.get('win_count')}/{row.get('tie_count')}/{row.get('loss_count')} | {row.get('stop')} | {row.get('promising')} |"
            )
    else:
        lines.append("Holdout did not run because no candidate passed tuning, or it has not completed yet.")
    lines.append("")
    lines.append("Do not claim stable improvement unless the holdout promising criteria are met.")
    (Path("analysis") / "server_attention_consistency_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_single(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_prompt_split(args)
    method = canonical_method(args.method)
    run_name = args.run_name or "_".join(
        [
            method,
            f"split{args.split}",
            f"init{args.init_seed}",
            f"layer{args.target_layer}",
            f"epoch{args.epoch}",
            f"k{args.k}",
        ]
    )
    output_dir = args.output_dir or str(Path(args.output_root) / "single" / run_name)
    cfg = config_from_args(args, method, run_name, output_dir)
    return run_one_config(cfg, Path(args.output_root), args.resume)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["split", "verify", "single", "smoke", "tune", "holdout", "summarize", "write-design"], default="single")
    parser.add_argument("--method", choices=METHODS, default="B0V")
    parser.add_argument("--output-root", default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dataset-name", default="Skytrax-28")
    parser.add_argument("--dataset-path", default="data/airline.json")
    parser.add_argument("--dataset-len", type=int, default=28)
    parser.add_argument("--prompt-split-seed", type=int, default=20260705)
    parser.add_argument("--tune-count", type=int, default=4)
    parser.add_argument("--holdout-count", type=int, default=4)
    parser.add_argument("--split", choices=["tune", "holdout", "all"], default="tune")
    parser.add_argument("--prompt-limit", type=int, default=0)
    parser.add_argument("--init-seed", type=int, default=42)
    parser.add_argument("--participant-number", type=int, default=4)
    parser.add_argument("--attacker-position", type=int, default=4)
    parser.add_argument("--target-layer", type=int, default=17)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lambda-vocab", type=float, default=0.1)
    parser.add_argument("--lambda-tail", type=float, default=0.05)
    parser.add_argument("--lambda-attn", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=0.10)
    parser.add_argument("--rho-tail", type=float, default=None)
    parser.add_argument("--rho-attn", type=float, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--y", type=int, default=10)
    parser.add_argument("--max-token-len", type=int, default=896)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--server-attention-layers", choices=["all", "last2"], default="all")
    parser.add_argument("--naive-discretization", action="store_true")
    parser.add_argument("--disable-semantic-speculation", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-free-mb-per-job", type=int, default=4096)
    parser.add_argument("--batch-reserve-free-mb", type=int, default=2048)
    parser.add_argument("--max-jobs-per-gpu", type=int, default=1)
    parser.add_argument("--max-parallel-jobs", type=int, default=4)
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--transformers-cache", default=os.environ.get("TRANSFORMERS_CACHE", "/data/zhiqi/hf_cache/transformers"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.transformers_cache:
        os.environ.setdefault("TRANSFORMERS_CACHE", args.transformers_cache)
    root = Path(args.output_root)
    write_design()
    if args.mode == "split":
        print(json.dumps(ensure_prompt_split(args), ensure_ascii=False, indent=2))
    elif args.mode == "verify":
        ensure_prompt_split(args)
        verify_no_leakage(args)
    elif args.mode == "single":
        run_single(args)
    elif args.mode == "smoke":
        ensure_prompt_split(args)
        run_task_queue(args, build_smoke_tasks())
        rows = summarize_smoke(root)
        json_dump(root / "reports" / "smoke_summary.json", {"rows": rows})
        write_report(root)
    elif args.mode == "tune":
        ensure_prompt_split(args)
        run_task_queue(args, build_tune_tasks(root))
        c3_tasks = build_c3_tune_tasks(root)
        if c3_tasks:
            run_task_queue(args, c3_tasks)
        summarize_tune(root, args.bootstrap_iters)
        write_report(root)
    elif args.mode == "holdout":
        ensure_prompt_split(args)
        tasks = build_holdout_tasks(root)
        if tasks:
            run_task_queue(args, tasks)
        summarize_holdout(root, args.bootstrap_iters)
        write_report(root)
    elif args.mode == "summarize":
        ensure_prompt_split(args)
        if (root / "smoke").exists():
            rows = summarize_smoke(root)
            json_dump(root / "reports" / "smoke_summary.json", {"rows": rows})
        if (root / "tune").exists():
            summarize_tune(root, args.bootstrap_iters)
        if (root / "holdout").exists():
            summarize_holdout(root, args.bootstrap_iters)
        write_report(root)
    elif args.mode == "write-design":
        write_design()


if __name__ == "__main__":
    main()
