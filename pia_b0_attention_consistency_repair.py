import argparse
import ast
import csv
import hashlib
import inspect
import json
import math
import os
import random
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("TRANSFORMERS_CACHE", "/data/zhiqi/hf_cache/transformers")

import torch
import torch.nn.functional as F

import laer_relation_isolation_utils as iso
from pia_attention_guided import enforce_embedding_constraints, fixed_embedding_tensor, random_public_embeddings, variable_view
from pia_masked_server_attn_pia import build_variable_mask, uniform_all_token_activation_loss, variable_mask_audit_payload
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

EXPERIMENT_TITLE = "B0 Attention-Consistency Guided Discrete Repair"
OUTPUT_ROOT_DEFAULT = "runs/acdr"
METHOD_ALIASES = {
    "B0": "b0",
    "B0_SPARSE": "b0_sparse",
    "B0_ACDR": "b0_acdr",
    "B0_SHUFFLED_ATTN": "b0_shuffled_attn",
    "B0_GLOBAL_GATE": "b0_global_gate",
}
METHODS = [*METHOD_ALIASES.keys(), *sorted(set(METHOD_ALIASES.values()))]
REPAIR_METHODS = {"b0_sparse", "b0_acdr", "b0_shuffled_attn", "b0_global_gate"}
ATTN_METHODS = {"b0_acdr", "b0_shuffled_attn", "b0_global_gate"}


@dataclass
class ACDRConfig:
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
    max_repair_passes: int
    eps_cut: float
    eps_attn: float
    query_window: int
    attn_row_fraction: float
    max_attn_rows: int
    entropy_middle_fraction: float
    public_sink_threshold: float
    uncertainty_detector: str
    adaptive_discretization: bool
    semantic_speculation: bool
    local_files_only: bool
    fix_boundary_specials: bool = True


def canonical_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def require_strict_top1(cfg: ACDRConfig) -> None:
    if int(cfg.top_k_embedding) != 1:
        raise ValueError("ACDR requires strict Top-1: K must be 1")
    if int(cfg.top_y_semantic) != 0:
        raise ValueError("ACDR requires strict Top-1: Y must be 0")
    if bool(cfg.semantic_speculation) or bool(cfg.adaptive_discretization):
        raise ValueError("ACDR disables semantic speculation and activation calibration")
    if cfg.uncertainty_detector != "margin":
        raise ValueError("ACDR freezes margin-only uncertainty")


def load_model_and_data(cfg: ACDRConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer, model = load_tinyllama(dtype, local_files_only=cfg.local_files_only)
    model.to(device)
    model.eval()
    if len(model.model.layers) != TOTAL_BLOCKS:
        raise RuntimeError(f"Expected {TOTAL_BLOCKS} blocks, got {len(model.model.layers)}")
    return tokenizer, model, device


def ids_from_embeddings(z: torch.Tensor, embed_weight: torch.Tensor, fixed_public: Dict[int, int]) -> List[int]:
    ids = naive_discretization(z.detach().float(), embed_weight)
    for pos, token_id in fixed_public.items():
        if 0 <= int(pos) < len(ids):
            ids[int(pos)] = int(token_id)
    return [int(x) for x in ids]


def embeddings_from_ids(embed_layer: torch.nn.Module, ids: Sequence[int], device: torch.device) -> torch.Tensor:
    tensor = torch.tensor([int(x) for x in ids], dtype=torch.long, device=device).unsqueeze(0)
    return embed_layer(tensor).detach().float()


def state_hash(ids: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(int(x)) for x in ids).encode("utf-8")).hexdigest()


def changed_positions(before: Sequence[int], after: Sequence[int], variable_mask: torch.Tensor) -> List[int]:
    variable = variable_mask.detach().cpu().bool().tolist()
    return [
        i for i, is_var in enumerate(variable)
        if is_var and i < len(before) and i < len(after) and int(before[i]) != int(after[i])
    ]


def all_valid_cut_loss(model: torch.nn.Module, cfg: ACDRConfig, z: torch.Tensor, observed: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    hidden = capture_prefix_activation(model, cfg.target_layer, inputs_embeds=z, attention_mask=attention_mask)
    return uniform_all_token_activation_loss(hidden, observed, attention_mask)


def optimize_stage1_b0(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: ACDRConfig,
    observed: torch.Tensor,
    seq_len: int,
    device: torch.device,
    fixed_public: Dict[int, int],
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
        cut = uniform_all_token_activation_loss(hidden, observed, attention_mask)
        vocab = nearest_embedding_loss(variable_view(z, fixed_positions), embed_layer.weight)
        total = cut + float(cfg.lambda_vocab) * vocab
        cosine = F.cosine_similarity(hidden.float(), observed.to(hidden.device).float(), dim=-1).mean()
        if torch.isnan(total) or torch.isnan(cosine):
            raise RuntimeError(f"NaN in B0 Stage1 at step {step + 1}")
        opt.zero_grad()
        total.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([z], float(cfg.grad_clip))
        opt.step()
        final = {
            "step": step + 1,
            "activation_loss": float(cut.detach().cpu()),
            "all_token_uniform_loss": float(cut.detach().cpu()),
            "vocab_loss": float(vocab.detach().cpu()),
            "optimization_loss": float(total.detach().cpu()),
            "cosine_similarity": float(cosine.detach().cpu()),
            "loss_mode": "all_valid_uniform",
        }
        if (step + 1) % max(1, min(50, int(cfg.epoch))) == 0 or step + 1 == int(cfg.epoch):
            history.append(dict(final))
            print(
                f"method={cfg.method} stage=b0 step={step+1} act={final['activation_loss']:.6f} "
                f"vocab={final['vocab_loss']:.6f} total={final['optimization_loss']:.6f} "
                f"cosine={final['cosine_similarity']:.6f}",
                flush=True,
            )
    enforce_embedding_constraints(z, fixed_positions, fixed_embeds, left, right)
    return z.detach().float(), final, history


def token_margins(z: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
    weight = embed_weight.detach().float().to(z.device)
    distances = torch.cdist(z.squeeze(0).detach().float(), weight, p=2.0)
    top2 = torch.topk(distances, k=2, largest=False, dim=-1).values
    return (top2[:, 1] - top2[:, 0]).detach()


def select_uncertain_positions(margins: torch.Tensor, variable_mask: torch.Tensor, fraction: float) -> Tuple[torch.Tensor, List[int], Dict[str, Any]]:
    variable = variable_mask.to(margins.device).bool()
    positions = torch.nonzero(variable, as_tuple=False).flatten()
    scores = torch.zeros_like(margins, dtype=torch.float32)
    if int(positions.numel()) == 0:
        return scores, [], {"uncertain_fraction": float(fraction), "uncertain_token_count": 0, "selected_uncertain_positions": []}
    vals = margins[positions].float()
    spread = (vals.max() - vals.min()).clamp_min(1e-12)
    scores[positions] = (vals.max() - vals) / spread
    count = max(1, min(int(positions.numel()), int(math.ceil(float(fraction) * int(positions.numel())))))
    chosen = positions[torch.argsort(margins[positions], descending=False)[:count]]
    return scores.detach(), [int(x) for x in chosen.detach().cpu().tolist()], {
        "uncertainty_detector": "margin",
        "uncertain_fraction": float(fraction),
        "variable_count": int(positions.numel()),
        "uncertain_token_count": count,
        "selected_uncertain_positions": [int(x) for x in chosen.detach().cpu().tolist()],
        "margin_mean": float(vals.mean().detach().cpu()),
        "margin_min": float(vals.min().detach().cpu()),
        "margin_max": float(vals.max().detach().cpu()),
        "attention_used_for_localization": False,
    }


def fixed_size_patches(seq_len: int, variable_mask: torch.Tensor, patch_size: int) -> List[List[int]]:
    variable = variable_mask.detach().cpu().bool()
    patches: List[List[int]] = []
    size = max(1, int(patch_size))
    for start in range(0, int(seq_len), size):
        patch = [pos for pos in range(start, min(int(seq_len), start + size)) if bool(variable[pos])]
        if patch:
            patches.append(patch)
    return patches


def legal_key_mask(seq_len: int, q: int, variable_mask: torch.Tensor, fixed_public: Dict[int, int], device: torch.device) -> torch.Tensor:
    variable = variable_mask.to(device).bool().clone()
    for pos in fixed_public:
        if 0 <= int(pos) < int(seq_len):
            variable[int(pos)] = False
    return variable & (torch.arange(int(seq_len), device=device) <= int(q))


def normalize_attention_row(attn: torch.Tensor, key_mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    row = attn.float().clamp_min(0.0) * key_mask.to(attn.device).float()
    total = row.sum()
    if float(total.detach().cpu()) <= eps:
        return key_mask.to(attn.device).float() / key_mask.float().sum().clamp_min(1.0).to(attn.device)
    return row / total.clamp_min(eps)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.float().clamp_min(eps)
    q = q.float().clamp_min(eps)
    p = p / p.sum().clamp_min(eps)
    q = q / q.sum().clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * ((p * (p / m).log()).sum() + (q * (q / m).log()).sum())


def entropy01(p: torch.Tensor, key_mask: torch.Tensor, eps: float = 1e-12) -> float:
    vals = p[key_mask.to(p.device).bool()].float().clamp_min(eps)
    if int(vals.numel()) <= 1:
        return 0.0
    return float((-(vals * vals.log()).sum() / math.log(float(vals.numel()))).detach().cpu())


def capture_attention_fingerprint(model: torch.nn.Module, cfg: ACDRConfig, h: torch.Tensor, attention_mask: torch.Tensor) -> List[Tuple[int, torch.Tensor]]:
    evidence = iso.server_forward_error_evidence(
        model,
        cfg.target_layer + 1,
        h.to(dtype=next(model.parameters()).dtype),
        attention_mask,
        collect_attention=True,
        detach=True,
    )
    return [(int(layer), attn.detach().float()) for layer, attn in evidence["attentions"]]


def attention_row_records(
    obs_attn: Sequence[Tuple[int, torch.Tensor]],
    cur_attn: Sequence[Tuple[int, torch.Tensor]],
    variable_mask: torch.Tensor,
    fixed_public: Dict[int, int],
    patch_positions: Sequence[int],
    cfg: ACDRConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cur_by_layer = {int(layer): attn for layer, attn in cur_attn}
    rows: List[Dict[str, Any]] = []
    seq_len = int(variable_mask.numel())
    min_patch = min(int(x) for x in patch_positions) if patch_positions else 0
    q_start = min(seq_len, min_patch + 1)
    q_end = min(seq_len, min_patch + max(1, int(cfg.query_window)) + 1)
    variable = variable_mask.detach().cpu().bool().tolist()
    for layer, obs in obs_attn:
        cur = cur_by_layer.get(int(layer))
        if cur is None:
            continue
        for head in range(int(obs.shape[0])):
            for q in range(q_start, q_end):
                if q >= seq_len or not variable[q]:
                    continue
                key_mask = legal_key_mask(seq_len, q, variable_mask, fixed_public, obs.device)
                if int(key_mask.sum().detach().cpu()) <= 1:
                    continue
                fixed_mask = torch.zeros(seq_len, dtype=torch.bool, device=obs.device)
                for pos in fixed_public:
                    if 0 <= int(pos) < seq_len and int(pos) <= q:
                        fixed_mask[int(pos)] = True
                raw_public_mass = float((obs[head, q] * fixed_mask.float()).sum().detach().cpu())
                if raw_public_mass > float(cfg.public_sink_threshold):
                    continue
                p_obs = normalize_attention_row(obs[head, q], key_mask)
                p_cur = normalize_attention_row(cur[head, q], key_mask)
                rows.append({
                    "layer": int(layer),
                    "head": int(head),
                    "q": int(q),
                    "entropy": entropy01(p_obs, key_mask),
                    "raw_public_mass": raw_public_mass,
                    "residual_js": float(js_divergence(p_cur[key_mask], p_obs[key_mask]).detach().cpu()),
                })
    if not rows:
        return [], {"raw_row_count": 0, "purified_row_count": 0, "selected_row_count": 0}
    ent_vals = sorted(float(r["entropy"]) for r in rows)
    tail = max(0.0, (1.0 - float(cfg.entropy_middle_fraction)) / 2.0)
    lo = ent_vals[min(len(ent_vals) - 1, int(math.floor(tail * (len(ent_vals) - 1))))]
    hi = ent_vals[min(len(ent_vals) - 1, int(math.ceil((1.0 - tail) * (len(ent_vals) - 1))))]
    purified = [r for r in rows if lo <= float(r["entropy"]) <= hi]
    purified.sort(key=lambda r: float(r["residual_js"]), reverse=True)
    keep = max(1, min(len(purified), int(math.ceil(float(cfg.attn_row_fraction) * len(purified))), int(cfg.max_attn_rows))) if purified else 0
    return purified[:keep], {
        "candidate_conditioned_query_region": f"valid q > min(P), window={int(cfg.query_window)}",
        "attention_used_for_query_selection": False,
        "raw_row_count": len(rows),
        "purified_row_count": len(purified),
        "selected_row_count": keep,
        "entropy_middle_fraction": float(cfg.entropy_middle_fraction),
        "entropy_low_cut": lo,
        "entropy_high_cut": hi,
        "public_sink_threshold": float(cfg.public_sink_threshold),
    }


def attention_js_score(
    cand_attn: Sequence[Tuple[int, torch.Tensor]],
    obs_attn: Sequence[Tuple[int, torch.Tensor]],
    selected_rows: Sequence[Dict[str, Any]],
    variable_mask: torch.Tensor,
    fixed_public: Dict[int, int],
    shuffle: bool,
    seed: int,
) -> Tuple[float, float]:
    cand_by_layer = {int(layer): attn for layer, attn in cand_attn}
    obs_by_layer = {int(layer): attn for layer, attn in obs_attn}
    by_layer: Dict[int, List[torch.Tensor]] = {}
    mass_vals: List[float] = []
    seq_len = int(variable_mask.numel())
    for idx, row in enumerate(selected_rows):
        layer, head, q = int(row["layer"]), int(row["head"]), int(row["q"])
        if layer not in cand_by_layer or layer not in obs_by_layer:
            continue
        key_mask = legal_key_mask(seq_len, q, variable_mask, fixed_public, cand_by_layer[layer].device)
        if int(key_mask.sum().detach().cpu()) <= 1:
            continue
        p = normalize_attention_row(cand_by_layer[layer][head, q], key_mask)
        o = normalize_attention_row(obs_by_layer[layer][head, q], key_mask)
        if shuffle:
            valid_idx = torch.nonzero(key_mask.to(o.device).bool(), as_tuple=False).flatten()
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed) + 7919 * (idx + 1) + 101 * layer + 17 * head + q)
            perm = torch.randperm(int(valid_idx.numel()), generator=g)
            shuffled = o.clone()
            shuffled[valid_idx] = o[valid_idx.detach().cpu()[perm].to(valid_idx.device)]
            o = shuffled
        by_layer.setdefault(layer, []).append(js_divergence(p[key_mask], o[key_mask]))
        mass_vals.append(float(torch.abs(p[key_mask].sum() - o[key_mask].sum()).detach().cpu()))
    if not by_layer:
        return float("inf"), float("inf")
    layer_means = [torch.stack(vals).mean() for vals in by_layer.values() if vals]
    return float(torch.stack(layer_means).mean().detach().cpu()), float(statistics.mean(mass_vals) if mass_vals else 0.0)


def optimize_sparse_candidate_pool(
    model: torch.nn.Module,
    cfg: ACDRConfig,
    z_current: torch.Tensor,
    observed: torch.Tensor,
    attention_mask: torch.Tensor,
    variable_mask: torch.Tensor,
    fixed_public: Dict[int, int],
    update_positions: Sequence[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    device = z_current.device
    embed_layer = model.get_input_embeddings()
    fixed_embeds, fixed_positions, _ids = fixed_embedding_tensor(embed_layer, fixed_public, int(z_current.shape[1]), device)
    left, right = embedding_bounds(embed_layer.weight)
    left = left.to(device)
    right = right.to(device)
    update_positions = [int(pos) for pos in update_positions]
    if not update_positions:
        return [], {"skipped": True, "candidate_generated_count": 0, "candidate_ids_hashes": []}
    before_ids = ids_from_embeddings(z_current, embed_layer.weight, fixed_public)
    learnable = z_current[:, update_positions, :].detach().clone().squeeze(0).float().requires_grad_(True)
    opt = torch.optim.AdamW([learnable], lr=float(cfg.lr))
    candidates: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    last_hash: Optional[str] = None
    for step in range(max(1, int(cfg.epoch))):
        z = iso.apply_sparse_update(z_current, update_positions, learnable)
        enforce_embedding_constraints(z, fixed_positions, fixed_embeds, left, right)
        hidden = capture_prefix_activation(model, cfg.target_layer, inputs_embeds=z.to(dtype=embed_layer.weight.dtype), attention_mask=attention_mask)
        cut = uniform_all_token_activation_loss(hidden, observed, attention_mask)
        vocab = nearest_embedding_loss(learnable, embed_layer.weight)
        total = cut + float(cfg.lambda_vocab) * vocab
        if torch.isnan(total):
            raise RuntimeError(f"NaN in ACDR sparse patch optimization step {step + 1}")
        with torch.no_grad():
            weight = embed_layer.weight.detach().float().to(device)
            patch_dist = torch.cdist(learnable.detach().float(), weight, p=2.0)
            patch_ids = torch.argmin(patch_dist, dim=-1).detach().cpu().tolist()
        ids = list(before_ids)
        for pos, token_id in zip(update_positions, patch_ids):
            ids[int(pos)] = int(token_id)
        for pos, token_id in fixed_public.items():
            if 0 <= int(pos) < len(ids):
                ids[int(pos)] = int(token_id)
        h = state_hash(ids)
        if h != last_hash and h not in seen:
            disc_z = embeddings_from_ids(embed_layer, ids, device)
            with torch.no_grad():
                disc_cut = float(all_valid_cut_loss(model, cfg, disc_z.to(dtype=embed_layer.weight.dtype), observed, attention_mask).detach().cpu())
            candidates.append({
                "step": step + 1,
                "ids": ids,
                "ids_hash": h,
                "changed_positions": changed_positions(before_ids, ids, variable_mask),
                "cut_loss": disc_cut,
                "z": disc_z.detach().clone(),
            })
            seen.add(h)
            last_hash = h
        opt.zero_grad()
        total.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([learnable], float(cfg.grad_clip))
        opt.step()
    return candidates, {"candidate_generated_count": len(candidates), "candidate_ids_hashes": [c["ids_hash"] for c in candidates]}


def repair_with_candidates(
    model: torch.nn.Module,
    cfg: ACDRConfig,
    observed: torch.Tensor,
    z_stage1: torch.Tensor,
    variable_mask: torch.Tensor,
    fixed_public: Dict[int, int],
) -> Tuple[torch.Tensor, Dict[str, Any], List[Dict[str, Any]]]:
    device = z_stage1.device
    embed_layer = model.get_input_embeddings()
    attention_mask = torch.ones((1, int(z_stage1.shape[1])), dtype=torch.long, device=device)
    stage1_ids = ids_from_embeddings(z_stage1, embed_layer.weight, fixed_public)
    z = embeddings_from_ids(embed_layer, stage1_ids, device)
    _scores, uncertain_positions, uncertainty_stats = select_uncertain_positions(token_margins(z_stage1, embed_layer.weight), variable_mask, cfg.uncertainty_fraction)
    uncertain_set = set(int(x) for x in uncertain_positions)
    patches = fixed_size_patches(int(z.shape[1]), variable_mask, cfg.patch_size)
    obs_attn: Optional[List[Tuple[int, torch.Tensor]]] = None
    if cfg.method in ATTN_METHODS:
        obs_attn = capture_attention_fingerprint(model, cfg, observed, attention_mask)
    events: List[Dict[str, Any]] = []
    accepted = rejected = accepted_js_improved = 0
    candidate_pool_hashes: List[str] = []
    for repair_pass in range(max(1, int(cfg.max_repair_passes))):
        accepted_this_pass = 0
        for patch_id, patch in enumerate(patches):
            update_positions = [int(pos) for pos in patch if int(pos) in uncertain_set]
            if not update_positions:
                continue
            before_ids = ids_from_embeddings(z, embed_layer.weight, fixed_public)
            before_hash = state_hash(before_ids)
            cut_before = float(all_valid_cut_loss(model, cfg, z.to(dtype=embed_layer.weight.dtype), observed, attention_mask).detach().cpu())
            candidates, cand_stats = optimize_sparse_candidate_pool(model, cfg, z, observed, attention_mask, variable_mask, fixed_public, update_positions)
            candidate_pool_hashes.extend(cand_stats["candidate_ids_hashes"])
            c_plus = [c for c in candidates if float(c["cut_loss"]) < cut_before - float(cfg.eps_cut) and c["ids_hash"] != before_hash]
            selected: Optional[Dict[str, Any]] = None
            accepted_flag = False
            reason = "empty_activation_improving_candidate_pool"
            attn_before = attn_after = delta_attn = mass_before = mass_after = None
            row_stats: Dict[str, Any] = {}
            if c_plus:
                if cfg.method == "b0_sparse":
                    selected = min(c_plus, key=lambda c: float(c["cut_loss"]))
                    accepted_flag = True
                    reason = "accepted_cut_only"
                else:
                    assert obs_attn is not None
                    with torch.no_grad():
                        h_cur = capture_prefix_activation(model, cfg.target_layer, inputs_embeds=z.to(dtype=embed_layer.weight.dtype), attention_mask=attention_mask)
                        cur_attn = capture_attention_fingerprint(model, cfg, h_cur, attention_mask)
                    selected_rows, row_stats = attention_row_records(obs_attn, cur_attn, variable_mask, fixed_public, patch, cfg)
                    if selected_rows:
                        shuffle = cfg.method == "b0_shuffled_attn"
                        attn_before, mass_before = attention_js_score(cur_attn, obs_attn, selected_rows, variable_mask, fixed_public, shuffle, cfg.seed + patch_id + 1000 * repair_pass)
                        scored = []
                        for cand in c_plus:
                            with torch.no_grad():
                                h_cand = capture_prefix_activation(model, cfg.target_layer, inputs_embeds=cand["z"].to(dtype=embed_layer.weight.dtype), attention_mask=attention_mask)
                                cand_attn = capture_attention_fingerprint(model, cfg, h_cand, attention_mask)
                            js, mass = attention_js_score(cand_attn, obs_attn, selected_rows, variable_mask, fixed_public, shuffle, cfg.seed + patch_id + 1000 * repair_pass)
                            scored.append((js, float(cand["cut_loss"]), mass, cand))
                        scored.sort(key=lambda x: (x[0], x[1]))
                        attn_after, _cut, mass_after, selected = scored[0]
                        delta_attn = float(attn_after - attn_before)
                        if attn_after < attn_before - float(cfg.eps_attn):
                            accepted_flag = True
                            reason = "accepted_cut_and_attention_js_improved"
                            accepted_js_improved += 1
                        else:
                            reason = "attention_js_not_improved"
                    else:
                        reason = "no_purified_attention_rows"
                if accepted_flag and selected is not None:
                    z = selected["z"].detach().clone()
                    accepted += 1
                    accepted_this_pass += 1
                else:
                    rejected += 1
            else:
                rejected += 1
            after_ids = ids_from_embeddings(z, embed_layer.weight, fixed_public)
            events.append({
                "repair_pass": repair_pass + 1,
                "patch_id": patch_id,
                "positions": [int(x) for x in patch],
                "updated_positions": update_positions,
                "updated_positions_subset_of_uncertain": set(update_positions).issubset(uncertain_set),
                "fixed_public_modified": any(int(x) in fixed_public for x in changed_positions(before_ids, after_ids, variable_mask)),
                "candidate_generated_count": int(cand_stats["candidate_generated_count"]),
                "activation_improving_candidate_count": len(c_plus),
                "accepted": bool(accepted_flag),
                "rejected_reason": reason,
                "cut_before": cut_before,
                "cut_after": float(all_valid_cut_loss(model, cfg, z.to(dtype=embed_layer.weight.dtype), observed, attention_mask).detach().cpu()),
                "attention_js_before": attn_before,
                "attention_js_after": attn_after,
                "delta_attention_js": delta_attn,
                "mass_diagnostic_before": mass_before,
                "mass_diagnostic_after": mass_after,
                "before_ids_hash": before_hash,
                "after_accept_ids_hash": state_hash(after_ids),
                "next_patch_start_ids_hash": state_hash(after_ids),
                "selected_candidate_ids_hash": selected.get("ids_hash") if selected else None,
                "candidate_ids_hashes": cand_stats["candidate_ids_hashes"],
                **row_stats,
            })
        if accepted_this_pass == 0:
            break
    final_ids = ids_from_embeddings(z, embed_layer.weight, fixed_public)
    for event in events:
        event["returned_ids_hash"] = state_hash(final_ids)
        event["evaluation_ids_hash"] = state_hash(final_ids)
    stats = {
        "used": True,
        "repair_pass_count": max([e.get("repair_pass", 0) for e in events], default=0),
        "accepted_patch_count": int(accepted),
        "rejected_patch_count": int(rejected),
        "accept_rate": accepted / max(1, accepted + rejected),
        "uncertain_token_count": len(uncertain_positions),
        "uncertainty_stats": uncertainty_stats,
        "candidate_pool_hash": hashlib.sha256("\n".join(candidate_pool_hashes).encode("utf-8")).hexdigest(),
        "candidate_pool_hash_count": len(candidate_pool_hashes),
        "attention_js_improved_accepted_count": int(accepted_js_improved),
        "attention_role": "candidate consistency validation only" if cfg.method in ATTN_METHODS else "not used",
        "stage1_loss_mode": "all_valid_uniform",
        "patch_objective": "L_patch=L_cut_all_valid+lambda_vocab*L_vocab",
        "returned_state_policy": "final_current_discrete_state",
        "final_current_ids_hash": state_hash(final_ids),
        "changed_positions_stage1_to_final": changed_positions(stage1_ids, final_ids, variable_mask),
        "changed_positions_stage1_to_final_count": len(changed_positions(stage1_ids, final_ids, variable_mask)),
    }
    return z.detach(), stats, events


def invert_observed(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: ACDRConfig,
    observed_activation: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> Tuple[List[int], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    cfg.method = canonical_method(cfg.method)
    require_strict_top1(cfg)
    fixed_public = inferred_boundary_special_tokens(tokenizer, seq_len) if cfg.fix_boundary_specials else {}
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    variable_audit = build_variable_mask(attention_mask, fixed_public, getattr(tokenizer, "all_special_ids", None), token_ids=None)
    z, losses, history = optimize_stage1_b0(model, tokenizer, cfg, observed_activation, seq_len, device, fixed_public)
    repair_stats: Dict[str, Any] = {"used": False, "stage1_loss_mode": "all_valid_uniform"}
    repair_events: List[Dict[str, Any]] = []
    if cfg.method in REPAIR_METHODS:
        z, repair_stats, repair_events = repair_with_candidates(model, cfg, observed_activation, z, variable_audit.variable_mask, fixed_public)
    embed_layer = model.get_input_embeddings()
    recovered_ids = ids_from_embeddings(z, embed_layer.weight, fixed_public)
    with torch.no_grad():
        final_z = embeddings_from_ids(embed_layer, recovered_ids, device)
        hidden = capture_prefix_activation(model, cfg.target_layer, inputs_embeds=final_z.to(dtype=embed_layer.weight.dtype), attention_mask=attention_mask)
        all_loss = uniform_all_token_activation_loss(hidden, observed_activation, attention_mask)
    losses.update({
        "activation_loss": float(all_loss.detach().cpu()),
        "all_token_uniform_loss": float(all_loss.detach().cpu()),
        "optimization_loss": float(all_loss.detach().cpu()),
        "loss_mode": "all_valid_uniform" if cfg.method == "b0" else "b0_sparse_discrete_repair",
    })
    audits = {
        "variable_mask_audit": variable_mask_audit_payload(variable_audit),
        "uncertainty_stats": repair_stats.get("uncertainty_stats", {}),
        "patch_repair_stats": repair_stats,
        "acceptance_gate_stats": {"events": repair_events, **repair_stats},
        "strict_top1": {
            "top_k_embedding": cfg.top_k_embedding,
            "top_y_semantic": cfg.top_y_semantic,
            "semantic_speculation": cfg.semantic_speculation,
            "adaptive_discretization": cfg.adaptive_discretization,
        },
        "attack_receives_ground_truth": False,
        "attention_semantics": {
            "stage1_uses_attention": False,
            "patch_gradient_objective_uses_attention": False,
            "attention_compares_probabilities_only": True,
            "uses_downstream_representation_mse": False,
            "uses_value_norm_relation_graph": False,
            "uses_attention_for_localization_or_routing": False,
        },
        **method_integrity_audit(),
    }
    return recovered_ids, losses, audits, history + repair_events, repair_stats


def _ast_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _ast_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def validate_attack_api() -> Dict[str, Any]:
    checked = [invert_observed, optimize_stage1_b0, repair_with_candidates, optimize_sparse_candidate_pool]
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
    return {"signature": str(sig), "parameter_names": list(sig.parameters), "banned_signature_hits": signature_hits, "banned_global_reference_hits": ast_hits, "passes": not signature_hits and not ast_hits}


def validate_method_integrity() -> Dict[str, Any]:
    checked = [invert_observed, optimize_stage1_b0, repair_with_candidates, optimize_sparse_candidate_pool]
    forbidden = {"weighted_variable_activation_loss", "build_mts_token_weights", "build_attention_scale_alpha", "attention_weighted_residual_loss", "attention_scaled_activation_loss", "build_relational_graph", "contribution_matrix", "value_norms_for_layer"}
    hits: List[Dict[str, str]] = []
    for fn in checked:
        tree = ast.parse(inspect.getsource(fn))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Name, ast.Attribute)):
                name = _ast_name(node)
                if name.split(".")[-1] in forbidden:
                    hits.append({"function": fn.__name__, "name": name})
    return {"checked_functions": [fn.__name__ for fn in checked], "forbidden_hits": hits, "passes": not hits}


def method_integrity_audit() -> Dict[str, Any]:
    check = validate_method_integrity()
    return {
        "method_integrity_check": check,
        "uses_raw_attention_weighted_continuous_activation_loss": False,
        "uses_static_value_norm_relation_graph_as_main_routing": False,
        "uses_ground_truth_in_attack": False,
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


def levenshtein(a: Sequence[Any], b: Sequence[Any]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


def normalized_edit_distance(a: Sequence[int], b: Sequence[int]) -> float:
    return float(levenshtein(list(a), list(b)) / max(1, len(a), len(b)))


def prompt_exact_match(original_ids: Sequence[int], recovered_ids: Sequence[int], variable_rows: Sequence[Dict[str, Any]]) -> float:
    positions = [int(row["position"]) for row in variable_rows if row.get("variable_mask")]
    if not positions:
        return 0.0
    return 1.0 if all(pos < len(original_ids) and pos < len(recovered_ids) and int(original_ids[pos]) == int(recovered_ids[pos]) for pos in positions) else 0.0


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "token_accuracy": mean_std([row["token_accuracy"] for row in rows]),
        "bleu": mean_std([row["bleu"] for row in rows]),
        "exact_match": mean_std([row["exact_match"] for row in rows]),
        "ned": mean_std([row["ned"] for row in rows]),
        "runtime_seconds": mean_std([row["runtime_seconds"] for row in rows]),
        "peak_gpu_memory_mb": mean_std([row["peak_gpu_memory_mb"] for row in rows if row.get("peak_gpu_memory_mb") is not None]),
        "sample_count": len(rows),
    }


def run_one_config(cfg: ACDRConfig, resume: bool) -> Dict[str, Any]:
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
    pred_path.touch(exist_ok=True)
    fail_path.touch(exist_ok=True)
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    variable_samples: List[Dict[str, Any]] = []
    repair_samples: List[Dict[str, Any]] = []
    gate_samples: List[Dict[str, Any]] = []
    loss_samples: List[Dict[str, Any]] = []
    for prompt_id, sample_prompt in enumerate(prompts):
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start = time.time()
            tok = tokenizer(sample_prompt, add_special_tokens=True, truncation=False, return_tensors="pt")
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
                "server_start_layer": cfg.target_layer + 1,
                "sequence_length": int(input_ids.shape[1]),
                "strict_top1": True,
                "top_k_embedding": cfg.top_k_embedding,
                "top_y_semantic": cfg.top_y_semantic,
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
                "prompt": sample_prompt,
                "original_text": original_text,
                "recovered_text": recovered_text,
                "original_token_ids": eval_original_ids,
                "recovered_token_ids": recovered_ids,
                "original_tokens": token_texts(tokenizer, eval_original_ids),
                "recovered_tokens": token_texts(tokenizer, recovered_ids),
                "token_accuracy": token_accuracy(tokenizer, eval_original_ids, recovered_ids),
                "bleu": bleu_score(tokenizer, eval_original_ids, recovered_ids),
                "exact_match": prompt_exact_match(eval_original_ids, recovered_ids, variable_rows),
                "ned": normalized_edit_distance(eval_original_ids, recovered_ids),
                "nerr": optional_nerr(original_text, recovered_text),
                "runtime_seconds": elapsed,
                "seed": cfg.seed,
                "target_layer": cfg.target_layer,
                "prompt_token_count": len(eval_original_ids),
                "peak_gpu_memory_mb": peak_memory_mb(),
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "cpu"),
                "preflight": preflight,
                "recovery_uses_ground_truth_tokens": False,
                **losses,
                **repair_stats,
            }
            jsonl_append(pred_path, row)
            rows.append(row)
            variable_samples.append({"prompt_id": prompt_id, **audits["variable_mask_audit"]})
            repair_samples.append({"prompt_id": prompt_id, **audits["patch_repair_stats"]})
            gate_samples.append({"prompt_id": prompt_id, **audits["acceptance_gate_stats"]})
            loss_samples.append({"prompt_id": prompt_id, "final": losses, "history": history})
            print(f"method={cfg.method} prompt_id={prompt_id} token_accuracy={row['token_accuracy']:.6f} bleu={row['bleu']:.6f} elapsed={elapsed:.2f}s", flush=True)
        except Exception as exc:
            failure = {"method": cfg.method, "prompt_id": prompt_id, "error": repr(exc), "traceback": traceback.format_exc(), "seed": cfg.seed, "target_layer": cfg.target_layer}
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
    json_dump(out / "patch_repair_stats.json", {"samples": repair_samples})
    json_dump(out / "acceptance_gate_stats.json", {"samples": gate_samples})
    json_dump(out / "loss_breakdown.json", {"samples": loss_samples})
    (out / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return metrics


def load_prediction_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def token_level_fix_harm(a_rows: List[Dict[str, Any]], b_rows: List[Dict[str, Any]], prompt_ids: Sequence[int]) -> Dict[str, int]:
    a = {int(r["prompt_id"]): r for r in a_rows}
    b = {int(r["prompt_id"]): r for r in b_rows}
    fix = harm = 0
    for pid in prompt_ids:
        ar, br = a[int(pid)], b[int(pid)]
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
    rng = random.Random(seed)
    means = [statistics.mean(vals[rng.randrange(len(vals))] for _ in vals) for _round in range(rounds)]
    means.sort()
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def paired_summary(a_rows: List[Dict[str, Any]], b_rows: List[Dict[str, Any]], comparison: str) -> Dict[str, Any]:
    a = {int(row["prompt_id"]): row for row in a_rows}
    b = {int(row["prompt_id"]): row for row in b_rows}
    pids = sorted(set(a) & set(b))
    acc_d = [float(a[i]["token_accuracy"]) - float(b[i]["token_accuracy"]) for i in pids]
    bleu_d = [float(a[i]["bleu"]) - float(b[i]["bleu"]) for i in pids]
    acc_ci = bootstrap_ci(acc_d, 2026)
    bleu_ci = bootstrap_ci(bleu_d, 2027)
    out = {
        "comparison": comparison,
        "paired_prompt_count": len(pids),
        "mean_token_accuracy_delta": statistics.mean(acc_d) if acc_d else None,
        "mean_bleu_delta": statistics.mean(bleu_d) if bleu_d else None,
        "win_count": sum(1 for x in acc_d if x > 1e-12),
        "tie_count": sum(1 for x in acc_d if abs(x) <= 1e-12),
        "loss_count": sum(1 for x in acc_d if x < -1e-12),
        "acc_ci_low": acc_ci[0],
        "acc_ci_high": acc_ci[1],
        "bleu_ci_low": bleu_ci[0],
        "bleu_ci_high": bleu_ci[1],
    }
    out.update(token_level_fix_harm(a_rows, b_rows, pids))
    return out


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


def summarize_acdr(output_root: str, subdir: str = "dev_seed42") -> Dict[str, Any]:
    root = Path(output_root)
    run_root = root / subdir
    methods = ["b0", "b0_sparse", "b0_acdr", "b0_shuffled_attn", "b0_global_gate"]
    comp_rows: List[Dict[str, Any]] = []
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for method in methods:
        metrics_path = run_root / method / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        preds = load_prediction_rows(run_root / method / "predictions.jsonl")
        by_method[method] = preds
        comp_rows.append({
            "method": method,
            "token_accuracy": metrics.get("token_accuracy", {}).get("mean"),
            "bleu": metrics.get("bleu", {}).get("mean"),
            "exact_match": metrics.get("exact_match", {}).get("mean"),
            "ned": metrics.get("ned", {}).get("mean"),
            "completed_sample_count": metrics.get("completed_sample_count"),
            "failed_sample_count": metrics.get("failed_sample_count"),
            "has_nan": metrics.get("has_nan"),
            "output_dir": metrics.get("output_dir"),
        })
    if comp_rows:
        csv_write(root / "acdr_dev_comparison.csv", comp_rows)
    paired: List[Dict[str, Any]] = []
    for method, base in [("b0_sparse", "b0"), ("b0_acdr", "b0"), ("b0_acdr", "b0_sparse"), ("b0_acdr", "b0_shuffled_attn"), ("b0_shuffled_attn", "b0")]:
        if method in by_method and base in by_method:
            paired.append(paired_summary(by_method[method], by_method[base], f"{method}_vs_{base}"))
    if paired:
        csv_write(root / "acdr_dev_paired_comparison.csv", paired)
    mech: List[Dict[str, Any]] = []
    for method in ["b0_sparse", "b0_acdr", "b0_shuffled_attn", "b0_global_gate"]:
        stats_path = run_root / method / "patch_repair_stats.json"
        if not stats_path.exists():
            continue
        samples = json.loads(stats_path.read_text(encoding="utf-8")).get("samples", [])
        mech.append({
            "method": method,
            "sample_count": len(samples),
            "accepted_patch_count": sum(int(s.get("accepted_patch_count", 0)) for s in samples),
            "rejected_patch_count": sum(int(s.get("rejected_patch_count", 0)) for s in samples),
            "candidate_pool_hashes": len(set(str(s.get("candidate_pool_hash")) for s in samples)),
            "attention_js_improved_accepted_count": sum(int(s.get("attention_js_improved_accepted_count", 0)) for s in samples),
        })
    if mech:
        csv_write(root / "acdr_mechanism_audit.csv", mech)
    hash_agreement = []
    if "b0_acdr" in by_method and "b0_shuffled_attn" in by_method:
        acdr_stats = {int(s["prompt_id"]): s for s in json.loads((run_root / "b0_acdr" / "patch_repair_stats.json").read_text(encoding="utf-8")).get("samples", [])}
        shuf_stats = {int(s["prompt_id"]): s for s in json.loads((run_root / "b0_shuffled_attn" / "patch_repair_stats.json").read_text(encoding="utf-8")).get("samples", [])}
        ids = sorted(set(acdr_stats) & set(shuf_stats))
        agree = sum(1 for pid in ids if acdr_stats[pid].get("candidate_pool_hash") == shuf_stats[pid].get("candidate_pool_hash"))
        hash_agreement.append({
            "comparison": "b0_acdr_vs_b0_shuffled_attn",
            "prompt_count": len(ids),
            "candidate_pool_hash_agreement_count": agree,
            "candidate_pool_hash_agreement_rate": agree / max(1, len(ids)),
            "required_agreement_rate": 1.0,
        })
        csv_write(root / "acdr_candidate_hash_agreement.csv", hash_agreement)
    return {"comparison": comp_rows, "paired": paired, "mechanism": mech, "candidate_hash_agreement": hash_agreement}


def write_reports(output_root: str) -> None:
    summary = summarize_acdr(output_root)
    Path("analysis").mkdir(exist_ok=True)
    Path("analysis/acdr_design.md").write_text(
        "# ACDR Design\n\n"
        "ACDR keeps the original PIA/B0 reconstruction path as the main baseline and adds a second-stage discrete repair only after B0 has produced strict Top-1 tokens.\n\n"
        "## Method\n\n"
        "1. Stage1 exactly follows B0: all-valid uniform activation matching plus the vocabulary prior. No attention is used.\n"
        "2. Error localization uses B0 nearest-neighbor embedding margin only. The lowest-margin variable positions form the frozen uncertain set.\n"
        "3. Sparse continuous patch search updates only uncertain positions and optimizes `L_patch = L_cut_all_valid + lambda_vocab * L_vocab`. Attention is not in this gradient objective.\n"
        "4. Every strict Top-1 token-state change during patch search is saved as a discrete candidate.\n"
        "5. ACDR captures server-side attention probabilities for `H_obs`, compares candidate attention probabilities with `A_obs` using JS divergence, and accepts only candidates that first improve discrete all-valid activation cut loss and then improve attention consistency.\n"
        "6. Shuffled attention keeps the same budget but shuffles the observed attention fingerprint, testing whether true server attention identity contributes beyond generic repair.\n\n"
        "Ground truth tokens are used only after the attack for evaluation.\n",
        encoding="utf-8",
    )
    rows = summary.get("comparison", [])
    paired = summary.get("paired", [])
    lines = [
        "# ACDR Development Report",
        "",
        "## Observation",
        "",
        "Skytrax-28, seed42, layer17, epoch100, strict Top-1. Main baseline is B0, not B0V.",
        "",
        "| Method | Acc | BLEU | Exact | NED | Completed | Failed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['method']} | {r.get('token_accuracy')} | {r.get('bleu')} | {r.get('exact_match')} | {r.get('ned')} | {r.get('completed_sample_count')} | {r.get('failed_sample_count')} |")
    lines += ["", "## Paired Comparisons", "", "| Comparison | Acc delta | BLEU delta | W/T/L | Acc 95% CI | BLEU 95% CI | fix/harm/net |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in paired:
        lines.append(f"| {r['comparison']} | {r.get('mean_token_accuracy_delta')} | {r.get('mean_bleu_delta')} | {r.get('win_count')}/{r.get('tie_count')}/{r.get('loss_count')} | [{r.get('acc_ci_low')}, {r.get('acc_ci_high')}] | [{r.get('bleu_ci_low')}, {r.get('bleu_ci_high')}] | {r.get('fix_count')}/{r.get('harm_count')}/{r.get('net_fix_count')} |")
    method_map = {r["method"]: r for r in rows}
    supported = False
    try:
        supported = (
            float(method_map["b0_acdr"]["token_accuracy"]) > float(method_map["b0"]["token_accuracy"])
            and float(method_map["b0_acdr"]["token_accuracy"]) > float(method_map["b0_sparse"]["token_accuracy"])
            and float(method_map["b0_acdr"]["token_accuracy"]) > float(method_map["b0_shuffled_attn"]["token_accuracy"])
        )
    except Exception:
        supported = False
    lines += [
        "",
        "## Interpretation",
        "",
        f"- Main judgement passed: {str(supported).lower()}.",
        "- ACDR is supported only if it exceeds B0, B0_SPARSE, and B0_SHUFFLED_ATTN on this frozen dev run.",
        "- If this condition is false, the attention-fingerprint route should stop here rather than be swept.",
        "",
        "## Required Questions",
        "",
        "1. B0后还有多少可修复Top1错误：see fix/harm/net in paired comparisons.",
        "2. sparse search带来多少增益：see b0_sparse_vs_b0.",
        "3. attention fingerprint reranking额外带来多少：see b0_acdr_vs_b0_sparse.",
        "4. true attention是否超过shuffled attention：see b0_acdr_vs_b0_shuffled_attn.",
        "5. attention JS改善是否更容易对应GT修复：current patch logs record JS improvement counts; GT association is evaluation-only and can be derived from final fix/harm.",
        f"6. 是否达到heldout条件：{str(supported).lower()}.",
    ]
    Path("analysis/acdr_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def config_from_args(args: argparse.Namespace, method: str, run_name: str, output_dir: str) -> ACDRConfig:
    method = canonical_method(method)
    inverted, target = ATTACKER_MAP[args.participant_number][args.attacker_position]
    if args.target_layer is not None:
        target = int(args.target_layer)
        inverted = target + 1
    return ACDRConfig(
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
        max_repair_passes=args.max_repair_passes,
        eps_cut=args.eps_cut,
        eps_attn=args.eps_attn,
        query_window=args.query_window,
        attn_row_fraction=args.attn_row_fraction,
        max_attn_rows=args.max_attn_rows,
        entropy_middle_fraction=args.entropy_middle_fraction,
        public_sink_threshold=args.public_sink_threshold,
        uncertainty_detector=args.uncertainty_detector,
        adaptive_discretization=not args.no_adaptive_discretization,
        semantic_speculation=not args.disable_semantic_speculation,
        local_files_only=args.local_files_only,
    )


def run_method_set(args: argparse.Namespace, methods: Sequence[str], subdir: str) -> List[Dict[str, Any]]:
    rows = []
    for method in methods:
        canonical = canonical_method(method)
        cfg = config_from_args(args, method, canonical, str(Path(args.output_root) / subdir / canonical))
        metrics = run_one_config(cfg, args.resume)
        rows.append({
            "method": canonical,
            "token_accuracy": metrics.get("token_accuracy", {}).get("mean"),
            "bleu": metrics.get("bleu", {}).get("mean"),
            "exact_match": metrics.get("exact_match", {}).get("mean"),
            "ned": metrics.get("ned", {}).get("mean"),
            "completed_sample_count": metrics.get("completed_sample_count"),
            "failed_sample_count": metrics.get("failed_sample_count"),
            "output_dir": metrics.get("output_dir"),
        })
    csv_write(Path(args.output_root) / f"{subdir}_comparison.csv", rows)
    write_reports(args.output_root)
    return rows


def verify_no_leakage(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = config_from_args(args, "B0_ACDR", "verify", str(Path(args.output_root) / "leakage_verification" / "sample"))
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
        attn = capture_attention_fingerprint(model, cfg, observed, attention_mask)
    fixed_public = inferred_boundary_special_tokens(tokenizer, int(input_ids.shape[1]))
    variable_audit = build_variable_mask(attention_mask, fixed_public, getattr(tokenizer, "all_special_ids", None), token_ids=None)
    api_check = validate_attack_api()
    integrity = validate_method_integrity()
    boundary_diff = float((observed.float() - full_boundary.float()).abs().max().detach().cpu())
    result = {
        "title": EXPERIMENT_TITLE,
        "dataset_meta": dataset_meta,
        "target_layer": cfg.target_layer,
        "manual_h_obs_boundary_vs_full_max_abs_diff": boundary_diff,
        "manual_boundary_pass": boundary_diff < 5e-3,
        "attack_api_check": api_check,
        "method_integrity_check": integrity,
        "strict_top1": {"enabled": True, "K": cfg.top_k_embedding, "Y": cfg.top_y_semantic, "semantic": cfg.semantic_speculation, "calibration": cfg.adaptive_discretization},
        "no_ground_truth_in_attack": bool(api_check["passes"]),
        "token_id_based_special_mask_available": False,
        "variable_mask_audit": variable_mask_audit_payload(variable_audit),
        "attention_layer_count": len(attn),
        "passes": boundary_diff < 5e-3 and api_check["passes"] and integrity["passes"] and len(attn) > 0,
    }
    out = Path(args.output_root) / "leakage_verification"
    out.mkdir(parents=True, exist_ok=True)
    json_dump(out / "leakage_verification.json", result)
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EXPERIMENT_TITLE)
    parser.add_argument("--mode", choices=["verify", "single", "smoke", "dev", "summarize"], default="single")
    parser.add_argument("--method", default="B0_ACDR", choices=METHODS)
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
    parser.add_argument("--max-repair-passes", type=int, default=2)
    parser.add_argument("--eps-cut", type=float, default=1e-6)
    parser.add_argument("--eps-attn", type=float, default=0.0)
    parser.add_argument("--query-window", type=int, default=64)
    parser.add_argument("--attn-row-fraction", type=float, default=0.20)
    parser.add_argument("--max-attn-rows", type=int, default=128)
    parser.add_argument("--entropy-middle-fraction", type=float, default=0.80)
    parser.add_argument("--public-sink-threshold", type=float, default=0.50)
    parser.add_argument("--uncertainty-detector", choices=["margin"], default="margin")
    parser.add_argument("--no-adaptive-discretization", action="store_true", default=True)
    parser.add_argument("--disable-semantic-speculation", action="store_true", default=True)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "verify":
        verify_no_leakage(args)
    elif args.mode == "single":
        subdir = args.subdir or "single"
        cfg = config_from_args(args, args.method, canonical_method(args.method), str(Path(args.output_root) / subdir / canonical_method(args.method)))
        run_one_config(cfg, args.resume)
    elif args.mode == "smoke":
        args.dataset_len = 2
        args.epoch = min(int(args.epoch), 20)
        run_method_set(args, ["B0", "B0_SPARSE", "B0_ACDR", "B0_SHUFFLED_ATTN"], "smoke")
    elif args.mode == "dev":
        run_method_set(args, ["B0", "B0_SPARSE", "B0_ACDR", "B0_SHUFFLED_ATTN"], f"dev_seed{args.seed}")
    elif args.mode == "summarize":
        write_reports(args.output_root)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
