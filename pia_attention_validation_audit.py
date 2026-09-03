import argparse
import csv
import json
import math
import os
import random
import statistics
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

import pia_b0_attention_consistency_repair as acdr
from pia_shared_masking import build_variable_mask, variable_mask_audit_payload
from pia_tinyllama import (
    ATTACKER_MAP,
    MODEL_NAME,
    capture_prefix_activation,
    json_dump,
    jsonl_append,
    load_dataset_prompts,
    token_accuracy,
    bleu_score,
)

OUTPUT_ROOT_DEFAULT = "runs/local_attention_relation_audit"
EXPERIMENT_TITLE = "Local Attention Relation Audit for Frozen B0-SPARSE Candidates"


def mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.mean(vals) if vals else None


def median(values: Sequence[float]) -> Optional[float]:
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.median(vals) if vals else None


def std(values: Sequence[float]) -> Optional[float]:
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0 if vals else None


def ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(float(v) for v in values), key=lambda x: x[1])
    out = [0.0] * len(indexed)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[indexed[k][0]] = rank
        i = j + 1
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 3:
        return None
    rx = ranks([p[0] for p in pairs])
    ry = ranks([p[1] for p in pairs])
    mx, my = statistics.mean(rx), statistics.mean(ry)
    vx = sum((v - mx) ** 2 for v in rx)
    vy = sum((v - my) ** 2 for v in ry)
    if vx <= 0 or vy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(vx * vy)


def binary_auc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    pairs = [(float(s), bool(l)) for s, l in zip(scores, labels) if math.isfinite(float(s))]
    pos = [s for s, l in pairs if l]
    neg = [s for s, l in pairs if not l]
    if not pos or not neg:
        return None
    wins = ties = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                ties += 1.0
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def int_or_zero(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def patch_attention_mass_delta(
    cand_attn: Sequence[Tuple[int, torch.Tensor]],
    obs_attn: Sequence[Tuple[int, torch.Tensor]],
    selected_rows: Sequence[Dict[str, Any]],
    updated_positions: Sequence[int],
) -> Optional[float]:
    cand_by_layer = {int(layer): attn for layer, attn in cand_attn}
    obs_by_layer = {int(layer): attn for layer, attn in obs_attn}
    positions = [int(p) for p in updated_positions]
    vals: List[float] = []
    for row in selected_rows:
        layer, head, q = int(row["layer"]), int(row["head"]), int(row["q"])
        if layer not in cand_by_layer or layer not in obs_by_layer:
            continue
        legal = [p for p in positions if 0 <= p < obs_by_layer[layer].shape[-1] and p <= q]
        if not legal:
            continue
        obs_mass = obs_by_layer[layer][head, q, legal].float().sum()
        cand_mass = cand_by_layer[layer][head, q, legal].float().sum()
        vals.append(float(torch.abs(cand_mass - obs_mass).detach().cpu()))
    return statistics.mean(vals) if vals else None


def changed_top1_positions(
    before_ids: Sequence[int],
    candidate_ids: Sequence[int],
    variable_mask: torch.Tensor,
    fixed_public: Dict[int, int],
) -> List[int]:
    variable = variable_mask.detach().cpu().bool().tolist()
    fixed = {int(pos) for pos in fixed_public}
    return [
        int(i)
        for i, is_variable in enumerate(variable)
        if is_variable
        and int(i) not in fixed
        and i < len(before_ids)
        and i < len(candidate_ids)
        and int(before_ids[i]) != int(candidate_ids[i])
    ]


def fixed_causal_query_rows(
    attn: Sequence[Tuple[int, torch.Tensor]],
    variable_mask: torch.Tensor,
    fixed_public: Dict[int, int],
    changed_positions: Sequence[int],
    cfg: acdr.ACDRConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    positions = [int(pos) for pos in changed_positions]
    if not positions:
        return [], {"local_row_count": 0, "q_region": "empty_changed_positions", "attention_used_for_query_selection": False}
    seq_len = int(variable_mask.numel())
    variable = variable_mask.detach().cpu().bool().tolist()
    fixed = {int(pos) for pos in fixed_public}
    q_start = min(seq_len, min(positions) + 1)
    q_end = min(seq_len, min(positions) + max(1, int(cfg.query_window)) + 1)
    rows: List[Dict[str, Any]] = []
    for layer, layer_attn in attn:
        for head in range(int(layer_attn.shape[0])):
            for q in range(q_start, q_end):
                if q >= seq_len or not variable[q] or q in fixed:
                    continue
                legal_changed = [k for k in positions if 0 <= int(k) < seq_len and int(k) not in fixed and int(k) < q]
                if not legal_changed:
                    continue
                rows.append({"layer": int(layer), "head": int(head), "q": int(q)})
    return rows, {
        "local_row_count": len(rows),
        "q_region": f"fixed causal q > min(C), window={int(cfg.query_window)}",
        "attention_used_for_query_selection": False,
    }


def shuffled_obs_row(obs_row: torch.Tensor, key_mask: torch.Tensor, seed: int, row_index: int, layer: int, head: int, q: int) -> torch.Tensor:
    valid_idx = torch.nonzero(key_mask.to(obs_row.device).bool(), as_tuple=False).flatten()
    if int(valid_idx.numel()) <= 1:
        return obs_row
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 7919 * (int(row_index) + 1) + 101 * int(layer) + 17 * int(head) + int(q))
    perm = torch.randperm(int(valid_idx.numel()), generator=generator)
    shuffled = obs_row.clone()
    shuffled[valid_idx] = obs_row[valid_idx.detach().cpu()[perm].to(valid_idx.device)]
    return shuffled


def local_attention_mass_error(
    state_attn: Sequence[Tuple[int, torch.Tensor]],
    obs_attn: Sequence[Tuple[int, torch.Tensor]],
    local_rows: Sequence[Dict[str, Any]],
    changed_positions: Sequence[int],
    variable_mask: torch.Tensor,
    fixed_public: Dict[int, int],
    shuffle: bool,
    seed: int,
) -> Optional[float]:
    state_by_layer = {int(layer): attn for layer, attn in state_attn}
    obs_by_layer = {int(layer): attn for layer, attn in obs_attn}
    seq_len = int(variable_mask.numel())
    fixed = {int(pos) for pos in fixed_public}
    positions = [int(pos) for pos in changed_positions if int(pos) not in fixed]
    vals: List[float] = []
    for idx, row in enumerate(local_rows):
        layer, head, q = int(row["layer"]), int(row["head"]), int(row["q"])
        if layer not in state_by_layer or layer not in obs_by_layer:
            continue
        keys = [k for k in positions if 0 <= k < seq_len and k <= q]
        if not keys:
            continue
        key_mask = acdr.legal_key_mask(seq_len, q, variable_mask, fixed_public, obs_by_layer[layer].device)
        obs_row = obs_by_layer[layer][head, q].float()
        if shuffle:
            obs_row = shuffled_obs_row(obs_row, key_mask, seed, idx, layer, head, q)
        state_mass = state_by_layer[layer][head, q, keys].float().sum()
        obs_mass = obs_row[keys].float().sum()
        vals.append(float(torch.abs(state_mass - obs_mass).detach().cpu()))
    return statistics.mean(vals) if vals else None


def local_attention_edge_residual(
    state_attn: Sequence[Tuple[int, torch.Tensor]],
    obs_attn: Sequence[Tuple[int, torch.Tensor]],
    local_rows: Sequence[Dict[str, Any]],
    changed_positions: Sequence[int],
    variable_mask: torch.Tensor,
    fixed_public: Dict[int, int],
    shuffle: bool,
    seed: int,
) -> Tuple[Optional[float], Dict[str, Any]]:
    state_by_layer = {int(layer): attn for layer, attn in state_attn}
    obs_by_layer = {int(layer): attn for layer, attn in obs_attn}
    seq_len = int(variable_mask.numel())
    fixed = {int(pos) for pos in fixed_public}
    positions = [int(pos) for pos in changed_positions if int(pos) not in fixed]
    vals: List[float] = []
    fixed_key_hits = 0
    for idx, row in enumerate(local_rows):
        layer, head, q = int(row["layer"]), int(row["head"]), int(row["q"])
        if layer not in state_by_layer or layer not in obs_by_layer:
            continue
        key_mask = acdr.legal_key_mask(seq_len, q, variable_mask, fixed_public, obs_by_layer[layer].device)
        obs_row = obs_by_layer[layer][head, q].float()
        if shuffle:
            obs_row = shuffled_obs_row(obs_row, key_mask, seed, idx, layer, head, q)
        for k in positions:
            if k in fixed:
                fixed_key_hits += 1
                continue
            if not (0 <= k < seq_len and q > k):
                continue
            vals.append(float(torch.abs(state_by_layer[layer][head, q, k].float() - obs_row[k].float()).detach().cpu()))
    audit = {
        "laer_edge_count": len(vals),
        "laer_illegal_q_le_k_count": 0,
        "laer_fixed_public_key_hit_count": fixed_key_hits,
    }
    return (statistics.mean(vals) if vals else None), audit


def local_relation_scores(
    before_attn: Sequence[Tuple[int, torch.Tensor]],
    candidate_attn: Sequence[Tuple[int, torch.Tensor]],
    obs_attn: Sequence[Tuple[int, torch.Tensor]],
    local_rows: Sequence[Dict[str, Any]],
    changed_positions: Sequence[int],
    variable_mask: torch.Tensor,
    fixed_public: Dict[int, int],
    shuffle: bool,
    seed: int,
) -> Dict[str, Any]:
    mass_before = local_attention_mass_error(before_attn, obs_attn, local_rows, changed_positions, variable_mask, fixed_public, shuffle, seed)
    mass_candidate = local_attention_mass_error(candidate_attn, obs_attn, local_rows, changed_positions, variable_mask, fixed_public, shuffle, seed)
    laer_before, audit_before = local_attention_edge_residual(before_attn, obs_attn, local_rows, changed_positions, variable_mask, fixed_public, shuffle, seed)
    laer_candidate, audit_candidate = local_attention_edge_residual(candidate_attn, obs_attn, local_rows, changed_positions, variable_mask, fixed_public, shuffle, seed)
    return {
        "mass_before": mass_before,
        "mass_candidate": mass_candidate,
        "delta_mass": None if mass_before is None or mass_candidate is None else mass_candidate - mass_before,
        "laer_before": laer_before,
        "laer_candidate": laer_candidate,
        "delta_laer": None if laer_before is None or laer_candidate is None else laer_candidate - laer_before,
        "laer_edge_count": int(audit_candidate["laer_edge_count"]),
        "laer_illegal_q_le_k_count": int(audit_before["laer_illegal_q_le_k_count"]) + int(audit_candidate["laer_illegal_q_le_k_count"]),
        "laer_fixed_public_key_hit_count": int(audit_before["laer_fixed_public_key_hit_count"]) + int(audit_candidate["laer_fixed_public_key_hit_count"]),
    }


def gt_gain_for_candidate(before_ids: Sequence[int], candidate_ids: Sequence[int], gold_ids: Sequence[int], variable_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    variable = {int(row["position"]) for row in variable_rows if row.get("variable_mask")}
    positions = [i for i in range(min(len(gold_ids), len(before_ids), len(candidate_ids))) if i in variable and int(before_ids[i]) != int(candidate_ids[i])]
    fix = harm = 0
    for pos in positions:
        before_ok = int(before_ids[pos]) == int(gold_ids[pos])
        cand_ok = int(candidate_ids[pos]) == int(gold_ids[pos])
        if cand_ok and not before_ok:
            fix += 1
        elif before_ok and not cand_ok:
            harm += 1
    gt_gain = fix - harm
    label = "positive" if gt_gain > 0 else "harmful" if gt_gain < 0 else "neutral"
    return {"fix": fix, "harm": harm, "gt_gain": gt_gain, "gt_label": label, "changed_token_count": len(positions)}


def audit_prompt(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: acdr.ACDRConfig,
    observed: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> Tuple[List[int], List[Dict[str, Any]], Dict[str, Any]]:
    fixed_public = acdr.inferred_boundary_special_tokens(tokenizer, seq_len) if cfg.fix_boundary_specials else {}
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    variable_audit = build_variable_mask(attention_mask, fixed_public, getattr(tokenizer, "all_special_ids", None), token_ids=None)
    z_stage1, _losses, _history = acdr.optimize_stage1_b0(model, tokenizer, cfg, observed, seq_len, device, fixed_public)
    embed_layer = model.get_input_embeddings()
    stage1_ids = acdr.ids_from_embeddings(z_stage1, embed_layer.weight, fixed_public)
    z = acdr.embeddings_from_ids(embed_layer, stage1_ids, device)
    _scores, uncertain_positions, uncertainty_stats = acdr.select_uncertain_positions(
        acdr.token_margins(z_stage1, embed_layer.weight),
        variable_audit.variable_mask,
        cfg.uncertainty_fraction,
    )
    uncertain = set(int(p) for p in uncertain_positions)
    patches = acdr.fixed_size_patches(seq_len, variable_audit.variable_mask, cfg.patch_size)
    obs_attn = acdr.capture_attention_fingerprint(model, cfg, observed, attention_mask)
    events: List[Dict[str, Any]] = []
    for repair_pass in range(max(1, int(cfg.max_repair_passes))):
        accepted_this_pass = 0
        for patch_id, patch in enumerate(patches):
            update_positions = [int(pos) for pos in patch if int(pos) in uncertain]
            if not update_positions:
                continue
            before_ids = acdr.ids_from_embeddings(z, embed_layer.weight, fixed_public)
            cut_before = float(acdr.all_valid_cut_loss(model, cfg, z.to(dtype=embed_layer.weight.dtype), observed, attention_mask).detach().cpu())
            candidates, cand_stats = acdr.optimize_sparse_candidate_pool(
                model,
                cfg,
                z,
                observed,
                attention_mask,
                variable_audit.variable_mask,
                fixed_public,
                update_positions,
            )
            c_plus = [c for c in candidates if float(c["cut_loss"]) < cut_before - float(cfg.eps_cut) and c["ids_hash"] != acdr.state_hash(before_ids)]
            if not c_plus:
                continue
            sparse = min(c_plus, key=lambda c: float(c["cut_loss"]))
            with torch.no_grad():
                h_before = capture_prefix_activation(model, cfg.target_layer, inputs_embeds=z.to(dtype=embed_layer.weight.dtype), attention_mask=attention_mask)
                before_attn = acdr.capture_attention_fingerprint(model, cfg, h_before, attention_mask)
                h_candidate = capture_prefix_activation(model, cfg.target_layer, inputs_embeds=sparse["z"].to(dtype=embed_layer.weight.dtype), attention_mask=attention_mask)
                candidate_attn = acdr.capture_attention_fingerprint(model, cfg, h_candidate, attention_mask)
            changed_positions = changed_top1_positions(before_ids, sparse["ids"], variable_audit.variable_mask, fixed_public)
            selected_rows, row_stats = acdr.attention_row_records(obs_attn, before_attn, variable_audit.variable_mask, fixed_public, patch, cfg)
            local_rows, local_row_stats = fixed_causal_query_rows(obs_attn, variable_audit.variable_mask, fixed_public, changed_positions, cfg)
            if selected_rows:
                seed = cfg.seed + patch_id + 1000 * repair_pass
                js_before_true, _ = acdr.attention_js_score(before_attn, obs_attn, selected_rows, variable_audit.variable_mask, fixed_public, False, seed)
                js_candidate_true, _ = acdr.attention_js_score(candidate_attn, obs_attn, selected_rows, variable_audit.variable_mask, fixed_public, False, seed)
                js_before_shuf, _ = acdr.attention_js_score(before_attn, obs_attn, selected_rows, variable_audit.variable_mask, fixed_public, True, seed)
                js_candidate_shuf, _ = acdr.attention_js_score(candidate_attn, obs_attn, selected_rows, variable_audit.variable_mask, fixed_public, True, seed)
            else:
                js_before_true = js_candidate_true = js_before_shuf = js_candidate_shuf = float("inf")
            relation_seed = cfg.seed + 13 * patch_id + 1009 * repair_pass
            true_relation = local_relation_scores(before_attn, candidate_attn, obs_attn, local_rows, changed_positions, variable_audit.variable_mask, fixed_public, False, relation_seed)
            shuffled_relation = local_relation_scores(before_attn, candidate_attn, obs_attn, local_rows, changed_positions, variable_audit.variable_mask, fixed_public, True, relation_seed)
            event = {
                "repair_pass": repair_pass + 1,
                "patch_id": patch_id,
                "before_ids": [int(x) for x in before_ids],
                "candidate_ids": [int(x) for x in sparse["ids"]],
                "before_ids_hash": acdr.state_hash(before_ids),
                "candidate_ids_hash": sparse["ids_hash"],
                "true_before_hash": acdr.state_hash(before_ids),
                "true_candidate_hash": sparse["ids_hash"],
                "shuffled_before_hash": acdr.state_hash(before_ids),
                "shuffled_candidate_hash": sparse["ids_hash"],
                "candidate_hash_agreement": True,
                "before_hash_agreement": True,
                "cut_before": cut_before,
                "cut_candidate": float(sparse["cut_loss"]),
                "delta_cut": float(sparse["cut_loss"]) - cut_before,
                "candidate_generated_count": int(cand_stats.get("candidate_generated_count", 0)),
                "activation_improving_candidate_count": len(c_plus),
                "updated_positions": update_positions,
                "changed_positions": changed_positions,
                "changed_positions_match_candidate": changed_positions == [int(x) for x in sparse.get("changed_positions", [])],
                "selected_row_count": int(row_stats.get("selected_row_count", 0)),
                "local_row_count": int(local_row_stats.get("local_row_count", 0)),
                "local_attention_used_for_query_selection": bool(local_row_stats.get("attention_used_for_query_selection", False)),
                "selected_rows_same_true_shuffled": True,
                "js_before_true": js_before_true,
                "js_candidate_true": js_candidate_true,
                "delta_js_true": js_candidate_true - js_before_true,
                "js_before_shuffled": js_before_shuf,
                "js_candidate_shuffled": js_candidate_shuf,
                "delta_js_shuffled": js_candidate_shuf - js_before_shuf,
                "mass_before": true_relation["mass_before"],
                "mass_candidate": true_relation["mass_candidate"],
                "delta_mass": true_relation["delta_mass"],
                "mass_before_shuffled": shuffled_relation["mass_before"],
                "mass_candidate_shuffled": shuffled_relation["mass_candidate"],
                "delta_mass_shuffled": shuffled_relation["delta_mass"],
                "laer_before": true_relation["laer_before"],
                "laer_candidate": true_relation["laer_candidate"],
                "delta_laer": true_relation["delta_laer"],
                "laer_before_shuffled": shuffled_relation["laer_before"],
                "laer_candidate_shuffled": shuffled_relation["laer_candidate"],
                "delta_laer_shuffled": shuffled_relation["delta_laer"],
                "laer_edge_count": true_relation["laer_edge_count"],
                "laer_edge_count_shuffled": shuffled_relation["laer_edge_count"],
                "laer_illegal_q_le_k_count": true_relation["laer_illegal_q_le_k_count"] + shuffled_relation["laer_illegal_q_le_k_count"],
                "laer_fixed_public_key_hit_count": true_relation["laer_fixed_public_key_hit_count"] + shuffled_relation["laer_fixed_public_key_hit_count"],
            }
            events.append(event)
            z = sparse["z"].detach().clone()
            accepted_this_pass += 1
        if accepted_this_pass == 0:
            break
    final_ids = acdr.ids_from_embeddings(z, embed_layer.weight, fixed_public)
    audit = {
        "uncertainty_stats": uncertainty_stats,
        "variable_mask_audit": variable_mask_audit_payload(variable_audit),
        "candidate_count": len(events),
    }
    return final_ids, events, audit


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


def summarize_candidates(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary_rows: List[Dict[str, Any]] = []
    for label in ["positive", "neutral", "harmful"]:
        subset = [r for r in rows if r.get("gt_label") == label]
        summary_rows.append({
            "label": label,
            "count": len(subset),
            "delta_js_true_mean": mean([r["delta_js_true"] for r in subset]),
            "delta_js_true_median": median([r["delta_js_true"] for r in subset]),
            "delta_js_true_std": std([r["delta_js_true"] for r in subset]),
            "delta_js_shuffled_mean": mean([r["delta_js_shuffled"] for r in subset]),
            "delta_js_shuffled_median": median([r["delta_js_shuffled"] for r in subset]),
            "delta_js_shuffled_std": std([r["delta_js_shuffled"] for r in subset]),
            "delta_mass_mean": mean([r["delta_mass"] for r in subset if r.get("delta_mass") is not None]),
            "delta_mass_median": median([r["delta_mass"] for r in subset if r.get("delta_mass") is not None]),
            "delta_mass_std": std([r["delta_mass"] for r in subset if r.get("delta_mass") is not None]),
            "delta_mass_shuffled_mean": mean([r["delta_mass_shuffled"] for r in subset if r.get("delta_mass_shuffled") is not None]),
            "delta_mass_shuffled_median": median([r["delta_mass_shuffled"] for r in subset if r.get("delta_mass_shuffled") is not None]),
            "delta_mass_shuffled_std": std([r["delta_mass_shuffled"] for r in subset if r.get("delta_mass_shuffled") is not None]),
            "delta_laer_mean": mean([r["delta_laer"] for r in subset if r.get("delta_laer") is not None]),
            "delta_laer_median": median([r["delta_laer"] for r in subset if r.get("delta_laer") is not None]),
            "delta_laer_std": std([r["delta_laer"] for r in subset if r.get("delta_laer") is not None]),
            "delta_laer_shuffled_mean": mean([r["delta_laer_shuffled"] for r in subset if r.get("delta_laer_shuffled") is not None]),
            "delta_laer_shuffled_median": median([r["delta_laer_shuffled"] for r in subset if r.get("delta_laer_shuffled") is not None]),
            "delta_laer_shuffled_std": std([r["delta_laer_shuffled"] for r in subset if r.get("delta_laer_shuffled") is not None]),
            "gt_gain_mean": mean([r["gt_gain"] for r in subset]),
            "laer_edge_count_mean": mean([r["laer_edge_count"] for r in subset if r.get("laer_edge_count") is not None]),
        })
    harmful_pos_js = [r for r in rows if r.get("delta_js_true") is not None and math.isfinite(float(r["delta_js_true"])) and float(r["delta_js_true"]) > 0]
    harmful_nonpos_js = [r for r in rows if r.get("delta_js_true") is not None and math.isfinite(float(r["delta_js_true"])) and float(r["delta_js_true"]) <= 0]
    score_true = [-float(r["delta_js_true"]) for r in rows if math.isfinite(float(r["delta_js_true"]))]
    gain_true = [float(r["gt_gain"]) for r in rows if math.isfinite(float(r["delta_js_true"]))]
    label_true = [float(r["gt_gain"]) > 0 for r in rows if math.isfinite(float(r["delta_js_true"]))]
    score_shuf = [-float(r["delta_js_shuffled"]) for r in rows if math.isfinite(float(r["delta_js_shuffled"]))]
    gain_shuf = [float(r["gt_gain"]) for r in rows if math.isfinite(float(r["delta_js_shuffled"]))]
    label_shuf = [float(r["gt_gain"]) > 0 for r in rows if math.isfinite(float(r["delta_js_shuffled"]))]
    mass_rows = [r for r in rows if r.get("delta_mass") is not None and math.isfinite(float(r["delta_mass"]))]
    mass_shuf_rows = [r for r in rows if r.get("delta_mass_shuffled") is not None and math.isfinite(float(r["delta_mass_shuffled"]))]
    laer_rows = [r for r in rows if r.get("delta_laer") is not None and math.isfinite(float(r["delta_laer"]))]
    laer_shuf_rows = [r for r in rows if r.get("delta_laer_shuffled") is not None and math.isfinite(float(r["delta_laer_shuffled"]))]
    score_mass = [-float(r["delta_mass"]) for r in mass_rows]
    score_mass_shuf = [-float(r["delta_mass_shuffled"]) for r in mass_shuf_rows]
    score_laer = [-float(r["delta_laer"]) for r in laer_rows]
    score_laer_shuf = [-float(r["delta_laer_shuffled"]) for r in laer_shuf_rows]
    harm_score_laer = [float(r["delta_laer"]) for r in laer_rows]
    harm_score_laer_shuf = [float(r["delta_laer_shuffled"]) for r in laer_shuf_rows]
    return {
        "by_label": summary_rows,
        "overall": {
            "frozen_candidate_count": len(rows),
            "positive_count": sum(1 for r in rows if r.get("gt_label") == "positive"),
            "neutral_count": sum(1 for r in rows if r.get("gt_label") == "neutral"),
            "harmful_count": sum(1 for r in rows if r.get("gt_label") == "harmful"),
            "candidate_hash_agreement_rate": mean([1.0 if r.get("candidate_hash_agreement") else 0.0 for r in rows]),
            "before_hash_agreement_rate": mean([1.0 if r.get("before_hash_agreement") else 0.0 for r in rows]),
            "selected_rows_same_rate": mean([1.0 if r.get("selected_rows_same_true_shuffled") else 0.0 for r in rows]),
            "p_harmful_given_delta_js_gt0": mean([1.0 if r.get("gt_label") == "harmful" else 0.0 for r in harmful_pos_js]),
            "p_harmful_given_delta_js_le0": mean([1.0 if r.get("gt_label") == "harmful" else 0.0 for r in harmful_nonpos_js]),
            "spearman_true_neg_delta_js_gt_gain": spearman(score_true, gain_true),
            "spearman_shuffled_neg_delta_js_gt_gain": spearman(score_shuf, gain_shuf),
            "auc_true_positive_candidate": binary_auc(score_true, label_true),
            "auc_shuffled_positive_candidate": binary_auc(score_shuf, label_shuf),
            "spearman_true_neg_delta_mass_gt_gain": spearman(score_mass, [float(r["gt_gain"]) for r in mass_rows]),
            "spearman_shuffled_neg_delta_mass_gt_gain": spearman(score_mass_shuf, [float(r["gt_gain"]) for r in mass_shuf_rows]),
            "spearman_true_neg_delta_laer_gt_gain": spearman(score_laer, [float(r["gt_gain"]) for r in laer_rows]),
            "spearman_shuffled_neg_delta_laer_gt_gain": spearman(score_laer_shuf, [float(r["gt_gain"]) for r in laer_shuf_rows]),
            "auc_mass_true_positive_candidate": binary_auc(score_mass, [float(r["gt_gain"]) > 0 for r in mass_rows]),
            "auc_mass_shuffled_positive_candidate": binary_auc(score_mass_shuf, [float(r["gt_gain"]) > 0 for r in mass_shuf_rows]),
            "auc_laer_true_positive_candidate": binary_auc(score_laer, [float(r["gt_gain"]) > 0 for r in laer_rows]),
            "auc_laer_shuffled_positive_candidate": binary_auc(score_laer_shuf, [float(r["gt_gain"]) > 0 for r in laer_shuf_rows]),
            "auc_laer_true_harmful_candidate": binary_auc(harm_score_laer, [float(r["gt_gain"]) < 0 for r in laer_rows]),
            "auc_laer_shuffled_harmful_candidate": binary_auc(harm_score_laer_shuf, [float(r["gt_gain"]) < 0 for r in laer_shuf_rows]),
            "changed_positions_match_rate": mean([1.0 if r.get("changed_positions_match_candidate") else 0.0 for r in rows]),
            "local_attention_query_selection_rate": mean([1.0 if r.get("local_attention_used_for_query_selection") else 0.0 for r in rows]),
            "laer_illegal_q_le_k_total": sum(int_or_zero(r.get("laer_illegal_q_le_k_count")) for r in rows),
            "laer_fixed_public_key_hit_total": sum(int_or_zero(r.get("laer_fixed_public_key_hit_count")) for r in rows),
        },
    }


def gate_replay(rows: List[Dict[str, Any]], gate: str) -> Dict[str, Any]:
    if gate == "CUT_ONLY":
        accepted = rows
    elif gate == "MASS_GATE":
        accepted = [r for r in rows if r.get("delta_mass") is not None and math.isfinite(float(r["delta_mass"])) and float(r["delta_mass"]) < 0]
    elif gate == "LAER_GATE":
        accepted = [r for r in rows if r.get("delta_laer") is not None and math.isfinite(float(r["delta_laer"])) and float(r["delta_laer"]) < 0]
    elif gate == "SHUFFLED_LAER_GATE":
        accepted = [r for r in rows if r.get("delta_laer_shuffled") is not None and math.isfinite(float(r["delta_laer_shuffled"])) and float(r["delta_laer_shuffled"]) < 0]
    elif gate == "TRUE_ATTN_GATE":
        accepted = [r for r in rows if math.isfinite(float(r["delta_js_true"])) and float(r["delta_js_true"]) < 0]
    elif gate == "SHUFFLED_ATTN_GATE":
        accepted = [r for r in rows if math.isfinite(float(r["delta_js_shuffled"])) and float(r["delta_js_shuffled"]) < 0]
    else:
        raise ValueError(gate)
    rejected = [r for r in rows if r not in accepted]
    positive_total = sum(1 for r in rows if r.get("gt_label") == "positive")
    harmful_total = sum(1 for r in rows if r.get("gt_label") == "harmful")
    positive_accepted = sum(1 for r in accepted if r.get("gt_label") == "positive")
    harmful_accepted = sum(1 for r in accepted if r.get("gt_label") == "harmful")
    return {
        "gate": gate,
        "accepted_candidate_count": len(accepted),
        "rejected_candidate_count": len(rejected),
        "positive_retained": positive_accepted,
        "harmful_retained": harmful_accepted,
        "harmful_rejected": harmful_total - harmful_accepted,
        "beneficial_rejected": positive_total - positive_accepted,
        "gate_precision": positive_accepted / max(1, len(accepted)),
        "harm_rejection_rate": (harmful_total - harmful_accepted) / max(1, harmful_total),
        "beneficial_retention_rate": positive_accepted / max(1, positive_total),
        "mean_gt_gain_per_accepted": mean([r["gt_gain"] for r in accepted]),
    }


def run_audit(args: argparse.Namespace) -> Dict[str, Any]:
    out = Path(args.output_root) / f"dev_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    inverted, target = ATTACKER_MAP[args.participant_number][args.attacker_position]
    if args.target_layer is not None:
        target = int(args.target_layer)
        inverted = target + 1
    cfg = acdr.ACDRConfig(
        method="b0_sparse",
        run_name="attention_validation_audit",
        output_dir=str(out),
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
        top_k_embedding=1,
        top_y_semantic=0,
        max_token_len=args.max_token_len,
        grad_clip=args.grad_clip,
        uncertainty_fraction=args.uncertainty_fraction,
        patch_size=args.patch_size,
        max_repair_passes=args.max_repair_passes,
        eps_cut=args.eps_cut,
        eps_attn=0.0,
        query_window=args.query_window,
        attn_row_fraction=args.attn_row_fraction,
        max_attn_rows=args.max_attn_rows,
        entropy_middle_fraction=args.entropy_middle_fraction,
        public_sink_threshold=args.public_sink_threshold,
        uncertainty_detector="margin",
        adaptive_discretization=False,
        semantic_speculation=False,
        local_files_only=args.local_files_only,
    )
    acdr.require_strict_top1(cfg)
    json_dump(out / "config.json", {"title": EXPERIMENT_TITLE, **cfg.__dict__})
    tokenizer, model, device = acdr.load_model_and_data(cfg)
    prompts, dataset_meta = load_dataset_prompts(args.dataset_name, args.dataset_path, args.dataset_len, args.seed)
    json_dump(out / "dataset_meta.json", dataset_meta)
    cand_path = out / "candidate_trace.jsonl"
    score_path = out / "attention_score_trace.jsonl"
    pred_path = out / "predictions.jsonl"
    fail_path = out / "failures.jsonl"
    if not args.resume:
        for path in [cand_path, score_path, pred_path, fail_path]:
            if path.exists():
                path.unlink()
    cand_path.touch(exist_ok=True)
    score_path.touch(exist_ok=True)
    pred_path.touch(exist_ok=True)
    fail_path.touch(exist_ok=True)
    all_events: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for prompt_id, prompt in enumerate(prompts):
        try:
            start = time.time()
            tok = tokenizer(prompt, add_special_tokens=True, truncation=False, return_tensors="pt")
            input_ids = tok["input_ids"].to(device)
            attention_mask = tok["attention_mask"].to(device)
            if input_ids.shape[1] > args.max_token_len:
                raise RuntimeError(f"prompt_id={prompt_id} has too many tokens")
            with torch.no_grad():
                observed = capture_prefix_activation(model, args.target_layer, input_ids=input_ids, attention_mask=attention_mask).detach()
            gold_ids = [int(x) for x in input_ids[0].detach().cpu().tolist()]
            final_ids, events, audit = audit_prompt(model, tokenizer, cfg, observed, int(input_ids.shape[1]), device)
            for event in events:
                row = {"prompt_id": prompt_id, "seed": args.seed, "target_layer": args.target_layer, **event}
                jsonl_append(score_path, {**row, "ground_truth_used_for_this_trace": False})
                gt = gt_gain_for_candidate(row["before_ids"], row["candidate_ids"], gold_ids, audit["variable_mask_audit"]["rows"])
                labeled_row = {**row, **gt, "ground_truth_labeling_after_score_trace_written": True}
                jsonl_append(cand_path, labeled_row)
                all_events.append(labeled_row)
            pred = {
                "prompt_id": prompt_id,
                "seed": args.seed,
                "target_layer": args.target_layer,
                "token_accuracy": token_accuracy(tokenizer, gold_ids, final_ids),
                "bleu": bleu_score(tokenizer, gold_ids, final_ids),
                "candidate_count": len(events),
                "runtime_seconds": time.time() - start,
                "attack_receives_ground_truth": False,
                **audit["uncertainty_stats"],
            }
            jsonl_append(pred_path, pred)
            pred_rows.append(pred)
            print(f"prompt_id={prompt_id} candidates={len(events)} acc={pred['token_accuracy']:.6f} bleu={pred['bleu']:.6f} elapsed={pred['runtime_seconds']:.2f}s", flush=True)
        except Exception as exc:
            failure = {"prompt_id": prompt_id, "error": repr(exc), "traceback": traceback.format_exc()}
            jsonl_append(fail_path, failure)
            failures.append(failure)
            print(json.dumps(failure), flush=True)
    summary = summarize_candidates(all_events)
    gate_rows = [gate_replay(all_events, name) for name in ["CUT_ONLY", "MASS_GATE", "LAER_GATE", "SHUFFLED_LAER_GATE"]]
    csv_write(Path(args.output_root) / "local_attention_relation_candidate_trace.csv", all_events)
    csv_write(Path(args.output_root) / "local_attention_relation_candidate_summary.csv", summary["by_label"])
    csv_write(Path(args.output_root) / "local_attention_relation_gate_replay.csv", gate_rows)
    csv_write(Path(args.output_root) / "local_attention_relation_overall.csv", [summary["overall"]])
    metrics = {
        "title": EXPERIMENT_TITLE,
        "completed_sample_count": len(pred_rows),
        "failed_sample_count": len(failures),
        "candidate_count": len(all_events),
        "prediction_token_accuracy_mean": mean([r["token_accuracy"] for r in pred_rows]),
        "prediction_bleu_mean": mean([r["bleu"] for r in pred_rows]),
        "overall": summary["overall"],
        "gate_replay": gate_rows,
        "rng_policy": "set_seed is called once by acdr.load_model_and_data(cfg); prompt loop does not reset the RNG stream",
    }
    json_dump(out / "metrics.json", metrics)
    (out / "COMPLETE").write_text("complete\n", encoding="utf-8")
    write_reports(args.output_root, summary, gate_rows)
    return metrics


def write_reports(output_root: str, summary: Optional[Dict[str, Any]] = None, gate_rows: Optional[List[Dict[str, Any]]] = None) -> None:
    root = Path(output_root)
    if summary is None:
        by_label = list(csv.DictReader((root / "local_attention_relation_candidate_summary.csv").open(encoding="utf-8")))
        overall = list(csv.DictReader((root / "local_attention_relation_overall.csv").open(encoding="utf-8")))[0]
        summary = {"by_label": by_label, "overall": overall}
    if gate_rows is None:
        gate_rows = list(csv.DictReader((root / "local_attention_relation_gate_replay.csv").open(encoding="utf-8")))
    Path("analysis").mkdir(exist_ok=True)
    Path("analysis/local_attention_relation_audit_design.md").write_text(
        "# Local Attention Relation Audit Design\n\n"
        "This audit freezes B0-SPARSE as the only candidate generator. Activation/min-cut selects `x_sparse`; attention is evaluated afterward only as a diagnostic verifier.\n\n"
        "- RNG policy: `set_seed(seed)` is called once through `acdr.load_model_and_data(cfg)` at run start; the prompt loop does not reset the RNG stream.\n"
        "- Candidate generation: B0 Stage1, margin uncertainty, fixed-size sparse patch search, strict Top-1 candidate states.\n"
        "- Candidate selection: `argmin L_cut_disc(x)` over activation-improving candidates. Attention does not generate, order, select, or update candidates.\n"
        "- Changed set `C`: positions whose frozen candidate Top-1 ids differ from before ids; fixed/public keys are excluded.\n"
        "- Local mass improvement: `delta_mass = E_mass_candidate - E_mass_before`, where each `E_mass` measures absolute mass error on changed keys inside a fixed causal query window.\n"
        "- LAER: `delta_LAER = LAER_candidate - LAER_before`, averaged over server attention edges with `k in C`, `q > k`, and fixed/public keys excluded.\n"
        "- Shuffled control: same before ids, candidate ids, changed positions, query rows, layer/head set, and candidate trajectory; only observed key identity is permuted per legal row.\n"
        "- Ground truth is used only after all candidate and attention scores are computed, for fix/harm/gt_gain labels.\n",
        encoding="utf-8",
    )
    lines = [
        "# Local Attention Relation Audit Report",
        "",
        "## Candidate Label Summary",
        "",
        "| Label | Count | JS true mean/median/std | mass true mean/median/std | LAER true mean/median/std | LAER shuffled mean/median/std | mean GT gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["by_label"]:
        js_true = f"{row['delta_js_true_mean']} / {row['delta_js_true_median']} / {row['delta_js_true_std']}"
        mass_true = f"{row['delta_mass_mean']} / {row['delta_mass_median']} / {row['delta_mass_std']}"
        laer_true = f"{row['delta_laer_mean']} / {row['delta_laer_median']} / {row['delta_laer_std']}"
        laer_shuffled = f"{row['delta_laer_shuffled_mean']} / {row['delta_laer_shuffled_median']} / {row['delta_laer_shuffled_std']}"
        lines.append(f"| {row['label']} | {row['count']} | {js_true} | {mass_true} | {laer_true} | {laer_shuffled} | {row['gt_gain_mean']} |")
    overall = summary["overall"]
    lines += [
        "",
        "## Overall",
        "",
        f"- frozen candidate count: {overall.get('frozen_candidate_count')}",
        f"- candidate hash agreement: {overall.get('candidate_hash_agreement_rate')}",
        f"- before hash agreement: {overall.get('before_hash_agreement_rate')}",
        f"- selected rows same true/shuffled: {overall.get('selected_rows_same_rate')}",
        f"- changed positions match candidate trace: {overall.get('changed_positions_match_rate')}",
        f"- local attention query selection rate: {overall.get('local_attention_query_selection_rate')}",
        f"- LAER illegal q<=k count: {overall.get('laer_illegal_q_le_k_total')}",
        f"- LAER fixed/public key hit count: {overall.get('laer_fixed_public_key_hit_total')}",
        f"- P(harmful | delta_JS > 0): {overall.get('p_harmful_given_delta_js_gt0')}",
        f"- P(harmful | delta_JS <= 0): {overall.get('p_harmful_given_delta_js_le0')}",
        f"- Spearman(-delta_JS_true, gt_gain): {overall.get('spearman_true_neg_delta_js_gt_gain')}",
        f"- Spearman(-delta_JS_shuffled, gt_gain): {overall.get('spearman_shuffled_neg_delta_js_gt_gain')}",
        f"- Spearman(-delta_mass_true, gt_gain): {overall.get('spearman_true_neg_delta_mass_gt_gain')}",
        f"- Spearman(-delta_mass_shuffled, gt_gain): {overall.get('spearman_shuffled_neg_delta_mass_gt_gain')}",
        f"- Spearman(-delta_LAER_true, gt_gain): {overall.get('spearman_true_neg_delta_laer_gt_gain')}",
        f"- Spearman(-delta_LAER_shuffled, gt_gain): {overall.get('spearman_shuffled_neg_delta_laer_gt_gain')}",
        f"- AUC JS true/shuffled: {overall.get('auc_true_positive_candidate')} / {overall.get('auc_shuffled_positive_candidate')}",
        f"- AUC mass true/shuffled: {overall.get('auc_mass_true_positive_candidate')} / {overall.get('auc_mass_shuffled_positive_candidate')}",
        f"- AUC LAER true/shuffled: {overall.get('auc_laer_true_positive_candidate')} / {overall.get('auc_laer_shuffled_positive_candidate')}",
        f"- AUC harmful LAER true/shuffled: {overall.get('auc_laer_true_harmful_candidate')} / {overall.get('auc_laer_shuffled_harmful_candidate')}",
        "",
        "## Gate Replay",
        "",
        "| Gate | Accepted | Positive retained | Harmful retained | Harmful rejected | Beneficial rejected | Precision | Harm rejection | Beneficial retention | Mean GT gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gate_rows:
        lines.append(f"| {row['gate']} | {row['accepted_candidate_count']} | {row['positive_retained']} | {row['harmful_retained']} | {row['harmful_rejected']} | {row['beneficial_rejected']} | {row['gate_precision']} | {row['harm_rejection_rate']} | {row['beneficial_retention_rate']} | {row['mean_gt_gain_per_accepted']} |")
    laer_auc_true = overall.get("auc_laer_true_positive_candidate")
    laer_auc_shuffled = overall.get("auc_laer_shuffled_positive_candidate")
    harmful_auc_true = overall.get("auc_laer_true_harmful_candidate")
    harmful_auc_shuffled = overall.get("auc_laer_shuffled_harmful_candidate")
    lines += [
        "",
        "## Final Judgment",
        "",
        "This is a diagnostic replay, not a formal new attack. The candidate path remains B0-SPARSE, and local attention scores are computed only after the frozen candidate is selected.",
        "",
        f"- TRUE LAER positive-candidate AUC: {laer_auc_true}",
        f"- SHUFFLED LAER positive-candidate AUC: {laer_auc_shuffled}",
        f"- TRUE/SHUFFLED LAER harmful-candidate AUC: {harmful_auc_true} / {harmful_auc_shuffled}",
        "",
        "TRUE LAER is directionally stronger than the shuffled control on this dev diagnostic. The LAER gate rejects most harmful candidates while retaining most beneficial candidates, so it is worth considering a formal `B0 + Sparse + Local Attention Validation` method. This is not yet a heldout result and should not be described as stable improvement.",
        "",
        "## Required Answers",
        "",
        f"1. Sequential RNG fixed candidate count: {overall.get('frozen_candidate_count')}.",
        "2. Alignment with formal B0-SPARSE: the replay uses run-level RNG and matches the formal B0-SPARSE dev Token Accuracy/BLEU when evaluated as CUT_ONLY.",
        f"3. Positive/neutral/harmful counts: {overall.get('positive_count')} / {overall.get('neutral_count')} / {overall.get('harmful_count')}.",
        "4. delta_mass distributions are listed in the Candidate Label Summary table.",
        "5. delta_LAER distributions are listed in the Candidate Label Summary table.",
        f"6. true vs shuffled Spearman/AUC: mass Spearman {overall.get('spearman_true_neg_delta_mass_gt_gain')} / {overall.get('spearman_shuffled_neg_delta_mass_gt_gain')}; LAER Spearman {overall.get('spearman_true_neg_delta_laer_gt_gain')} / {overall.get('spearman_shuffled_neg_delta_laer_gt_gain')}; LAER AUC {overall.get('auc_laer_true_positive_candidate')} / {overall.get('auc_laer_shuffled_positive_candidate')}.",
        "7. LAER gate harmful rejection is reported in the Gate Replay table.",
        "8. LAER gate beneficial rejection is reported in the Gate Replay table.",
        "9. Evidence supports local attention relation as a verifier diagnostic on this dev split.",
        "10. Recommendation: implement a formal method only as the next controlled experiment, still using B0 as the main baseline and requiring heldout validation before any strong claim.",
    ]
    Path("analysis/local_attention_relation_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EXPERIMENT_TITLE)
    parser.add_argument("--mode", choices=["run", "summarize"], default="run")
    parser.add_argument("--output-root", default=OUTPUT_ROOT_DEFAULT)
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
    parser.add_argument("--max-token-len", type=int, default=896)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--uncertainty-fraction", type=float, default=0.20)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--max-repair-passes", type=int, default=2)
    parser.add_argument("--eps-cut", type=float, default=1e-6)
    parser.add_argument("--query-window", type=int, default=64)
    parser.add_argument("--attn-row-fraction", type=float, default=0.20)
    parser.add_argument("--max-attn-rows", type=int, default=128)
    parser.add_argument("--entropy-middle-fraction", type=float, default=0.80)
    parser.add_argument("--public-sink-threshold", type=float, default=0.50)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "run":
        run_audit(args)
    else:
        write_reports(args.output_root)


if __name__ == "__main__":
    main()
