import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
import traceback
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

import pia_b0_attention_consistency_repair as acdr
from pia_attention_validation_audit import (
    binary_auc,
    changed_top1_positions,
    fixed_causal_query_rows,
    local_relation_scores,
    mean,
)
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
    text_from_ids,
    token_texts,
)


EXPERIMENT_TITLE = "B0 + Sparse + Local Attention Edge Residual Validation"
OUTPUT_ROOT_DEFAULT = "runs/b0_laer"
BASELINE_ROOT_DEFAULT = "runs/acdr/dev_seed42"
METHOD_ALIASES = {
    "B0": "b0",
    "B0_SPARSE": "b0_sparse",
    "B0_LAER": "b0_laer",
    "B0_SHUFFLED_LAER": "b0_shuffled_laer",
}
FORMAL_METHODS = ["b0", "b0_sparse", "b0_laer", "b0_shuffled_laer"]


def canonical_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method).lower()


def has_nan(value: Any) -> bool:
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    if isinstance(value, dict):
        return any(has_nan(v) for v in value.values())
    if isinstance(value, list):
        return any(has_nan(v) for v in value)
    return False


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json_samples(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    return samples if isinstance(samples, list) else []


def flush_audit_files(
    out: Path,
    variable_samples: List[Dict[str, Any]],
    repair_samples: List[Dict[str, Any]],
    gate_samples: List[Dict[str, Any]],
    loss_samples: List[Dict[str, Any]],
) -> None:
    json_dump(out / "variable_mask_audit.json", {"samples": variable_samples})
    json_dump(out / "patch_repair_stats.json", {"samples": repair_samples})
    json_dump(out / "acceptance_gate_stats.json", {"samples": gate_samples})
    json_dump(out / "loss_breakdown.json", {"samples": loss_samples})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prompt_items(args: argparse.Namespace) -> Tuple[List[Tuple[int, str]], Dict[str, Any]]:
    if not getattr(args, "split_path", None):
        prompts, dataset_meta = load_dataset_prompts(args.dataset_name, args.dataset_path, args.dataset_len, args.seed)
        return [(idx, prompt) for idx, prompt in enumerate(prompts)], dataset_meta

    split_path = Path(args.split_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split_name = str(args.split_name)
    key = f"{split_name}_ids"
    if key not in split:
        raise ValueError(f"split file {split_path} does not contain {key}")
    full_len = int(args.full_dataset_len or split.get("dataset_len") or args.dataset_len)
    prompt_seed = int(args.prompt_split_seed if args.prompt_split_seed is not None else split.get("prompt_split_seed", args.seed))
    prompts, canonical_meta = load_dataset_prompts(args.dataset_name, args.dataset_path, full_len, prompt_seed)
    ids = [int(x) for x in split[key]]
    if int(args.prompt_limit) > 0:
        ids = ids[: int(args.prompt_limit)]
    bad = [idx for idx in ids if idx < 0 or idx >= len(prompts)]
    if bad:
        raise ValueError(f"split ids out of range for canonical prompt list: {bad[:10]}")
    dataset_meta = {
        "dataset_name": args.dataset_name,
        "dataset_path": args.dataset_path,
        "canonical_dataset_meta": canonical_meta,
        "split_path": str(split_path),
        "split_sha256": sha256_file(split_path),
        "split_name": split_name,
        "prompt_split_seed": prompt_seed,
        "full_dataset_len": full_len,
        "selected_prompt_ids": ids,
        "selected_prompts": len(ids),
        "requested_prompts": len(ids),
        "note": "prompt_id stores the fixed split id; ground truth is used only after attack completion for evaluation",
    }
    return [(idx, prompts[idx]) for idx in ids], dataset_meta


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


def mean_std(values: Sequence[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"mean": None, "std": None}
    return {"mean": statistics.mean(vals), "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0}


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


def token_level_fix_harm(a_rows: List[Dict[str, Any]], b_rows: List[Dict[str, Any]], prompt_ids: Sequence[int]) -> Dict[str, int]:
    a = {int(row["prompt_id"]): row for row in a_rows}
    b = {int(row["prompt_id"]): row for row in b_rows}
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


def patch_gt_gain(before_ids: Sequence[int], candidate_ids: Sequence[int], gold_ids: Sequence[int], variable_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
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
    return {
        "gt_beneficial": fix,
        "gt_harmful": harm,
        "gt_gain": fix - harm,
        "gt_label": "positive" if fix > harm else "harmful" if harm > fix else "neutral",
    }


def build_cfg(args: argparse.Namespace, method: str, out: Path) -> acdr.ACDRConfig:
    inverted, target = ATTACKER_MAP[args.participant_number][args.attacker_position]
    if args.target_layer is not None:
        target = int(args.target_layer)
        inverted = target + 1
    return acdr.ACDRConfig(
        method=method,
        run_name="b0_laer",
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


def repair_with_laer_gate(
    model: torch.nn.Module,
    cfg: acdr.ACDRConfig,
    observed: torch.Tensor,
    z_stage1: torch.Tensor,
    variable_mask: torch.Tensor,
    fixed_public: Dict[int, int],
    shuffle: bool,
) -> Tuple[torch.Tensor, Dict[str, Any], List[Dict[str, Any]]]:
    device = z_stage1.device
    embed_layer = model.get_input_embeddings()
    attention_mask = torch.ones((1, int(z_stage1.shape[1])), dtype=torch.long, device=device)
    stage1_ids = acdr.ids_from_embeddings(z_stage1, embed_layer.weight, fixed_public)
    z = acdr.embeddings_from_ids(embed_layer, stage1_ids, device)
    _scores, uncertain_positions, uncertainty_stats = acdr.select_uncertain_positions(acdr.token_margins(z_stage1, embed_layer.weight), variable_mask, cfg.uncertainty_fraction)
    uncertain_set = set(int(x) for x in uncertain_positions)
    patches = acdr.fixed_size_patches(int(z.shape[1]), variable_mask, cfg.patch_size)
    obs_attn = acdr.capture_attention_fingerprint(model, cfg, observed, attention_mask)
    events: List[Dict[str, Any]] = []
    accepted = rejected = rejected_by_laer = rejected_empty_c = 0
    candidate_pool_hashes: List[str] = []
    for repair_pass in range(max(1, int(cfg.max_repair_passes))):
        accepted_this_pass = 0
        for patch_id, patch in enumerate(patches):
            update_positions = [int(pos) for pos in patch if int(pos) in uncertain_set]
            if not update_positions:
                continue
            before_ids = acdr.ids_from_embeddings(z, embed_layer.weight, fixed_public)
            before_hash = acdr.state_hash(before_ids)
            cut_before = float(acdr.all_valid_cut_loss(model, cfg, z.to(dtype=embed_layer.weight.dtype), observed, attention_mask).detach().cpu())
            candidates, cand_stats = acdr.optimize_sparse_candidate_pool(model, cfg, z, observed, attention_mask, variable_mask, fixed_public, update_positions)
            candidate_pool_hashes.extend(cand_stats.get("candidate_ids_hashes", []))
            c_plus = [cand for cand in candidates if float(cand["cut_loss"]) < cut_before - float(cfg.eps_cut) and cand["ids_hash"] != before_hash]
            selected = min(c_plus, key=lambda cand: float(cand["cut_loss"])) if c_plus else None
            reason = "empty_activation_improving_candidate_pool"
            accepted_flag = False
            changed_positions: List[int] = []
            relation: Dict[str, Any] = {}
            local_stats: Dict[str, Any] = {}
            if selected is not None:
                changed_positions = changed_top1_positions(before_ids, selected["ids"], variable_mask, fixed_public)
                if not changed_positions:
                    rejected_empty_c += 1
                    reason = "empty_changed_top1_set"
                else:
                    with torch.no_grad():
                        h_before = capture_prefix_activation(model, cfg.target_layer, inputs_embeds=z.to(dtype=embed_layer.weight.dtype), attention_mask=attention_mask)
                        before_attn = acdr.capture_attention_fingerprint(model, cfg, h_before, attention_mask)
                        h_candidate = capture_prefix_activation(model, cfg.target_layer, inputs_embeds=selected["z"].to(dtype=embed_layer.weight.dtype), attention_mask=attention_mask)
                        candidate_attn = acdr.capture_attention_fingerprint(model, cfg, h_candidate, attention_mask)
                    local_rows, local_stats = fixed_causal_query_rows(obs_attn, variable_mask, fixed_public, changed_positions, cfg)
                    relation = local_relation_scores(before_attn, candidate_attn, obs_attn, local_rows, changed_positions, variable_mask, fixed_public, shuffle, cfg.seed + 13 * patch_id + 1009 * repair_pass)
                    if relation.get("delta_laer") is not None and math.isfinite(float(relation["delta_laer"])) and float(relation["delta_laer"]) < 0:
                        accepted_flag = True
                        reason = "accepted_cut_and_laer_improved"
                    else:
                        rejected_by_laer += 1
                        reason = "laer_not_improved"
                if accepted_flag:
                    z = selected["z"].detach().clone()
                    accepted += 1
                    accepted_this_pass += 1
                else:
                    rejected += 1
            else:
                rejected += 1
            after_ids = acdr.ids_from_embeddings(z, embed_layer.weight, fixed_public)
            event = {
                "repair_pass": repair_pass + 1,
                "patch_id": patch_id,
                "positions": [int(x) for x in patch],
                "updated_positions": update_positions,
                "updated_positions_subset_of_uncertain": set(update_positions).issubset(uncertain_set),
                "before_ids": [int(x) for x in before_ids],
                "candidate_ids": [int(x) for x in selected["ids"]] if selected else None,
                "before_ids_hash": before_hash,
                "candidate_ids_hash": selected.get("ids_hash") if selected else None,
                "changed_positions": changed_positions,
                "cut_before": cut_before,
                "cut_candidate": float(selected["cut_loss"]) if selected else None,
                "delta_cut": float(selected["cut_loss"]) - cut_before if selected else None,
                "laer_before": relation.get("laer_before"),
                "laer_candidate": relation.get("laer_candidate"),
                "delta_laer": relation.get("delta_laer"),
                "laer_edge_count": relation.get("laer_edge_count"),
                "laer_illegal_q_le_k_count": relation.get("laer_illegal_q_le_k_count", 0),
                "laer_fixed_public_key_hit_count": relation.get("laer_fixed_public_key_hit_count", 0),
                "accepted": bool(accepted_flag),
                "rejected_reason": reason,
                "state_hash_after_gate": acdr.state_hash(after_ids),
                "next_patch_start_ids_hash": acdr.state_hash(after_ids),
                "candidate_generated_count": int(cand_stats.get("candidate_generated_count", 0)),
                "activation_improving_candidate_count": len(c_plus),
                "selected_candidate_source": "argmin_cut_over_c_plus" if selected else None,
                "attention_used_for_candidate_generation": False,
                "attention_used_for_candidate_ordering": False,
                "attention_used_for_candidate_selection": False,
                "attention_used_for_gate_only": bool(selected is not None),
                "shuffle_observed_attention": bool(shuffle),
                **local_stats,
            }
            events.append(event)
        if accepted_this_pass == 0:
            break
    final_ids = acdr.ids_from_embeddings(z, embed_layer.weight, fixed_public)
    stats = {
        "used": True,
        "repair_pass_count": max([event.get("repair_pass", 0) for event in events], default=0),
        "accepted_patch_count": int(accepted),
        "rejected_patch_count": int(rejected),
        "rejected_by_laer_count": int(rejected_by_laer),
        "rejected_empty_changed_set_count": int(rejected_empty_c),
        "accept_rate": accepted / max(1, accepted + rejected),
        "uncertain_token_count": len(uncertain_positions),
        "uncertainty_stats": uncertainty_stats,
        "candidate_pool_hash": acdr.hashlib.sha256("\n".join(candidate_pool_hashes).encode("utf-8")).hexdigest(),
        "candidate_pool_hash_count": len(candidate_pool_hashes),
        "attention_role": "local attention edge residual gate only",
        "stage1_loss_mode": "all_valid_uniform",
        "patch_objective": "L_patch=L_cut_all_valid+lambda_vocab*L_vocab",
        "gate_rule": "accept iff cut_candidate < cut_before - eps_cut and laer_candidate < laer_before",
        "shuffle_observed_attention": bool(shuffle),
        "returned_state_policy": "final_current_discrete_state",
        "final_current_ids_hash": acdr.state_hash(final_ids),
        "changed_positions_stage1_to_final": acdr.changed_positions(stage1_ids, final_ids, variable_mask),
        "changed_positions_stage1_to_final_count": len(acdr.changed_positions(stage1_ids, final_ids, variable_mask)),
    }
    return z.detach(), stats, events


def invert_observed(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: acdr.ACDRConfig,
    observed_activation: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> Tuple[List[int], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    method = canonical_method(cfg.method)
    cfg.method = method
    acdr.require_strict_top1(cfg)
    fixed_public = acdr.inferred_boundary_special_tokens(tokenizer, seq_len) if cfg.fix_boundary_specials else {}
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    variable_audit = build_variable_mask(attention_mask, fixed_public, getattr(tokenizer, "all_special_ids", None), token_ids=None)
    z, losses, history = acdr.optimize_stage1_b0(model, tokenizer, cfg, observed_activation, seq_len, device, fixed_public)
    repair_stats: Dict[str, Any] = {"used": False, "stage1_loss_mode": "all_valid_uniform"}
    repair_events: List[Dict[str, Any]] = []
    if method == "b0_sparse":
        z, repair_stats, repair_events = acdr.repair_with_candidates(model, cfg, observed_activation, z, variable_audit.variable_mask, fixed_public)
    elif method in {"b0_laer", "b0_shuffled_laer"}:
        z, repair_stats, repair_events = repair_with_laer_gate(model, cfg, observed_activation, z, variable_audit.variable_mask, fixed_public, method == "b0_shuffled_laer")
    embed_layer = model.get_input_embeddings()
    recovered_ids = acdr.ids_from_embeddings(z, embed_layer.weight, fixed_public)
    with torch.no_grad():
        final_z = acdr.embeddings_from_ids(embed_layer, recovered_ids, device)
        hidden = capture_prefix_activation(model, cfg.target_layer, inputs_embeds=final_z.to(dtype=embed_layer.weight.dtype), attention_mask=attention_mask)
        all_loss = acdr.uniform_all_token_activation_loss(hidden, observed_activation, attention_mask)
    losses.update({
        "activation_loss": float(all_loss.detach().cpu()),
        "all_token_uniform_loss": float(all_loss.detach().cpu()),
        "optimization_loss": float(all_loss.detach().cpu()),
        "loss_mode": "all_valid_uniform" if method == "b0" else "b0_sparse_laer_gate" if method in {"b0_laer", "b0_shuffled_laer"} else "b0_sparse_discrete_repair",
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
            "attention_used_for_candidate_generation": False,
            "attention_used_for_candidate_ordering": False,
            "attention_used_for_candidate_selection": False,
            "attention_used_for_gate_only": method in {"b0_laer", "b0_shuffled_laer"},
            "uses_js_gate": False,
            "uses_attention_mass_gate": False,
            "uses_downstream_representation_mse": False,
        },
    }
    return recovered_ids, losses, audits, history + repair_events, repair_stats


def run_method(args: argparse.Namespace, method: str) -> Dict[str, Any]:
    method = canonical_method(method)
    out = Path(args.output_root) / str(args.run_subdir) / method if getattr(args, "run_subdir", "") else Path(args.output_root) / f"dev_seed{args.seed}" / method
    out.mkdir(parents=True, exist_ok=True)
    if args.resume and (out / "COMPLETE").exists():
        return json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    cfg = build_cfg(args, method, out)
    acdr.require_strict_top1(cfg)
    json_dump(out / "config.json", {"title": EXPERIMENT_TITLE, **cfg.__dict__})
    tokenizer, model, device = acdr.load_model_and_data(cfg)
    prompt_items, dataset_meta = load_prompt_items(args)
    json_dump(out / "dataset_meta.json", dataset_meta)
    pred_path = out / "predictions.jsonl"
    fail_path = out / "failures.jsonl"
    for path in [pred_path, fail_path]:
        if path.exists() and not args.resume:
            path.unlink()
    pred_path.touch(exist_ok=True)
    if args.resume and fail_path.exists() and fail_path.stat().st_size > 0:
        archive = fail_path.with_name(f"failures.previous_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        shutil.copy2(fail_path, archive)
        fail_path.unlink()
    fail_path.touch(exist_ok=True)
    rows: List[Dict[str, Any]] = load_jsonl(pred_path) if args.resume else []
    completed_prompt_ids = {int(row["prompt_id"]) for row in rows}
    failures: List[Dict[str, Any]] = []
    variable_samples: List[Dict[str, Any]] = load_json_samples(out / "variable_mask_audit.json") if args.resume else []
    repair_samples: List[Dict[str, Any]] = load_json_samples(out / "patch_repair_stats.json") if args.resume else []
    gate_samples: List[Dict[str, Any]] = load_json_samples(out / "acceptance_gate_stats.json") if args.resume else []
    loss_samples: List[Dict[str, Any]] = load_json_samples(out / "loss_breakdown.json") if args.resume else []
    for prompt_id, sample_prompt in prompt_items:
        if args.resume and prompt_id in completed_prompt_ids:
            print(f"method={method} prompt_id={prompt_id} skipped_existing_prediction", flush=True)
            continue
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
            variable_rows = audits["variable_mask_audit"]["rows"]
            for event in audits["acceptance_gate_stats"].get("events", []):
                if event.get("candidate_ids") is not None:
                    event.update(patch_gt_gain(event.get("before_ids", []), event.get("candidate_ids", []), eval_original_ids, variable_rows))
                    event["ground_truth_labeling_after_attack"] = True
            row = {
                "title": EXPERIMENT_TITLE,
                "method": method,
                "prompt_id": prompt_id,
                "prompt": sample_prompt,
                "original_text": text_from_ids(tokenizer, eval_original_ids),
                "recovered_text": text_from_ids(tokenizer, recovered_ids),
                "original_token_ids": eval_original_ids,
                "recovered_token_ids": recovered_ids,
                "original_tokens": token_texts(tokenizer, eval_original_ids),
                "recovered_tokens": token_texts(tokenizer, recovered_ids),
                "token_accuracy": token_accuracy(tokenizer, eval_original_ids, recovered_ids),
                "bleu": bleu_score(tokenizer, eval_original_ids, recovered_ids),
                "exact_match": prompt_exact_match(eval_original_ids, recovered_ids, variable_rows),
                "ned": normalized_edit_distance(eval_original_ids, recovered_ids),
                "runtime_seconds": time.time() - start,
                "seed": cfg.seed,
                "target_layer": cfg.target_layer,
                "prompt_token_count": len(eval_original_ids),
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() else None,
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
            flush_audit_files(out, variable_samples, repair_samples, gate_samples, loss_samples)
            print(f"method={method} prompt_id={prompt_id} token_accuracy={row['token_accuracy']:.6f} bleu={row['bleu']:.6f} elapsed={row['runtime_seconds']:.2f}s", flush=True)
        except Exception as exc:
            failure = {"method": method, "prompt_id": prompt_id, "error": repr(exc), "traceback": traceback.format_exc(), "seed": cfg.seed, "target_layer": cfg.target_layer}
            jsonl_append(fail_path, failure)
            failures.append(failure)
            flush_audit_files(out, variable_samples, repair_samples, gate_samples, loss_samples)
            print(f"failure={json.dumps(failure, ensure_ascii=True)}", flush=True)
    metrics = {
        "title": EXPERIMENT_TITLE,
        "method": method,
        "token_accuracy": mean_std([row["token_accuracy"] for row in rows]),
        "bleu": mean_std([row["bleu"] for row in rows]),
        "exact_match": mean_std([row["exact_match"] for row in rows]),
        "ned": mean_std([row["ned"] for row in rows]),
        "runtime_seconds": mean_std([row["runtime_seconds"] for row in rows]),
        "peak_gpu_memory_mb": mean_std([row["peak_gpu_memory_mb"] for row in rows if row.get("peak_gpu_memory_mb") is not None]),
        "sample_count": len(rows),
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
        "output_dir": str(out),
    }
    json_dump(out / "metrics.json", metrics)
    flush_audit_files(out, variable_samples, repair_samples, gate_samples, loss_samples)
    (out / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return metrics


def method_dir(output_root: Path, baseline_root: Path, method: str, seed: int) -> Path:
    if method in {"b0", "b0_sparse"}:
        return baseline_root / method
    return output_root / f"dev_seed{seed}" / method


def load_method_rows(output_root: Path, baseline_root: Path, method: str, seed: int) -> List[Dict[str, Any]]:
    return load_jsonl(method_dir(output_root, baseline_root, method, seed) / "predictions.jsonl")


def load_method_metrics(output_root: Path, baseline_root: Path, method: str, seed: int) -> Dict[str, Any]:
    path = method_dir(output_root, baseline_root, method, seed) / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def summarize_patch_labels(events: List[Dict[str, Any]], pred_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    gt_by_prompt = {int(row["prompt_id"]): row for row in pred_rows}
    accepted_positive = accepted_harmful = rejected_positive = rejected_harmful = 0
    for event in events:
        prompt_row = gt_by_prompt.get(int(event["prompt_id"]))
        if not prompt_row or not event.get("candidate_ids_hash"):
            continue
        gt = patch_gt_gain(event.get("before_ids", []), event.get("candidate_ids", []), prompt_row.get("original_token_ids", []), event.get("variable_rows", []))
        event.update(gt)
        if event.get("accepted"):
            accepted_positive += int(gt["gt_beneficial"] > gt["gt_harmful"])
            accepted_harmful += int(gt["gt_harmful"] > gt["gt_beneficial"])
        else:
            rejected_positive += int(gt["gt_beneficial"] > gt["gt_harmful"])
            rejected_harmful += int(gt["gt_harmful"] > gt["gt_beneficial"])
    return {
        "accepted_beneficial_patch_count": accepted_positive,
        "accepted_harmful_patch_count": accepted_harmful,
        "rejected_beneficial_patch_count": rejected_positive,
        "rejected_harmful_patch_count": rejected_harmful,
    }


def patch_mechanism(output_root: Path, baseline_root: Path, seed: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method in FORMAL_METHODS:
        d = method_dir(output_root, baseline_root, method, seed)
        stats_path = d / "acceptance_gate_stats.json"
        if not stats_path.exists():
            continue
        samples = json.loads(stats_path.read_text(encoding="utf-8")).get("samples", [])
        events = []
        for sample in samples:
            for event in sample.get("events", []):
                events.append({"prompt_id": int(sample.get("prompt_id", -1)), **event})
        rows.append({
            "method": method,
            "sample_count": len(samples),
            "patch_event_count": len(events),
            "accepted_patch_count": sum(int(event.get("accepted", False)) for event in events),
            "rejected_patch_count": sum(1 - int(event.get("accepted", False)) for event in events),
            "rejected_by_laer_count": sum(1 for event in events if event.get("rejected_reason") == "laer_not_improved"),
            "accept_rate": mean([1.0 if event.get("accepted", False) else 0.0 for event in events]),
            "delta_laer_mean": mean([event["delta_laer"] for event in events if event.get("delta_laer") is not None]),
            "laer_edge_count_mean": mean([event["laer_edge_count"] for event in events if event.get("laer_edge_count") is not None]),
            "illegal_q_le_k_total": sum(int(event.get("laer_illegal_q_le_k_count", 0) or 0) for event in events),
            "fixed_public_key_hit_total": sum(int(event.get("laer_fixed_public_key_hit_count", 0) or 0) for event in events),
            "accepted_beneficial_patch_count": sum(1 for event in events if event.get("accepted") and event.get("gt_label") == "positive"),
            "accepted_harmful_patch_count": sum(1 for event in events if event.get("accepted") and event.get("gt_label") == "harmful"),
            "rejected_beneficial_patch_count": sum(1 for event in events if not event.get("accepted") and event.get("gt_label") == "positive"),
            "rejected_harmful_patch_count": sum(1 for event in events if not event.get("accepted") and event.get("gt_label") == "harmful"),
        })
    return rows


def pre_divergence_audit(output_root: Path, seed: int) -> Dict[str, Any]:
    root = output_root / f"dev_seed{seed}"
    true_path = root / "b0_laer" / "acceptance_gate_stats.json"
    shuf_path = root / "b0_shuffled_laer" / "acceptance_gate_stats.json"
    true_samples = json.loads(true_path.read_text(encoding="utf-8")).get("samples", []) if true_path.exists() else []
    shuf_samples = json.loads(shuf_path.read_text(encoding="utf-8")).get("samples", []) if shuf_path.exists() else []
    true_by_pid = {int(s["prompt_id"]): s.get("events", []) for s in true_samples}
    shuf_by_pid = {int(s["prompt_id"]): s.get("events", []) for s in shuf_samples}
    compared = agree = first_divergence_count = 0
    for pid in sorted(set(true_by_pid) & set(shuf_by_pid)):
        for t, s in zip(true_by_pid[pid], shuf_by_pid[pid]):
            if t.get("before_ids_hash") != s.get("before_ids_hash"):
                first_divergence_count += 1
                break
            compared += 1
            if t.get("candidate_ids_hash") == s.get("candidate_ids_hash"):
                agree += 1
            if t.get("state_hash_after_gate") != s.get("state_hash_after_gate"):
                first_divergence_count += 1
                break
    return {
        "comparison": "b0_laer_vs_b0_shuffled_laer",
        "pre_divergence_patch_count": compared,
        "pre_divergence_candidate_hash_agreement_count": agree,
        "pre_divergence_candidate_hash_agreement_rate": agree / max(1, compared),
        "prompt_divergence_count": first_divergence_count,
        "required_pre_divergence_agreement_rate": 1.0,
    }


def summarize(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = Path(args.output_root)
    baseline_root = Path(args.baseline_root)
    seed = int(args.seed)
    comparison_rows: List[Dict[str, Any]] = []
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for method in FORMAL_METHODS:
        metrics = load_method_metrics(output_root, baseline_root, method, seed)
        rows = load_method_rows(output_root, baseline_root, method, seed)
        by_method[method] = rows
        comparison_rows.append({
            "method": method,
            "token_accuracy": metrics.get("token_accuracy", {}).get("mean"),
            "bleu": metrics.get("bleu", {}).get("mean"),
            "exact_match": metrics.get("exact_match", {}).get("mean"),
            "ned": metrics.get("ned", {}).get("mean"),
            "completed_sample_count": metrics.get("completed_sample_count"),
            "failed_sample_count": metrics.get("failed_sample_count"),
            "has_nan": metrics.get("has_nan"),
            "output_dir": str(method_dir(output_root, baseline_root, method, seed)),
            "reused_from_acdr": method in {"b0", "b0_sparse"},
        })
    paired = []
    for a, b in [("b0_sparse", "b0"), ("b0_laer", "b0"), ("b0_laer", "b0_sparse"), ("b0_laer", "b0_shuffled_laer")]:
        if by_method.get(a) and by_method.get(b):
            paired.append(paired_summary(by_method[a], by_method[b], f"{a}_vs_{b}"))
    mechanism = patch_mechanism(output_root, baseline_root, seed)
    prediv = [pre_divergence_audit(output_root, seed)]
    csv_write(output_root / "b0_laer_dev_comparison.csv", comparison_rows)
    csv_write(output_root / "b0_laer_dev_paired_comparison.csv", paired)
    csv_write(output_root / "b0_laer_patch_mechanism.csv", mechanism)
    csv_write(output_root / "b0_laer_pre_divergence_audit.csv", prediv)
    write_reports(output_root, comparison_rows, paired, mechanism, prediv)
    return {"comparison": comparison_rows, "paired": paired, "mechanism": mechanism, "pre_divergence": prediv}


def write_reports(output_root: Path, comparison_rows: List[Dict[str, Any]], paired: List[Dict[str, Any]], mechanism: List[Dict[str, Any]], prediv: List[Dict[str, Any]]) -> None:
    Path("analysis").mkdir(exist_ok=True)
    Path("analysis/b0_laer_design.md").write_text(
        "# B0-LAER Design\n\n"
        "B0-LAER is a controlled end-to-end experiment built on the formal B0/B0-SPARSE path. B0 remains the main baseline.\n\n"
        "## Pipeline\n\n"
        "1. Stage1 exactly reuses B0 all-valid uniform activation matching with strict Top-1 decoding (`K=1`, `Y=0`, semantic speculation off, calibration off).\n"
        "2. B0-SPARSE generates candidates using margin uncertainty, fixed sparse patches, sparse continuous search, and strict Top-1 discrete candidates.\n"
        "3. The sparse candidate is frozen as `x_sparse = argmin L_cut_disc(candidate)` over activation-improving candidates. Attention is not used for generation, ordering, or selection.\n"
        "4. For changed Top-1 positions `C`, compute `LAER(x; C)=mean |A_x[l,h,q,k]-A_obs[l,h,q,k]|` over legal server attention edges with `k in C`, `q > k`, and fixed/public keys excluded.\n"
        "5. B0-LAER accepts a frozen sparse candidate iff `cut_candidate < cut_before - eps_cut` and `laer_candidate < laer_before`. Otherwise the patch rolls back.\n"
        "6. B0-SHUFFLED-LAER uses the same candidate generator and gate budget, but shuffles observed key identity inside each legal row before LAER scoring.\n\n"
        "Ground truth is used only after attack completion for evaluation and patch-level mechanism labels.\n",
        encoding="utf-8",
    )
    lines = [
        "# B0-LAER Development Report",
        "",
        "Skytrax-28 / seed42 / layer17 / epoch100 / strict Top-1. B0 and B0-SPARSE are reused from matching COMPLETE ACDR runs.",
        "",
        "## Method Results",
        "",
        "| Method | Acc | BLEU | Exact | NED | Completed | Failed | Reused |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(f"| {row['method']} | {row.get('token_accuracy')} | {row.get('bleu')} | {row.get('exact_match')} | {row.get('ned')} | {row.get('completed_sample_count')} | {row.get('failed_sample_count')} | {row.get('reused_from_acdr')} |")
    lines += ["", "## Paired Comparisons", "", "| Comparison | Acc delta | BLEU delta | W/T/L | Acc 95% CI | BLEU 95% CI | fix/harm/net |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in paired:
        lines.append(f"| {row['comparison']} | {row.get('mean_token_accuracy_delta')} | {row.get('mean_bleu_delta')} | {row.get('win_count')}/{row.get('tie_count')}/{row.get('loss_count')} | [{row.get('acc_ci_low')}, {row.get('acc_ci_high')}] | [{row.get('bleu_ci_low')}, {row.get('bleu_ci_high')}] | {row.get('fix_count')}/{row.get('harm_count')}/{row.get('net_fix_count')} |")
    lines += ["", "## Patch Mechanism", "", "| Method | Events | Accepted | Rejected | Rejected by LAER | Accept rate | delta LAER mean | rejected beneficial/harmful | accepted beneficial/harmful | q<=k violations | fixed key hits |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in mechanism:
        rejected_labels = f"{row.get('rejected_beneficial_patch_count')}/{row.get('rejected_harmful_patch_count')}"
        accepted_labels = f"{row.get('accepted_beneficial_patch_count')}/{row.get('accepted_harmful_patch_count')}"
        lines.append(f"| {row['method']} | {row.get('patch_event_count')} | {row.get('accepted_patch_count')} | {row.get('rejected_patch_count')} | {row.get('rejected_by_laer_count')} | {row.get('accept_rate')} | {row.get('delta_laer_mean')} | {rejected_labels} | {accepted_labels} | {row.get('illegal_q_le_k_total')} | {row.get('fixed_public_key_hit_total')} |")
    lines += ["", "## Pre-Gate Matched Audit", "", "| Comparison | Pre-div patches | Candidate hash agree | Rate | Prompt divergence count |", "|---|---:|---:|---:|---:|"]
    for row in prediv:
        lines.append(f"| {row['comparison']} | {row.get('pre_divergence_patch_count')} | {row.get('pre_divergence_candidate_hash_agreement_count')} | {row.get('pre_divergence_candidate_hash_agreement_rate')} | {row.get('prompt_divergence_count')} |")
    lookup = {r["method"]: r for r in comparison_rows}
    success = (
        lookup.get("b0_laer", {}).get("token_accuracy") is not None
        and lookup.get("b0", {}).get("token_accuracy") is not None
        and lookup.get("b0_sparse", {}).get("token_accuracy") is not None
        and lookup.get("b0_shuffled_laer", {}).get("token_accuracy") is not None
        and float(lookup["b0_laer"]["token_accuracy"]) > float(lookup["b0"]["token_accuracy"])
        and float(lookup["b0_laer"]["token_accuracy"]) > float(lookup["b0_sparse"]["token_accuracy"])
        and float(lookup["b0_laer"]["token_accuracy"]) > float(lookup["b0_shuffled_laer"]["token_accuracy"])
    )
    lines += [
        "",
        "## Decision",
        "",
        f"Heldout preparation condition met: {success}.",
        "This is a dev-only result. Do not describe it as stable improvement unless frozen heldout validation later supports it.",
    ]
    Path("analysis/b0_laer_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EXPERIMENT_TITLE)
    parser.add_argument("--mode", choices=["single", "summarize"], default="single")
    parser.add_argument("--method", default="B0_LAER")
    parser.add_argument("--output-root", default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--baseline-root", default=BASELINE_ROOT_DEFAULT)
    parser.add_argument("--run-subdir", default="")
    parser.add_argument("--dataset-name", default="Skytrax")
    parser.add_argument("--dataset-path", default="data/airline.json")
    parser.add_argument("--dataset-len", type=int, default=28)
    parser.add_argument("--full-dataset-len", type=int, default=0)
    parser.add_argument("--split-path", default="")
    parser.add_argument("--split-name", default="holdout")
    parser.add_argument("--prompt-split-seed", type=int, default=None)
    parser.add_argument("--prompt-limit", type=int, default=0)
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
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.mode == "single":
        run_method(args, args.method)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
