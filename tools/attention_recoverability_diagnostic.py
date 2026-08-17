#!/usr/bin/env python3
"""Attention-recoverability diagnostic for completed strict Top-1 B0V runs.

The script is evaluation-only.  It reads completed predictions, uses ground
truth only to compute labels/metrics, and never calls an attack optimizer.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pia_masked_server_attn_pia import (
    SAWConfig,
    build_variable_mask,
    capture_prefix_activation,
    inferred_boundary_special_tokens,
    load_model_and_data,
    server_attention_bundle,
)


DEFAULT_RESULT_DIRS = [
    "runs/attention_weighted_pia/stage1/B0V_layer11_top1_seed42_epoch100",
    "runs/attention_weighted_pia/stage1/B0V_layer17_top1_seed42_epoch100",
    "runs/attention_weighted_pia/stage1/B0V_layer19_top1_seed42_epoch100",
]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom_x = math.sqrt(sum(x * x for x in dx))
    denom_y = math.sqrt(sum(y * y for y in dy))
    if denom_x <= 0.0 or denom_y <= 0.0:
        return float("nan")
    return sum(x * y for x, y in zip(dx, dy)) / (denom_x * denom_y)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan")
    return pearson(average_ranks(xs), average_ranks(ys))


def mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def assign_quartiles(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (float(row["attention_importance"]), int(row["position"])))
    n = len(ordered)
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(ordered):
        q_idx = min(3, int(idx * 4 / max(1, n)))
        copied = dict(row)
        copied["attention_quartile"] = f"Q{q_idx + 1}"
        out.append(copied)
    return out


def normalize_mean_one(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    total = sum(values)
    if abs(total) <= 1e-12:
        return [1.0 for _ in values]
    scale = len(values) / total
    return [float(v) * scale for v in values]


def invert_mean_one_weights(weights: Sequence[float]) -> List[float]:
    if not weights:
        return []
    low = min(weights)
    high = max(weights)
    inverted = [high + low - float(w) for w in weights]
    return normalize_mean_one(inverted)


def counterfactual_objectives(
    residuals: Sequence[float],
    attention_weights: Sequence[float],
    rng: random.Random,
) -> Dict[str, Any]:
    if len(residuals) != len(attention_weights):
        raise ValueError("residuals and attention_weights must have the same length")
    if not residuals:
        return {
            "uniform_objective": float("nan"),
            "original_attention_objective": float("nan"),
            "shuffled_attention_objective": float("nan"),
            "inverted_attention_objective": float("nan"),
            "shuffled_weights": [],
            "inverted_weights": [],
        }
    shuffled = list(attention_weights)
    rng.shuffle(shuffled)
    inverted = invert_mean_one_weights(attention_weights)
    n = len(residuals)
    return {
        "uniform_objective": sum(float(r) for r in residuals) / n,
        "original_attention_objective": sum(float(w) * float(r) for w, r in zip(attention_weights, residuals)) / n,
        "shuffled_attention_objective": sum(float(w) * float(r) for w, r in zip(shuffled, residuals)) / n,
        "inverted_attention_objective": sum(float(w) * float(r) for w, r in zip(inverted, residuals)) / n,
        "shuffled_weights": shuffled,
        "inverted_weights": inverted,
    }


def config_from_json(config: Dict[str, Any]) -> SAWConfig:
    fields = {field.name for field in dataclasses.fields(SAWConfig)}
    defaults = {
        "stage_a_epoch": int(config.get("epoch", 100)),
        "lambda_dummy": 0.1,
        "lambda_context": 0.0,
        "gamma": 0.3,
        "weight_source": "mean_query",
        "weight_floor": 0.0,
        "weight_power": float(config.get("weight_power", 0.5)),
        "weight_min": float(config.get("weight_min", 0.5)),
        "weight_max": float(config.get("weight_max", 1.5)),
        "alpha_min": 0.5,
        "alpha_max": 1.5,
        "attention_start_ratio": 0.5,
        "attention_full_ratio": 0.7,
        "uncertainty_fraction": 0.0,
        "adaptive_min_beta": 0.0,
        "refine_epoch": 0,
        "lambda_projection": 0.0,
        "refine_lr_scale": 1.0,
        "beta": 1.0,
        "last_window_size": int(config.get("last_window_size", 10)),
        "residual_rollout": bool(config.get("residual_rollout", True)),
        "server_rollout_depth": str(config.get("server_rollout_depth", "all")),
        "adaptive_discretization": bool(config.get("adaptive_discretization", False)),
        "semantic_speculation": bool(config.get("semantic_speculation", False)),
        "local_files_only": bool(config.get("local_files_only", True)),
        "residual_alpha_rho": 0.0,
        "fix_boundary_specials": bool(config.get("fix_boundary_specials", True)),
        "inverted_block_count": int(config.get("target_layer", 17)) + 1,
    }
    payload = {**defaults, **{key: value for key, value in config.items() if key in fields}}
    payload["method"] = "attn_linear_weighted_residual_schedule"
    payload["run_name"] = str(config.get("run_name", "attention_recoverability_diagnostic"))
    payload["output_dir"] = str(config.get("output_dir", "runs/attention_recoverability_diagnostic"))
    payload["top_k_embedding"] = int(config.get("top_k_embedding", config.get("K", 1)))
    payload["top_y_semantic"] = int(config.get("top_y_semantic", config.get("Y", 0)))
    return SAWConfig(**{key: payload[key] for key in fields})


def assert_strict_top1(config: Dict[str, Any], result_dir: Path) -> None:
    method = str(config.get("method", ""))
    k = int(config.get("top_k_embedding", config.get("K", 0)))
    y = int(config.get("top_y_semantic", config.get("Y", -1)))
    semantic = bool(config.get("semantic_speculation", False))
    if method != "B0V":
        raise RuntimeError(f"{result_dir} is not a B0V run: method={method}")
    if k != 1 or y != 0 or semantic:
        raise RuntimeError(f"{result_dir} is not strict Top-1: K={k}, Y={y}, semantic={semantic}")


def token_rows_for_result_dir(result_dir: Path, output_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    config = read_json(result_dir / "config.json")
    assert_strict_top1(config, result_dir)
    metrics = read_json(result_dir / "metrics.json")
    if int(metrics.get("failed_sample_count", 0)) != 0:
        raise RuntimeError(f"{result_dir} has failed samples")
    if not (result_dir / "COMPLETE").exists():
        raise RuntimeError(f"{result_dir} is missing COMPLETE")

    cfg = config_from_json(config)
    tokenizer, model, device = load_model_and_data(cfg)
    embed = model.get_input_embeddings()
    predictions = read_jsonl(result_dir / "predictions.jsonl")
    token_rows: List[Dict[str, Any]] = []
    prompt_objective_rows: List[Dict[str, Any]] = []

    for pred in predictions:
        prompt_id = int(pred["prompt_id"])
        original_ids = [int(x) for x in pred["original_token_ids"]]
        recovered_ids = [int(x) for x in pred["recovered_token_ids"]]
        if len(original_ids) != len(recovered_ids):
            raise RuntimeError(f"{result_dir} prompt {prompt_id} has mismatched token lengths")
        seq_len = len(original_ids)
        attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
        fixed_public = inferred_boundary_special_tokens(tokenizer, seq_len) if cfg.fix_boundary_specials else {}
        variable_audit = build_variable_mask(
            attention_mask=attention_mask,
            fixed_public=fixed_public,
            special_token_ids=getattr(tokenizer, "all_special_ids", None),
            token_ids=None,
        )
        orig = torch.tensor([original_ids], dtype=torch.long, device=device)
        rec = torch.tensor([recovered_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            observed = capture_prefix_activation(model, cfg.target_layer, input_ids=orig, attention_mask=attention_mask).detach()
            recovered_hidden = capture_prefix_activation(model, cfg.target_layer, input_ids=rec, attention_mask=attention_mask).detach()
            residual = torch.mean((recovered_hidden.float() - observed.float()) ** 2, dim=-1)[0]
            recovered_emb = embed(rec)[0].float()
            original_emb = embed(orig)[0].float()
            embedding_distance = torch.linalg.vector_norm(recovered_emb - original_emb, dim=-1)
            server_weights, _attn_stats, _rollout_stats, weight_stats = server_attention_bundle(
                model,
                observed,
                attention_mask,
                cfg,
                fixed_public,
                variable_audit,
            )
        if server_weights is None:
            raise RuntimeError("server_attention_bundle returned no weights")
        variable_mask = variable_audit.variable_mask.detach().cpu().bool().tolist()
        residual_cpu = residual.detach().cpu().tolist()
        dist_cpu = embedding_distance.detach().cpu().tolist()
        weights_cpu = server_weights.detach().cpu().float().tolist()
        prompt_residuals: List[float] = []
        prompt_weights: List[float] = []
        for position, is_variable in enumerate(variable_mask):
            if not is_variable:
                continue
            correct = int(original_ids[position] == recovered_ids[position])
            row = {
                "seed": int(config["seed"]),
                "layer": int(config["target_layer"]),
                "prompt_id": prompt_id,
                "position": int(position),
                "server_attention_importance": float(weights_cpu[position]),
                "attention_importance": float(weights_cpu[position]),
                "token_correct": correct,
                "activation_residual": float(residual_cpu[position]),
                "embedding_nn_distance": float(dist_cpu[position]),
                "source_result_dir": str(result_dir),
                "nn_distance_note": "evaluation proxy: ||E[recovered_token]-E[original_token]||; z_star projection distance was not saved",
            }
            token_rows.append(row)
            prompt_residuals.append(row["activation_residual"])
            prompt_weights.append(row["attention_importance"])
        cf = counterfactual_objectives(
            prompt_residuals,
            prompt_weights,
            random.Random(20260816 + int(config["seed"]) * 100000 + int(config["target_layer"]) * 1000 + prompt_id),
        )
        prompt_objective_rows.append(
            {
                "seed": int(config["seed"]),
                "layer": int(config["target_layer"]),
                "prompt_id": prompt_id,
                "variable_token_count": len(prompt_residuals),
                "uniform_objective": cf["uniform_objective"],
                "original_attention_objective": cf["original_attention_objective"],
                "shuffled_attention_objective": cf["shuffled_attention_objective"],
                "inverted_attention_objective": cf["inverted_attention_objective"],
                "attention_mean": mean(prompt_weights),
                "attention_std": statistics.pstdev(prompt_weights) if len(prompt_weights) > 1 else 0.0,
                "base_fixed_public_weight_sum": float(weight_stats.get("fixed_public_final_weight_sum", weight_stats.get("final_fixed_weight_sum", 0.0))),
                "source_result_dir": str(result_dir),
            }
        )
    return token_rows, prompt_objective_rows


def summarize_correlation(rows: List[Dict[str, Any]], scope: str, scope_value: str) -> Dict[str, Any]:
    attentions = [float(row["attention_importance"]) for row in rows]
    correctness = [float(row["token_correct"]) for row in rows]
    residuals = [float(row["activation_residual"]) for row in rows]
    dists = [float(row["embedding_nn_distance"]) for row in rows]
    return {
        "scope": scope,
        "scope_value": scope_value,
        "token_count": len(rows),
        "token_accuracy": mean(correctness),
        "spearman_attention_correctness": spearman(attentions, correctness),
        "spearman_attention_residual": spearman(attentions, residuals),
        "spearman_attention_embedding_nn_distance": spearman(attentions, dists),
        "mean_attention_importance": mean(attentions),
        "mean_activation_residual": mean(residuals),
        "mean_embedding_nn_distance": mean(dists),
    }


def summarize_quartiles(rows: List[Dict[str, Any]], scope: str, scope_value: str) -> List[Dict[str, Any]]:
    assigned = assign_quartiles(rows)
    out: List[Dict[str, Any]] = []
    for q in ("Q1", "Q2", "Q3", "Q4"):
        q_rows = [row for row in assigned if row["attention_quartile"] == q]
        out.append(
            {
                "scope": scope,
                "scope_value": scope_value,
                "quartile": q,
                "token_count": len(q_rows),
                "attention_min": min((float(row["attention_importance"]) for row in q_rows), default=float("nan")),
                "attention_max": max((float(row["attention_importance"]) for row in q_rows), default=float("nan")),
                "token_accuracy": mean([float(row["token_correct"]) for row in q_rows]),
                "mean_activation_residual": mean([float(row["activation_residual"]) for row in q_rows]),
                "mean_embedding_nn_distance": mean([float(row["embedding_nn_distance"]) for row in q_rows]),
            }
        )
    return out


def summarize_counterfactual(rows: List[Dict[str, Any]], scope: str, scope_value: str) -> Dict[str, Any]:
    return {
        "scope": scope,
        "scope_value": scope_value,
        "prompt_count": len(rows),
        "uniform_objective": mean([float(row["uniform_objective"]) for row in rows]),
        "original_attention_objective": mean([float(row["original_attention_objective"]) for row in rows]),
        "shuffled_attention_objective": mean([float(row["shuffled_attention_objective"]) for row in rows]),
        "inverted_attention_objective": mean([float(row["inverted_attention_objective"]) for row in rows]),
        "original_minus_uniform": mean([float(row["original_attention_objective"]) - float(row["uniform_objective"]) for row in rows]),
        "shuffled_minus_uniform": mean([float(row["shuffled_attention_objective"]) - float(row["uniform_objective"]) for row in rows]),
        "inverted_minus_uniform": mean([float(row["inverted_attention_objective"]) - float(row["uniform_objective"]) for row in rows]),
    }


def group_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


def build_report(
    output_dir: Path,
    correlation_rows: List[Dict[str, Any]],
    quartile_rows: List[Dict[str, Any]],
    counterfactual_rows: List[Dict[str, Any]],
    result_dirs: Sequence[str],
) -> None:
    global_corr = next(row for row in correlation_rows if row["scope"] == "all")
    layer_corrs = [row for row in correlation_rows if row["scope"] == "layer"]
    seed_corrs = [row for row in correlation_rows if row["scope"] == "seed"]
    all_quartiles = [row for row in quartile_rows if row["scope"] == "all"]
    all_counter = next(row for row in counterfactual_rows if row["scope"] == "all")
    stable_positive = (
        float(global_corr["spearman_attention_correctness"]) > 0.1
        and all(float(row["spearman_attention_correctness"]) > 0.1 for row in layer_corrs)
        and all(float(row["spearman_attention_correctness"]) > 0.1 for row in seed_corrs)
    )
    unreliable = (not math.isfinite(float(global_corr["spearman_attention_correctness"]))) or abs(float(global_corr["spearman_attention_correctness"])) < 0.1
    inconsistent = len({math.copysign(1, float(row["spearman_attention_correctness"])) for row in layer_corrs if math.isfinite(float(row["spearman_attention_correctness"]))}) > 1
    lines = [
        "# Attention-Recoverability Correlation Diagnostic",
        "",
        "## Scope",
        "",
        "This diagnostic is evaluation-only. It uses completed strict Top-1 B0V runs and does not modify B0V, rerun prompt inversion optimization, tune thresholds, or add attention-weighting components.",
        "",
        "Ground truth token ids are used only after the attack has completed, to compute correctness and evaluation residuals.",
        "",
        "Input result directories:",
    ]
    lines.extend([f"- `{item}`" for item in result_dirs])
    lines.extend(
        [
            "",
            "## Main Answer",
            "",
            f"- Spearman(attention importance, token correctness): {float(global_corr['spearman_attention_correctness']):.6f}",
            f"- Spearman(attention importance, activation residual): {float(global_corr['spearman_attention_residual']):.6f}",
            f"- Spearman(attention importance, embedding NN distance proxy): {float(global_corr['spearman_attention_embedding_nn_distance']):.6f}",
            "",
        ]
    )
    if stable_positive:
        lines.append("The diagnostic supports a positive, cross-layer/cross-seed relationship in the completed data. A new attention-aware initialization method may be worth proposing, but it is not implemented in this round.")
    elif unreliable or inconsistent:
        lines.append("server-side attention importance is not a reliable predictor of prompt inversion recoverability.")
    else:
        lines.append("The relationship is weak or mixed and does not justify adding more attention-weighting components in this round.")
    lines.extend(["", "## Quartile Accuracy", "", "| Quartile | Token Accuracy | Residual | NN distance proxy |", "| --- | ---: | ---: | ---: |"])
    for row in all_quartiles:
        lines.append(
            f"| {row['quartile']} | {float(row['token_accuracy']):.6f} | {float(row['mean_activation_residual']):.6g} | {float(row['mean_embedding_nn_distance']):.6f} |"
        )
    lines.extend(["", "## Layer-Wise Correlation", "", "| Layer | Token Count | Spearman correct | Token Accuracy | Residual |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in layer_corrs:
        lines.append(
            f"| {row['scope_value']} | {row['token_count']} | {float(row['spearman_attention_correctness']):.6f} | {float(row['token_accuracy']):.6f} | {float(row['mean_activation_residual']):.6g} |"
        )
    lines.extend(["", "## Seed-Wise Correlation", "", "| Seed | Token Count | Spearman correct | Token Accuracy |", "| --- | ---: | ---: | ---: |"])
    for row in seed_corrs:
        lines.append(f"| {row['scope_value']} | {row['token_count']} | {float(row['spearman_attention_correctness']):.6f} | {float(row['token_accuracy']):.6f} |")
    lines.extend(
        [
            "",
            "## Counterfactual Attention Objective",
            "",
            "| Variant | Mean objective | Delta vs uniform |",
            "| --- | ---: | ---: |",
            f"| Uniform | {float(all_counter['uniform_objective']):.8f} | 0.00000000 |",
            f"| Original Attention | {float(all_counter['original_attention_objective']):.8f} | {float(all_counter['original_minus_uniform']):.8f} |",
            f"| Shuffled Attention | {float(all_counter['shuffled_attention_objective']):.8f} | {float(all_counter['shuffled_minus_uniform']):.8f} |",
            f"| Inverted Attention | {float(all_counter['inverted_attention_objective']):.8f} | {float(all_counter['inverted_minus_uniform']):.8f} |",
            "",
            "## Required Questions",
            "",
            f"1. Positive correlation with recovery accuracy: {'yes' if float(global_corr['spearman_attention_correctness']) > 0.1 else 'no/weak'}.",
            "2. High-attention tokens are easier to recover only if Q4 accuracy exceeds Q1-Q3 consistently; see quartile table.",
            f"3. High-attention tokens have lower activation residual only if Q4 residual is lower; observed Q4 residual is {float(all_quartiles[-1]['mean_activation_residual']):.6g}.",
            "4. Shuffled attention is diagnostic-only; compare its objective against original attention above.",
            "5. Inverted attention is diagnostic-only; compare its objective against original/uniform above.",
            f"6. Cross-layer stability: {'stable positive' if stable_positive else 'not established'}.",
            f"7. Cross-seed stability: {'not testable beyond completed seed(s)' if len(seed_corrs) <= 1 else ('stable positive' if stable_positive else 'not established')}.",
            "",
            "## Notes",
            "",
            "- `embedding_nn_distance` is an evaluation proxy `||E[recovered_token] - E[original_token]||` because historical B0V outputs did not save the continuous optimized embedding `z*`.",
            "- Fixed public framing positions are excluded using the same public-mask semantics as the attack path. Token ids are not passed to the attack path for special-token masking.",
        ]
    )
    (output_dir / "attention_recoverability_diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path("analysis/attention_recoverability_diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dirs = [Path(item) for item in args.result_dirs]
    all_token_rows: List[Dict[str, Any]] = []
    all_prompt_objective_rows: List[Dict[str, Any]] = []
    for result_dir in result_dirs:
        token_rows, objective_rows = token_rows_for_result_dir(result_dir, output_dir)
        all_token_rows.extend(token_rows)
        all_prompt_objective_rows.extend(objective_rows)

    write_csv(output_dir / "token_position_records.csv", all_token_rows)
    write_csv(output_dir / "prompt_counterfactual_records.csv", all_prompt_objective_rows)

    correlation_rows = [summarize_correlation(all_token_rows, "all", "all")]
    for layer, rows in sorted(group_by(all_token_rows, "layer").items(), key=lambda item: int(item[0])):
        correlation_rows.append(summarize_correlation(rows, "layer", layer))
    for seed, rows in sorted(group_by(all_token_rows, "seed").items(), key=lambda item: int(item[0])):
        correlation_rows.append(summarize_correlation(rows, "seed", seed))
    write_csv(output_dir / "correlation_summary.csv", correlation_rows)

    quartile_rows: List[Dict[str, Any]] = []
    quartile_rows.extend(summarize_quartiles(all_token_rows, "all", "all"))
    for layer, rows in sorted(group_by(all_token_rows, "layer").items(), key=lambda item: int(item[0])):
        quartile_rows.extend(summarize_quartiles(rows, "layer", layer))
    for seed, rows in sorted(group_by(all_token_rows, "seed").items(), key=lambda item: int(item[0])):
        quartile_rows.extend(summarize_quartiles(rows, "seed", seed))
    write_csv(output_dir / "quartile_summary.csv", quartile_rows)

    layer_summary = [row for row in correlation_rows if row["scope"] == "layer"]
    seed_summary = [row for row in correlation_rows if row["scope"] == "seed"]
    write_csv(output_dir / "layer_summary.csv", layer_summary)
    write_csv(output_dir / "seed_summary.csv", seed_summary)

    counterfactual_rows = [summarize_counterfactual(all_prompt_objective_rows, "all", "all")]
    for layer, rows in sorted(group_by(all_prompt_objective_rows, "layer").items(), key=lambda item: int(item[0])):
        counterfactual_rows.append(summarize_counterfactual(rows, "layer", layer))
    for seed, rows in sorted(group_by(all_prompt_objective_rows, "seed").items(), key=lambda item: int(item[0])):
        counterfactual_rows.append(summarize_counterfactual(rows, "seed", seed))
    write_csv(output_dir / "counterfactual_attention_summary.csv", counterfactual_rows)

    write_json(
        output_dir / "run_config.json",
        {
            "result_dirs": [str(path) for path in result_dirs],
            "output_dir": str(output_dir),
            "diagnostic_only": True,
            "attack_path_reads_ground_truth": False,
            "ground_truth_use": "evaluation labels, residuals, and embedding-distance proxy only",
        },
    )
    build_report(output_dir, correlation_rows, quartile_rows, counterfactual_rows, [str(path) for path in result_dirs])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/attention_recoverability_diagnostic")
    parser.add_argument("--result-dirs", nargs="+", default=DEFAULT_RESULT_DIRS)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
