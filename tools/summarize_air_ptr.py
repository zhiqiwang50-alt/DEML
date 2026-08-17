#!/usr/bin/env python3
"""Read-only AIR-PTR experiment summarizer.

This script intentionally does not import pia_air_ptr.py or load a model.  It
only reads completed JSON outputs and writes aggregate CSV/Markdown audit files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List


RHO_VALUES = [0.0, 0.25, 0.50, 0.75, 1.0]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_md_table(path: Path, title: str, rows: List[Dict[str, Any]]) -> None:
    lines = [f"# {title}", ""]
    if rows:
        keys = list(rows[0])
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_rho(rho: float) -> str:
    if abs(rho - 0.5) < 1e-9:
        return "rho0p50"
    return f"rho{str(rho).replace('.', 'p')}"


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def summarize_weight_stats(path: Path) -> Dict[str, float]:
    stats = read_json(path)
    samples = stats.get("samples", [])
    numeric_fields = [
        "fixed_public_final_weight_sum",
        "max_variable_final_weight",
        "mean_variable_final_weight",
        "std_variable_final_weight",
        "final_bos_alpha",
        "final_last_valid_alpha",
        "raw_bos_mass",
        "raw_last_valid_mass",
    ]
    summary: Dict[str, float] = {}
    for field in numeric_fields:
        values = [float(sample[field]) for sample in samples if field in sample and finite(sample[field])]
        if values:
            summary[f"{field}_mean"] = statistics.mean(values)
            summary[f"{field}_max"] = max(values)
    summary["sample_count_in_weight_stats"] = len(samples)
    return summary


def summarize_rho(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    rows: List[Dict[str, Any]] = []
    for layer in args.layers:
        for rho in RHO_VALUES:
            output_dir = root / "rho_ablation" / f"layer{layer}" / fmt_rho(rho)
            metrics_path = output_dir / "metrics.json"
            complete = (output_dir / "COMPLETE").exists()
            if not metrics_path.exists():
                rows.append(
                    {
                        "layer": layer,
                        "rho": rho,
                        "Token Accuracy": "",
                        "BLEU": "",
                        "completed": 0,
                        "failed": "",
                        "NaN": "",
                        "base_fixed_public_weight_sum_max": "",
                        "base_max_variable_weight": "",
                        "valid": False,
                        "output_dir": str(output_dir),
                    }
                )
                continue
            metrics = read_json(metrics_path)
            weights = summarize_weight_stats(output_dir / "layer_adaptive_stats.json")
            valid = (
                complete
                and int(metrics.get("completed_sample_count", 0)) == int(args.dataset_len)
                and int(metrics.get("failed_sample_count", 0)) == 0
                and int(metrics.get("nan_count", 0)) == 0
                and float(weights.get("fixed_public_final_weight_sum_max", 0.0)) <= 1e-6
                and float(weights.get("max_variable_final_weight_max", 0.0)) <= 1.5 + 1e-6
            )
            rows.append(
                {
                    "layer": layer,
                    "rho": rho,
                    "Token Accuracy": metrics["token_accuracy"]["mean"],
                    "BLEU": metrics["bleu"]["mean"],
                    "completed": metrics.get("completed_sample_count", ""),
                    "failed": metrics.get("failed_sample_count", ""),
                    "NaN": metrics.get("nan_count", ""),
                    "base_fixed_public_weight_sum_max": weights.get("fixed_public_final_weight_sum_max", ""),
                    "base_max_variable_weight": weights.get("max_variable_final_weight_max", ""),
                    "valid": valid,
                    "output_dir": str(output_dir),
                }
            )

    write_csv(root / "dev_rho_ablation.csv", rows)
    write_md_table(root / "dev_rho_ablation.md", "AIR-PTR rho ablation", rows)

    selected: List[Dict[str, Any]] = []
    for layer in args.layers:
        valid_rows = [row for row in rows if int(row["layer"]) == int(layer) and row["valid"]]
        valid_rows.sort(key=lambda row: (-float(row["Token Accuracy"]), -float(row["BLEU"]), float(row["rho"])))
        if not valid_rows:
            raise RuntimeError(f"no valid rho candidate for layer {layer}")
        selected.append(valid_rows[0])
    write_csv(root / "layer_rho_selection.csv", selected)
    write_md_table(root / "layer_rho_selection.md", "AIR-PTR layer rho selection", selected)

    global_rows = []
    for rho in RHO_VALUES:
        matching = [row for row in rows if float(row["rho"]) == rho and row["valid"]]
        if len(matching) == len(args.layers):
            global_rows.append(
                {
                    "rho": rho,
                    "mean_Token_Accuracy": statistics.mean(float(row["Token Accuracy"]) for row in matching),
                    "mean_BLEU": statistics.mean(float(row["BLEU"]) for row in matching),
                }
            )
    global_rows.sort(key=lambda row: (-row["mean_Token_Accuracy"], -row["mean_BLEU"], row["rho"]))
    write_csv(root / "global_rho_selection.csv", global_rows)
    write_md_table(root / "global_rho_selection.md", "AIR-PTR global rho selection", global_rows)
    write_json(
        root / "frozen_layer_config.json",
        {
            "gamma": 0.3,
            "rho_by_layer": {str(row["layer"]): float(row["rho"]) for row in selected},
            "rho_global": float(global_rows[0]["rho"]) if global_rows else None,
            "weight_source": "mean_query",
            "weight_power": 0.5,
            "residual_rollout": True,
            "rollout_depth": "all",
            "attention_start_ratio": 0.50,
            "attention_full_ratio": 0.70,
            "weight_min": 0.5,
            "weight_max": 1.5,
            "strict_top1": {"K": 1, "Y": 0, "semantic_speculation": False},
            "ptr": {
                "eta": 0.10,
                "repair_fraction": 0.20,
                "max_positions_per_pass": 8,
                "max_passes": 2,
                "epsilon_accept": 1e-6,
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rho"], required=True)
    parser.add_argument("--output-root", default="runs/air_ptr")
    parser.add_argument("--dataset-len", type=int, default=28)
    parser.add_argument("--layers", type=int, nargs="+", default=[11, 17, 19])
    args = parser.parse_args()
    if args.mode == "rho":
        summarize_rho(args)


if __name__ == "__main__":
    main()
