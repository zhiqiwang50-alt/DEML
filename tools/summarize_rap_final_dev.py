import csv
import json
import pathlib
import statistics


ROOT = pathlib.Path("runs/rap_final")
ANALYSIS = pathlib.Path("analysis")
ANALYSIS.mkdir(exist_ok=True)
METHODS = [
    "original_pia_baseline",
    "variable_only_uniform",
    "sparse_fixed",
    "rel_order",
    "rap_cont_gate",
    "rap_final",
    "shuffled_rel",
]
SEEDS = [42, 43, 44]


def load_jsonl(path: pathlib.Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: pathlib.Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired(a_rows, b_rows):
    a = {int(row["prompt_id"]): row for row in a_rows}
    b = {int(row["prompt_id"]): row for row in b_rows}
    prompt_ids = sorted(set(a) & set(b))
    acc_delta = [float(a[i]["token_accuracy"]) - float(b[i]["token_accuracy"]) for i in prompt_ids]
    bleu_delta = [float(a[i]["bleu"]) - float(b[i]["bleu"]) for i in prompt_ids]
    return {
        "prompt_count": len(prompt_ids),
        "acc_delta": statistics.mean(acc_delta),
        "bleu_delta": statistics.mean(bleu_delta),
        "win": sum(x > 1e-12 for x in acc_delta),
        "tie": sum(abs(x) <= 1e-12 for x in acc_delta),
        "loss": sum(x < -1e-12 for x in acc_delta),
    }


rows = []
for seed in SEEDS:
    for method in METHODS:
        metrics = json.loads((ROOT / f"dev_seed{seed}" / method / "metrics.json").read_text(encoding="utf-8"))
        rows.append({
            "seed": seed,
            "method": method,
            "token_accuracy": metrics["token_accuracy"]["mean"],
            "bleu": metrics["bleu"]["mean"],
            "completed": metrics["completed_sample_count"],
            "failed": metrics["failed_sample_count"],
            "has_nan": metrics.get("has_nan"),
        })
write_csv(ROOT / "dev_component_summary_clean.csv", rows)

comparisons = {
    "sparse_fixed_vs_b0v": ("sparse_fixed", "variable_only_uniform"),
    "rel_order_vs_b0v": ("rel_order", "variable_only_uniform"),
    "rap_cont_gate_vs_b0v": ("rap_cont_gate", "variable_only_uniform"),
    "rap_final_vs_b0v": ("rap_final", "variable_only_uniform"),
    "shuffled_rel_vs_b0v": ("shuffled_rel", "variable_only_uniform"),
    "rap_final_vs_shuffled_rel": ("rap_final", "shuffled_rel"),
    "rap_final_vs_rap_cont_gate": ("rap_final", "rap_cont_gate"),
    "rap_final_vs_b0": ("rap_final", "original_pia_baseline"),
}
paired_rows = []
for seed in SEEDS:
    for comparison, (a_method, b_method) in comparisons.items():
        a = load_jsonl(ROOT / f"dev_seed{seed}" / a_method / "predictions.jsonl")
        b = load_jsonl(ROOT / f"dev_seed{seed}" / b_method / "predictions.jsonl")
        paired_rows.append({"seed": seed, "comparison": comparison, **paired(a, b)})
write_csv(ROOT / "dev_paired_summary.csv", paired_rows)

pooled = []
for method in METHODS:
    vals = [row for row in rows if row["method"] == method]
    pooled.append({
        "method": method,
        "token_accuracy": statistics.mean(row["token_accuracy"] for row in vals),
        "bleu": statistics.mean(row["bleu"] for row in vals),
        "failed": sum(row["failed"] for row in vals),
        "has_nan": any(row["has_nan"] for row in vals),
    })
write_csv(ROOT / "dev_pooled_summary.csv", pooled)

scale_rows = []
for path in sorted(ROOT.glob("dev_seed*/rap_final/relational_graph_stats.json")):
    seed = path.parents[1].name.replace("dev_seed", "")
    data = json.loads(path.read_text(encoding="utf-8"))
    for sample in data.get("samples", []):
        for layer in sample.get("layer_stats", []):
            scale_rows.append({
                "seed": seed,
                "prompt_id": sample.get("prompt_id"),
                "layer": layer.get("layer"),
                "raw_mean": layer.get("raw_mean"),
                "normalized_mean": layer.get("normalized_mean"),
                "raw_max": layer.get("raw_max"),
                "scale_factor": layer.get("scale_factor"),
            })
if scale_rows:
    write_csv(ROOT / "dev_layer_scale_audit.csv", scale_rows)

report = [
    "# RAP-FINAL Dev Report",
    "",
    "Seed42/43/44 are development/diagnostic only, not final held-out evidence.",
    "",
    "## Pooled Metrics",
    "",
    "| method | Acc | BLEU | failed | NaN |",
    "| --- | ---: | ---: | ---: | --- |",
]
for row in pooled:
    report.append("| {method} | {acc:.6f} | {bleu:.6f} | {failed} | {nan} |".format(
        method=row["method"],
        acc=row["token_accuracy"],
        bleu=row["bleu"],
        failed=row["failed"],
        nan=row["has_nan"],
    ))
report.extend([
    "",
    "## Key Attribution",
    "",
    "- Margin uncertainty audit kept margin-only detector: enrichment ratio 3.211586.",
    "- SPARSE_FIXED vs B0V is the most plausible useful component on development, but it is still seed-dependent.",
    "- RAP_FINAL does not beat SHUFFLED_REL on development; relational structure cannot be claimed as the core contributor.",
    "- RAP_FINAL is often equal to B0V because the discrete Top-1 acceptance gate rejects candidates that do not improve the discrete path.",
    "- The archived GPU0 OOM partial result is kept under `dev_seed44/sparse_fixed_oom_gpu0_*` and is excluded from clean summaries.",
    "",
    "Detailed CSV outputs: `runs/rap_final/dev_component_summary_clean.csv`, `runs/rap_final/dev_pooled_summary.csv`, `runs/rap_final/dev_paired_summary.csv`.",
])
(ANALYSIS / "rap_final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
(ANALYSIS / "rap_layer_scale_audit.md").write_text(
    "# RAP Layer Scale Audit\n\n"
    "Layer-wise normalization uses legal variable causal edges `q > k`. "
    "Full CSV: `runs/rap_final/dev_layer_scale_audit.csv`.\n\n"
    f"Scale rows: {len(scale_rows)}\n",
    encoding="utf-8",
)
print(f"wrote dev summaries rows={len(rows)} pairs={len(paired_rows)} scale={len(scale_rows)}")
