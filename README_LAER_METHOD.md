# DEML LAER Prompt Inversion Experiments

This branch is a clean LAER method package for white-box TinyLlama prompt inversion experiments. It is intended for a fresh Codex session on another computer, without relying on prior chat context.

## What This Branch Contains

LAER means Local Attention Edge Residual repair. The branch keeps the original paper-style PIA baseline `B0`, adds an uncertainty-based sparse repair baseline `B0_SPARSE`, and evaluates the relation-aware method `B0_LAER` against a shuffled relation control `B0_SHUFFLED_LAER`.

The final naming is LAER-specific:

- final driver: `laer_final_experiment_driver.py`
- final result root: `runs/laer_final/`
- final report: `analysis/laer_final_report.md`

The LAER driver in this branch writes to `runs/laer_final/`.

## Threat Model

The experiments use a white-box academic setting:

- the attacker knows the full TinyLlama model parameters;
- the attacker observes the transmitted intermediate activation at the collaborative split point;
- the attacker can run the server-side suffix locally/offline;
- the attack API must not read `prompt`, `original_ids`, `original_tokens`, or ground-truth token ids during reconstruction;
- ground truth is used only for final evaluation metrics.

## Main Files

| Path | Purpose |
|---|---|
| `pia_tinyllama.py` | TinyLlama PIA reproduction and shared model/evaluation utilities. |
| `pia_b0_laer.py` | Main B0/B0_SPARSE/B0_LAER/B0_SHUFFLED_LAER experiment implementation. |
| `pia_b0_attention_consistency_repair.py` | Sparse repair and attention-consistency helper used by LAER. |
| `pia_attention_validation_audit.py` | Read-only attention validation audit. |
| `laer_relation_isolation_utils.py` | LAER relation utility code, renamed from the earlier RAP-oriented file name. |
| `laer_final_experiment_driver.py` | Frozen final experiment driver. |
| `scripts/run_laer.sh` | Convenience wrapper for LAER single/verify runs. |
| `tests/` | Unit tests for API safety, Top-1 behavior, relation controls, and repair bookkeeping. |

## Data

| Path | Purpose |
|---|---|
| `data/airline.json` | Skytrax-28 development set. |
| `data/skytrax_150.json` | Skytrax-150 validation source. |
| `runs/suffix_attention_edge_rerank_pia/splits/skytrax150_split_20260706.json` | Fixed split used by the frozen final experiment. |

## Setup

```bash
conda create -n deml python=3.10 -y
conda activate deml
pip install -r requirements.txt
```

If TinyLlama is already cached, keep `--local-files-only`. Otherwise, omit it for the first model download.

## Quick Checks

```bash
python -m py_compile pia_tinyllama.py pia_b0_laer.py pia_b0_attention_consistency_repair.py pia_attention_validation_audit.py laer_final_experiment_driver.py
python -m unittest tests.test_b0_laer tests.test_b0_attention_consistency_repair tests.test_attention_validation_audit -v
python - <<'PY'
import csv
rows = list(csv.DictReader(open("runs/laer_final/laer_final_completeness_audit.csv")))
print(sum(r["valid_complete"] == "True" for r in rows), "/", len(rows))
PY
```

## Reproducing the Frozen Final Experiment

The frozen final experiment uses strict Top-1 reconstruction and writes to `runs/laer_final/`. The GitHub branch keeps compact final results and raw predictions, but omits very large per-token intermediate audit traces.

```bash
python laer_final_experiment_driver.py --mode all --local-files-only
```

For individual runs, use `pia_b0_laer.py` or the wrapper:

```bash
PYTHON_BIN=$(which python) scripts/run_laer.sh --mode verify
```

## Final Results Summary

Read the final report first:

- `analysis/laer_final_report.md`
- `runs/laer_final/laer_final_pooled_summary.csv`
- `runs/laer_final/laer_final_paired_comparisons.csv`

Main layer17 pooled over seeds 42/43/44, 300 prompt-runs per method:

| Method | Token Accuracy | BLEU | Delta Acc vs B0 | Delta BLEU vs B0 |
|---|---:|---:|---:|---:|
| B0 | 0.838174 | 0.649034 | 0.000000 | 0.000000 |
| B0_SPARSE | 0.868064 | 0.713764 | +0.029889 | +0.064730 |
| B0_LAER | 0.867192 | 0.711139 | +0.029018 | +0.062105 |
| B0_SHUFFLED_LAER | 0.857869 | 0.692163 | +0.019695 | +0.043129 |

Conservative interpretation: LAER improves over the original B0 baseline and over the shuffled relation control, supporting a positive contribution from true local attention edge relations. However, B0_SPARSE is slightly higher than B0_LAER in this frozen run, so the paper should not claim that LAER is uniformly stronger than sparse repair or use wording like stable/significant improvement without additional evidence.
