# LAER Final Experiment Report

Generated at: 2026-09-05 23:06:49

## Status

Final status: **PAPER_FINAL_COMPLETE**. All 20 formal runs are complete and valid: 12/12 main layer17 runs and 8/8 layer robustness runs.

This report only summarizes frozen experiments. It does not modify attack algorithms, search parameters, or rerun completed results.

## Frozen Setting

- Dataset: `data/skytrax_150.json`
- Split: `runs/suffix_attention_edge_rerank_pia/splits/skytrax150_split_20260706.json`
- Split SHA256: `bea33b8f74654a3f98a9fa474521931e96614c17d6f80fb0282124c2688fbe86`
- Dataset SHA256: `f93e82c5ea50cbf2d20869ca2850959f5306631aa2210d63556b2626cceec108`
- Holdout count used by LAER final provenance: `100`
- Dev overlap count: `0`
- Main seeds: 42 / 43 / 44
- Main layer: 17
- Robustness layers: 11 / 19 with seed42
- Strict Top-1: `top_k_embedding=1`, `top_y_semantic=0`
- Epoch: 100

Integrity audit: `True`. Smoke audit passed: `True`.

## Run Completeness

| scope | layer | seed | method | valid | pred | completed | failed | Acc | BLEU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | 17 | 42 | B0 | yes | 100 | 100 | 0 | 0.836287 | 0.649887 |
| main | 17 | 42 | B0_SPARSE | yes | 100 | 100 | 0 | 0.867182 | 0.714965 |
| main | 17 | 42 | B0_LAER | yes | 100 | 100 | 0 | 0.863880 | 0.707264 |
| main | 17 | 42 | B0_SHUFFLED_LAER | yes | 100 | 100 | 0 | 0.855977 | 0.692345 |
| main | 17 | 43 | B0 | yes | 100 | 100 | 0 | 0.833378 | 0.636980 |
| main | 17 | 43 | B0_SPARSE | yes | 100 | 100 | 0 | 0.864915 | 0.706309 |
| main | 17 | 43 | B0_LAER | yes | 100 | 100 | 0 | 0.865709 | 0.707964 |
| main | 17 | 43 | B0_SHUFFLED_LAER | yes | 100 | 100 | 0 | 0.854916 | 0.684596 |
| main | 17 | 44 | B0 | yes | 100 | 100 | 0 | 0.844858 | 0.660235 |
| main | 17 | 44 | B0_SPARSE | yes | 100 | 100 | 0 | 0.872094 | 0.720017 |
| main | 17 | 44 | B0_LAER | yes | 100 | 100 | 0 | 0.871988 | 0.718188 |
| main | 17 | 44 | B0_SHUFFLED_LAER | yes | 100 | 100 | 0 | 0.862714 | 0.699548 |
| robustness | 11 | 42 | B0 | yes | 100 | 100 | 0 | 0.783754 | 0.611612 |
| robustness | 11 | 42 | B0_SPARSE | yes | 100 | 100 | 0 | 0.823700 | 0.684616 |
| robustness | 11 | 42 | B0_LAER | yes | 100 | 100 | 0 | 0.821523 | 0.681791 |
| robustness | 11 | 42 | B0_SHUFFLED_LAER | yes | 100 | 100 | 0 | 0.816160 | 0.663610 |
| robustness | 19 | 42 | B0 | yes | 100 | 100 | 0 | 0.834651 | 0.637417 |
| robustness | 19 | 42 | B0_SPARSE | yes | 100 | 100 | 0 | 0.859215 | 0.693778 |
| robustness | 19 | 42 | B0_LAER | yes | 100 | 100 | 0 | 0.858724 | 0.688855 |
| robustness | 19 | 42 | B0_SHUFFLED_LAER | yes | 100 | 100 | 0 | 0.849757 | 0.671373 |

## Main Results by Seed

| seed | method | Token Acc | BLEU |
| --- | --- | --- | --- |
| 42 | B0 | 0.836287 | 0.649887 |
| 42 | B0_SPARSE | 0.867182 | 0.714965 |
| 42 | B0_LAER | 0.863880 | 0.707264 |
| 42 | B0_SHUFFLED_LAER | 0.855977 | 0.692345 |
| 43 | B0 | 0.833378 | 0.636980 |
| 43 | B0_SPARSE | 0.864915 | 0.706309 |
| 43 | B0_LAER | 0.865709 | 0.707964 |
| 43 | B0_SHUFFLED_LAER | 0.854916 | 0.684596 |
| 44 | B0 | 0.844858 | 0.660235 |
| 44 | B0_SPARSE | 0.872094 | 0.720017 |
| 44 | B0_LAER | 0.871988 | 0.718188 |
| 44 | B0_SHUFFLED_LAER | 0.862714 | 0.699548 |

## Main Pooled Results

Pooled over layer17 seeds 42/43/44, 300 prompt-runs per method. Delta is relative to B0.

| method | N | Token Acc | BLEU | Acc delta vs B0 | BLEU delta vs B0 |
| --- | --- | --- | --- | --- | --- |
| B0 | 300 | 0.838174 | 0.649034 | 0.000000 | 0.000000 |
| B0_SPARSE | 300 | 0.868064 | 0.713764 | 0.029889 | 0.064730 |
| B0_LAER | 300 | 0.867192 | 0.711139 | 0.029018 | 0.062105 |
| B0_SHUFFLED_LAER | 300 | 0.857869 | 0.692163 | 0.019695 | 0.043129 |

## Paired Comparisons

Bootstrap uses 10,000 paired resamples over the 300 main prompt-runs. W/T/L is based on per-prompt Token Accuracy.

| comparison | N | Acc delta | Acc 95% CI | BLEU delta | BLEU 95% CI | W/T/L |
| --- | --- | --- | --- | --- | --- | --- |
| B0_SPARSE vs B0 | 300 | 0.029889 | [0.027551, 0.032247] | 0.064730 | [0.059624, 0.069864] | 268/23/9 |
| B0_LAER vs B0 | 300 | 0.029018 | [0.026613, 0.031497] | 0.062105 | [0.056990, 0.067461] | 266/29/5 |
| B0_SHUFFLED_LAER vs B0 | 300 | 0.019695 | [0.017335, 0.021970] | 0.043129 | [0.038763, 0.047622] | 237/54/9 |
| B0_LAER vs B0_SPARSE | 300 | -0.000871 | [-0.002574, 0.000944] | -0.002624 | [-0.006041, 0.001008] | 76/127/97 |
| B0_LAER vs B0_SHUFFLED_LAER | 300 | 0.009323 | [0.007218, 0.011621] | 0.018976 | [0.014910, 0.023213] | 164/107/29 |
| B0_SPARSE vs B0_SHUFFLED_LAER | 300 | 0.010195 | [0.008078, 0.012421] | 0.021601 | [0.017519, 0.025823] | 176/85/39 |

## Layer Robustness

Layer robustness is a seed42 check only and should not be interpreted as a multi-seed stability claim. Delta is relative to B0 at the same layer.

| layer | method | N | Token Acc | BLEU | Acc delta vs B0 | BLEU delta vs B0 |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | B0 | 100 | 0.783754 | 0.611612 | 0.000000 | 0.000000 |
| 11 | B0_SPARSE | 100 | 0.823700 | 0.684616 | 0.039945 | 0.073004 |
| 11 | B0_LAER | 100 | 0.821523 | 0.681791 | 0.037769 | 0.070179 |
| 11 | B0_SHUFFLED_LAER | 100 | 0.816160 | 0.663610 | 0.032405 | 0.051998 |
| 19 | B0 | 100 | 0.834651 | 0.637417 | 0.000000 | 0.000000 |
| 19 | B0_SPARSE | 100 | 0.859215 | 0.693778 | 0.024564 | 0.056362 |
| 19 | B0_LAER | 100 | 0.858724 | 0.688855 | 0.024073 | 0.051438 |
| 19 | B0_SHUFFLED_LAER | 100 | 0.849757 | 0.671373 | 0.015106 | 0.033956 |

## Interpretation

- B0_SPARSE vs B0: Acc 0.029889, BLEU 0.064730, W/T/L 268/23/9.
- B0_LAER vs B0: Acc 0.029018, BLEU 0.062105, W/T/L 266/29/5.
- B0_LAER does not exceed B0_SPARSE on pooled main metrics; sparse repair remains the stronger component in this frozen run.
- B0_LAER exceeds B0_SHUFFLED_LAER, supporting that the real local attention edge relation contributes beyond the shuffled control.

In this frozen experiment, both B0_SPARSE and B0_LAER improve over the original B0 baseline on the main layer17 heldout setting. However, B0_SPARSE is slightly higher than B0_LAER in pooled Acc/BLEU. The strongest evidence for LAER's relational component is that B0_LAER outperforms B0_SHUFFLED_LAER, meaning the real local attention edge relation is better than a relation-randomized control under the same repair budget.

The wording in the paper should therefore be conservative: LAER provides a relation-aware repair signal that improves over B0 and over shuffled relation control, while sparse uncertainty repair accounts for a substantial part of the gain. Do not call it a stable or significant improvement unless additional statistical framing is added and accepted.

## Generated Files

- `runs/laer_final/laer_final_completeness_audit.csv`
- `runs/laer_final/laer_final_completeness_audit_final.json`
- `runs/laer_final/laer_final_run_metrics.csv`
- `runs/laer_final/laer_final_pooled_summary.csv`
- `runs/laer_final/laer_final_paired_comparisons.csv`
- `runs/laer_final/laer_final_paired_comparisons.md`
- `runs/laer_final/laer_final_prompt_order_audit.csv`
- `runs/laer_final/laer_final_config_audit.csv`
- `runs/laer_final/laer_final_mechanism_readonly_diagnostic.csv`
