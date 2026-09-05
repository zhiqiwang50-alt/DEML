# LAER Final Experiment Interim Report

- Generated at: 2026-09-05 10:19:19
- Scope: frozen strict Top-1 paper-final experiment for B0, B0_SPARSE, B0_LAER, and B0_SHUFFLED_LAER.
- Note: partial rows use the mean over currently written `predictions.jsonl`; they are not final metrics.

## Completion Overview
- Valid completed runs: 18/20
- Started runs: 20/20
- Remaining active/partial runs: 2

## Per-run Metrics
| Phase | Layer | Seed | Method | Status | Predictions | Failed | Token Acc | BLEU |
|---|---:|---:|---|---|---:|---:|---:|---:|
| main | 17 | 42 | B0 | VALID_COMPLETE | 100 | 0 | 0.836287 | 0.649887 |
| main | 17 | 42 | B0_SPARSE | VALID_COMPLETE | 100 | 0 | 0.867182 | 0.714965 |
| main | 17 | 42 | B0_LAER | VALID_COMPLETE | 100 | 0 | 0.863880 | 0.707264 |
| main | 17 | 42 | B0_SHUFFLED_LAER | VALID_COMPLETE | 100 | 0 | 0.855977 | 0.692345 |
| main | 17 | 43 | B0 | VALID_COMPLETE | 100 | 0 | 0.833378 | 0.636980 |
| main | 17 | 43 | B0_SPARSE | VALID_COMPLETE | 100 | 0 | 0.864915 | 0.706309 |
| main | 17 | 43 | B0_LAER | VALID_COMPLETE | 100 | 0 | 0.865709 | 0.707964 |
| main | 17 | 43 | B0_SHUFFLED_LAER | VALID_COMPLETE | 100 | 0 | 0.854916 | 0.684596 |
| main | 17 | 44 | B0 | VALID_COMPLETE | 100 | 0 | 0.844858 | 0.660235 |
| main | 17 | 44 | B0_SPARSE | VALID_COMPLETE | 100 | 0 | 0.872094 | 0.720017 |
| main | 17 | 44 | B0_LAER | VALID_COMPLETE | 100 | 0 | 0.871988 | 0.718188 |
| main | 17 | 44 | B0_SHUFFLED_LAER | VALID_COMPLETE | 100 | 0 | 0.862714 | 0.699548 |
| layer_robustness | 11 | 42 | B0 | VALID_COMPLETE | 100 | 0 | 0.783754 | 0.611612 |
| layer_robustness | 11 | 42 | B0_SPARSE | VALID_COMPLETE | 100 | 0 | 0.823700 | 0.684616 |
| layer_robustness | 11 | 42 | B0_LAER | RUNNING/PARTIAL | 93 | 0 | 0.833850 | 0.693039 |
| layer_robustness | 11 | 42 | B0_SHUFFLED_LAER | RUNNING/PARTIAL | 56 | 0 | 0.828830 | 0.671851 |
| layer_robustness | 19 | 42 | B0 | VALID_COMPLETE | 100 | 0 | 0.834651 | 0.637417 |
| layer_robustness | 19 | 42 | B0_SPARSE | VALID_COMPLETE | 100 | 0 | 0.859215 | 0.693778 |
| layer_robustness | 19 | 42 | B0_LAER | VALID_COMPLETE | 100 | 0 | 0.858724 | 0.688855 |
| layer_robustness | 19 | 42 | B0_SHUFFLED_LAER | VALID_COMPLETE | 100 | 0 | 0.849757 | 0.671373 |

## Main Layer17 Pooled Valid Results
| Method | Valid Runs | Mean Token Acc | Mean BLEU | Delta Acc vs B0 | Delta BLEU vs B0 |
|---|---:|---:|---:|---:|---:|
| B0 | 3 | 0.838174 | 0.649034 | +0.000000 | +0.000000 |
| B0_SPARSE | 3 | 0.868064 | 0.713764 | +0.029889 | +0.064730 |
| B0_LAER | 3 | 0.867192 | 0.711139 | +0.029018 | +0.062105 |
| B0_SHUFFLED_LAER | 3 | 0.857869 | 0.692163 | +0.019695 | +0.043129 |

## Interim Interpretation
- On layer17, B0_SPARSE is currently the strongest completed method by pooled Token Accuracy and BLEU.
- B0_LAER is also above B0 on layer17, but slightly below B0_SPARSE in pooled completed results.
- B0_SHUFFLED_LAER is above B0 but below B0_SPARSE and B0_LAER on layer17, which does not by itself prove an independent gain from shuffled attention structure.
- Layer11 still has two partial LAER-family runs; their current averages are provisional.

## Existing Design/Report Files
- `analysis/b0_laer_design.md`
- `analysis/b0_laer_report.md`
- `analysis/acdr_design.md`
- `analysis/acdr_report.md`
- `analysis/attention_validation_audit_design.md`
- `analysis/attention_validation_audit_report.md`
- `analysis/local_attention_relation_audit_design.md`
- `analysis/local_attention_relation_audit_report.md`
- `analysis/attention_recoverability_diagnostic.md`
- `analysis/rap_final_design.md`
- `analysis/rap_final_report.md`
- `runs/laer_final/PRE_REGISTRATION.json`
- `runs/laer_final/laer_experiment_manifest.json`
- `runs/laer_final/laer_provenance_audit.json`
- `runs/laer_final/laer_integrity_audit.json`
- `runs/laer_final/laer_smoke_audit.json`
