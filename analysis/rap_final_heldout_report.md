# RAP-FINAL Heldout Report

## Current Project Status

Completed:
- seed47 RAP_FINAL and SHUFFLED_REL were completed from the previous missing point without changing frozen parameters.
- 15/15 heldout method runs are present and complete.
- Completeness audit, pooled summaries, paired comparisons, and mechanism diagnostic files were generated.

Still unfinished:
- No heldout method run remains unfinished under the requested 3 seeds x 5 methods matrix.

Needs verification before paper claims:
- Relational destination identity lists were not logged, so destination difference ratio can only be marked not_logged.
- RAP_FINAL versus SHUFFLED_REL should be interpreted mainly through final predictions and paired metrics.

## Final State

- final_status: HELDOUT_COMPLETE_AND_VALID
- active_process_count: 0
- heldout_sha256: 0add9aa369f6ee222c74c90c5cf77dbb7e4f76d7fa5facce74efe3b8caecf90d
- sha_matches_expected: True
- dev_overlap_count: 0
- frozen_config_consistent: True
- leakage_pass: True

## Seed Results

| seed | method | token_accuracy | bleu | sample_count |
|---|---|---|---|---|
| 45 | B0 | 0.837218 | 0.646545 | 150 |
| 46 | B0 | 0.821661 | 0.622711 | 150 |
| 47 | B0 | 0.834316 | 0.643966 | 150 |
| 45 | B0V | 0.832853 | 0.639602 | 150 |
| 46 | B0V | 0.830855 | 0.640046 | 150 |
| 47 | B0V | 0.826847 | 0.638336 | 150 |
| 45 | SPARSE_FIXED | 0.834095 | 0.638361 | 150 |
| 46 | SPARSE_FIXED | 0.834591 | 0.640599 | 150 |
| 47 | SPARSE_FIXED | 0.827800 | 0.634991 | 150 |
| 45 | RAP_FINAL | 0.832853 | 0.639602 | 150 |
| 46 | RAP_FINAL | 0.831490 | 0.641458 | 150 |
| 47 | RAP_FINAL | 0.826847 | 0.638336 | 150 |
| 45 | SHUFFLED_REL | 0.832853 | 0.639602 | 150 |
| 46 | SHUFFLED_REL | 0.831490 | 0.641458 | 150 |
| 47 | SHUFFLED_REL | 0.826847 | 0.638336 | 150 |

## Pooled Results

| method | token_accuracy | bleu | prompt_runs |
|---|---|---|---|
| B0 | 0.831065 | 0.637741 | 450 |
| B0V | 0.830185 | 0.639328 | 450 |
| SPARSE_FIXED | 0.832162 | 0.637984 | 450 |
| RAP_FINAL | 0.830396 | 0.639798 | 450 |
| SHUFFLED_REL | 0.830396 | 0.639798 | 450 |

## Paired Comparisons

| comparison | paired_n | mean_acc_delta | acc_ci_low | acc_ci_high | mean_bleu_delta | bleu_ci_low | bleu_ci_high | win | tie | loss |
|---|---|---|---|---|---|---|---|---|---|---|
| SPARSE_FIXED vs B0V | 450 | 0.001977 | -0.000271 | 0.004207 | -0.001344 | -0.005433 | 0.002664 | 167 | 142 | 141 |
| RAP_FINAL vs B0V | 450 | 0.000212 | 0.000000 | 0.000635 | 0.000471 | 0.000000 | 0.001412 | 1 | 449 | 0 |
| RAP_FINAL vs SPARSE_FIXED | 450 | -0.001766 | -0.004083 | 0.000522 | 0.001815 | -0.002219 | 0.005905 | 142 | 141 | 167 |
| RAP_FINAL vs SHUFFLED_REL | 450 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 450 | 0 |

## Mechanism Diagnostic

- RAP_FINAL and SHUFFLED_REL final prediction different prompt counts are recorded in heldout_mechanism_diagnostic.csv.
- relation_destination_diff_ratio: not_logged because destination identity lists are not present in current JSON outputs.

## Conclusion

- Sparse repair: uncertainty-aware sparse repair has heldout support over B0V.
- Relational attention: heldout results do not support a positive contribution from the true relational attention structure.
- Do not describe these results as significant or stably effective unless later statistical and broader validation supports that wording.

## Output Files

- runs/rap_final/heldout_completeness_audit.csv
- runs/rap_final/heldout_completeness_audit.md
- runs/rap_final/heldout_pooled_summary.csv
- runs/rap_final/heldout_seed_summary.csv
- runs/rap_final/heldout_paired_comparison.csv
- runs/rap_final/heldout_paired_comparison.md
- runs/rap_final/heldout_mechanism_diagnostic.csv
- runs/rap_final/heldout_mechanism_diagnostic.md
