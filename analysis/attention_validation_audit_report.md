# Attention Validation Audit Report

## Candidate Label Summary

| Label | Count | true delta_JS mean | true median | shuffled mean | shuffled median | mean GT gain | mass delta mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| positive | 118 | -0.02605368383228779 | -0.014851432293653488 | -0.00012894782979609604 | -0.00024653971195220947 | 1.2881355932203389 | 0.033426206936112916 |
| neutral | 94 | -0.02380430706320925 | -0.01677525043487549 | -0.00011188616143896225 | 0.00021091103553771973 | 0.0 | 0.05073802909910975 |
| harmful | 27 | -0.018687706026766036 | -0.01383865624666214 | -0.000823780342384621 | -0.0006143748760223389 | -1.2962962962962963 | 0.0715114587160461 |

## Overall

- frozen candidate count: 239
- candidate hash agreement: 1.0
- before hash agreement: 1.0
- selected rows same true/shuffled: 1.0
- P(harmful | delta_JS > 0): 0.16129032258064516
- P(harmful | delta_JS <= 0): 0.10576923076923077
- Spearman(-delta_JS_true, gt_gain): 0.061731810137967885
- Spearman(-delta_JS_shuffled, gt_gain): -0.0518672922538772
- AUC true: 0.5023812858943829
- AUC shuffled: 0.5063034038380726

## Gate Replay

| Gate | Accepted | Positive retained | Harmful retained | Harmful rejected | Beneficial rejected | Precision | Harm rejection | Beneficial retention | Mean GT gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CUT_ONLY | 239 | 118 | 27 | 0 | 0 | 0.49372384937238495 | 0.0 | 1.0 | 0.4895397489539749 |
| TRUE_ATTN_GATE | 208 | 106 | 22 | 5 | 12 | 0.5096153846153846 | 0.18518518518518517 | 0.8983050847457628 | 0.5240384615384616 |
| SHUFFLED_ATTN_GATE | 125 | 64 | 17 | 10 | 54 | 0.512 | 0.37037037037037035 | 0.5423728813559322 | 0.456 |

## Audit Answers

1. Candidate source: all 239 candidates are frozen B0-SPARSE cut-improving candidates. Attention is not used for candidate generation, candidate pool construction, ordering, or sparse candidate selection.
2. Hash agreement: TRUE and SHUFFLED replay use the same before ids, candidate ids, selected query rows, and candidate count. Candidate hash agreement = 1.0, before hash agreement = 1.0, selected-row agreement = 1.0.
3. Ground truth usage: ground truth token ids are used only for final token accuracy/BLEU and for offline fix/harm/gt_gain labels after candidate and attention scores are computed. They do not enter the attack path or candidate selection.
4. Positive vs harmful separation: positive candidates have a slightly more negative true delta_JS mean than harmful candidates (-0.0261 vs -0.0187), but the separation is weak.
5. Rank correlation: Spearman(-delta_JS_true, gt_gain) = 0.0617, which is close to zero. This means true attention consistency is not a reliable monotonic predictor of whether a frozen sparse candidate improves reconstruction.
6. Shuffled control: Spearman(-delta_JS_shuffled, gt_gain) = -0.0519 and AUC = 0.5063. The true attention AUC is 0.5024, so true attention does not clearly outperform shuffled attention as a verifier.
7. Harm probability: P(harmful | delta_JS > 0) = 0.1613 and P(harmful | delta_JS <= 0) = 0.1058. This is not strong evidence that the raw sign of delta_JS cleanly separates harmful candidates.
8. Gate replay: TRUE_ATTN_GATE rejects 5 harmful candidates but also rejects 12 beneficial candidates. Its precision rises only from 0.4937 to 0.5096, and the shuffled gate has similar precision (0.5120) while discarding many more beneficial candidates.
9. Attention mass diagnostic: patch attention mass delta is not constant zero. Harmful candidates have the largest mean mass delta (0.0715), but this diagnostic alone is not sufficient to form a reliable verifier.
10. Decision: this dev audit does not provide enough evidence to formalize a B0+Sparse+Attention Validation method. The result supports freezing the current attention-verifier hypothesis rather than expanding to heldout, Skytrax-150, or multilayer experiments.

## Final Judgment

The attention verifier hypothesis is only weakly supported by this diagnostic. TRUE attention gives a small positive gate-replay change over CUT_ONLY, but it does not clearly beat the shuffled control and its correlation with candidate ground-truth gain is near zero.

Conclusion: stop this attention-verifier route for now. The current evidence is useful as a negative/diagnostic result, but it should not be claimed as a stable improvement mechanism.
