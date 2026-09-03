# Attention-Recoverability Correlation Diagnostic

## Scope

This diagnostic is evaluation-only. It uses completed strict Top-1 B0V runs and does not modify B0V, rerun prompt inversion optimization, tune thresholds, or add attention-weighting components.

Ground truth token ids are used only after the attack has completed, to compute correctness and evaluation residuals.

Input result directories:
- `runs/attention_weighted_pia/stage1/B0V_layer11_top1_seed42_epoch100`
- `runs/attention_weighted_pia/stage1/B0V_layer17_top1_seed42_epoch100`
- `runs/attention_weighted_pia/stage1/B0V_layer19_top1_seed42_epoch100`

## Main Answer

- Spearman(attention importance, token correctness): -0.044371
- Spearman(attention importance, activation residual): 0.137748
- Spearman(attention importance, embedding NN distance proxy): 0.048206

server-side attention importance is not a reliable predictor of prompt inversion recoverability.

## Quartile Accuracy

| Quartile | Token Accuracy | Residual | NN distance proxy |
| --- | ---: | ---: | ---: |
| Q1 | 0.789660 | 0.0219684 | 0.148599 |
| Q2 | 0.811521 | 0.0356825 | 0.129036 |
| Q3 | 0.816248 | 0.0504493 | 0.125415 |
| Q4 | 0.736407 | 0.0482896 | 0.193586 |

## Layer-Wise Correlation

| Layer | Token Count | Spearman correct | Token Accuracy | Residual |
| --- | ---: | ---: | ---: | ---: |
| 11 | 4513 | -0.109326 | 0.706847 | 0.0336422 |
| 17 | 4513 | -0.036803 | 0.820075 | 0.0237597 |
| 19 | 4513 | -0.007717 | 0.838467 | 0.0598884 |

## Seed-Wise Correlation

| Seed | Token Count | Spearman correct | Token Accuracy |
| --- | ---: | ---: | ---: |
| 42 | 13539 | -0.044371 | 0.788463 |

## Counterfactual Attention Objective

| Variant | Mean objective | Delta vs uniform |
| --- | ---: | ---: |
| Uniform | 0.04270999 | 0.00000000 |
| Original Attention | 0.04354756 | 0.00083757 |
| Shuffled Attention | 0.04244515 | -0.00026484 |
| Inverted Attention | 0.04200875 | -0.00070125 |

## Required Questions

1. Positive correlation with recovery accuracy: no/weak.
2. High-attention tokens are easier to recover only if Q4 accuracy exceeds Q1-Q3 consistently; see quartile table.
3. High-attention tokens have lower activation residual only if Q4 residual is lower; observed Q4 residual is 0.0482896.
4. Shuffled attention is diagnostic-only; compare its objective against original attention above.
5. Inverted attention is diagnostic-only; compare its objective against original/uniform above.
6. Cross-layer stability: not established.
7. Cross-seed stability: not testable beyond completed seed(s).

## Notes

- `embedding_nn_distance` is an evaluation proxy `||E[recovered_token] - E[original_token]||` because historical B0V outputs did not save the continuous optimized embedding `z*`.
- Fixed public framing positions are excluded using the same public-mask semantics as the attack path. Token ids are not passed to the attack path for special-token masking.
