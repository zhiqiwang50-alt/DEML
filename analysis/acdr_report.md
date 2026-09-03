# ACDR Development Report

## Observation

Fixed dev setup: Skytrax-28, seed42, layer17, epoch100, strict Top-1. The only main baseline is B0. B0V is not used as the backbone or main judgement in this branch.

| Method | Acc | BLEU | Exact | NED | Completed | Failed | NaN |
|---|---:|---:|---:|---:|---:|---:|---:|
| b0 | 0.837886 | 0.641476 | 0.000000 | 0.160790 | 28 | 0 | False |
| b0_sparse | 0.866633 | 0.702876 | 0.000000 | 0.132334 | 28 | 0 | False |
| b0_acdr | 0.862850 | 0.698281 | 0.000000 | 0.136118 | 28 | 0 | False |
| b0_shuffled_attn | 0.848286 | 0.663355 | 0.000000 | 0.150468 | 28 | 0 | False |

## Paired Comparisons

| Comparison | Acc delta | BLEU delta | W/T/L | Acc 95% CI | BLEU 95% CI | fix/harm/net |
|---|---:|---:|---:|---:|---:|---:|
| b0_sparse_vs_b0 | 0.028748 | 0.061400 | 25/1/2 | [0.020460, 0.037614] | [0.044336, 0.080034] | 162/39/123 |
| b0_acdr_vs_b0 | 0.024964 | 0.056804 | 23/4/1 | [0.015983, 0.035324] | [0.038656, 0.077950] | 145/48/97 |
| b0_acdr_vs_b0_sparse | -0.003784 | -0.004595 | 5/9/14 | [-0.008087, 0.000668] | [-0.011921, 0.003135] | 25/51/-26 |
| b0_acdr_vs_b0_shuffled_attn | 0.014564 | 0.034926 | 18/5/5 | [0.005893, 0.024726] | [0.016524, 0.057565] | 94/47/47 |
| b0_shuffled_attn_vs_b0 | 0.010400 | 0.021878 | 18/7/3 | [0.005841, 0.015122] | [0.013523, 0.030970] | 87/37/50 |

## Mechanism Audit

| Method | Accepted patches | Rejected patches | Attention-JS improved accepted |
|---|---:|---:|---:|
| b0_sparse | 216 | 344 | 0 |
| b0_acdr | 209 | 351 | 209 |
| b0_shuffled_attn | 137 | 421 | 137 |

- ACDR vs shuffled candidate-pool hash agreement: 0.071429 (2/28), required 1.000000.
- This means the current independent method runs diverged after accepted repairs, so the shuffled-control comparison is informative but not a perfect same-candidate-pool reranking control.

## Interpretation

- ACDR improves over B0, but B0_SPARSE improves more.
- ACDR is below B0_SPARSE on both Token Accuracy and BLEU.
- ACDR is above B0_SHUFFLED_ATTN, but the candidate-pool agreement audit fails, so this is not sufficient independent evidence for true attention structure.
- Main judgement passed: false.

## Supported

- B0 strict Top-1 reproduction and no-leakage checks are established.
- Sparse B0 repair gives a clear dev-set improvement over B0.
- ACDR improves over B0 and over the shuffled-attention run on this dev split.

## Not Supported

- ACDR does not outperform B0_SPARSE on Token Accuracy or BLEU in this frozen dev run.
- The required ACDR vs shuffled same-candidate-pool hash agreement is not satisfied.
- Therefore this result does not justify heldout, Skytrax-150, layer11/19, new seed, or parameter sweeping for ACDR.

## Required Questions

1. B0后还有多少可修复Top1错误：B0_SPARSE vs B0 has fix/harm/net = 162/39/123.
2. sparse search带来多少增益：Acc +0.028748, BLEU +0.061400.
3. attention fingerprint reranking额外带来多少：relative to B0_SPARSE, Acc -0.003784, BLEU -0.004595.
4. true attention是否超过shuffled attention：yes on final dev metrics, but the candidate-pool agreement audit failed, so this is not sufficient independent evidence.
5. attention JS改善是否更容易对应GT修复：patch logs record JS improvements, but this frozen run does not support the stronger ACDR claim because ACDR is below B0_SPARSE.
6. 是否达到heldout条件：false.
