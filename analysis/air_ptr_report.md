# AIR-PTR Experimental Report

## Historical Facts

Historical gradient fusion/hard reranking, MTS held-out validation, P4C gating, and several direct residual-weighting variants did not establish a stable improvement over B0V. These branches are not rerun in AIR-PTR. The no-gradient-matching `alpha_nn_init_residual` pilot used `gamma=0.3`, which is frozen here.

## Current Development Results

Completed on Skytrax-28, seed42, target layer 17, epoch 100, strict Top-1 (`K=1`, `Y=0`, semantic speculation disabled). During audit, the initial rho ablation was found invalid because the reused `stage_b_optimize` branch did not apply `residual_alpha_rho` to `attn_linear_weighted_residual_schedule`. A regression test was added and the method name was added to the residualization branch. The invalid outputs under `runs/air_ptr/` were retained; the corrected rerun is under `runs/air_ptr_fixed/`.

Corrected rho selection:

| Layer | Selected rho | Token Accuracy | BLEU |
| --- | ---: | ---: | ---: |
| 11 | 0.75 | 0.822770 | 0.651357 |
| 17 | 0.50 | 0.846206 | 0.661478 |
| 19 | 0.75 | 0.848558 | 0.663665 |

Six-method development comparison at layer17:

| Method | Token Accuracy | BLEU | Delta Acc vs B0 | Delta BLEU vs B0 | Delta Acc vs B0V | Delta BLEU vs B0V |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.821239 | 0.625943 | 0.000000 | 0.000000 | -0.024809 | -0.035554 |
| B0V | 0.846048 | 0.661497 | 0.024809 | 0.035554 | 0.000000 | 0.000000 |
| ASINIT | 0.803204 | 0.618280 | -0.018035 | -0.007663 | -0.042844 | -0.043217 |
| LAR | 0.846206 | 0.661478 | 0.024967 | 0.035535 | 0.000158 | -0.000019 |
| LAR_PTR | 0.846206 | 0.661478 | 0.024967 | 0.035535 | 0.000158 | -0.000019 |
| FINAL | 0.837696 | 0.647576 | 0.016458 | 0.021633 | -0.008351 | -0.013921 |

Paired analysis vs B0V:

| Method | Mean Acc Delta | Mean BLEU Delta | Win/Tie/Loss |
| --- | ---: | ---: | ---: |
| LAR | 0.000158 | -0.000019 | 7 / 6 / 15 |
| LAR_PTR | 0.000158 | -0.000019 | 7 / 6 / 15 |
| FINAL | -0.008351 | -0.013921 | 10 / 3 / 15 |

PTR proposed 223 token repairs for LAR_PTR and FINAL, accepted 0, fixed 0, harmed 0. Therefore PTR was neutral rather than beneficial in this development run.

The development gate was not passed. FINAL does not exceed B0V, and LAR/LAR_PTR only show a negligible Token Accuracy delta with a slightly negative BLEU delta and more paired losses than wins.

## Held-Out Results

Not run. Skytrax-150 seed45 was blocked because the development threshold was not satisfied. Seeds 46/47 and additional expansions remain blocked.

## Interpretation

AIR-PTR improves over the paper-style B0 baseline on Skytrax-28 seed42, but the stronger B0V control already explains most of that gain. The corrected attention residual weighting does not provide convincing evidence of improvement over B0V in this setting. No stable-improvement claim should be made from these results.
