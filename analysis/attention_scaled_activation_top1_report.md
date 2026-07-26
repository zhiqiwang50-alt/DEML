# Attention-Scaled Activation PIA Top-1 Report

## Scope

This report covers the Top-1 development experiment only: Skytrax-28, seed42, layer17, epoch100, K=1, Y=1. B0 is the primary paper baseline; B0V is the strong variable-only constraint.

## Main Results

| method | run | Acc | BLEU | dAcc vs B0 | dAcc vs B0V | dBLEU vs B0 | dBLEU vs B0V | done/fail |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| original_pia_baseline | original_pia_baseline_seed42_layer17_epoch100_k1 | 0.821239 | 0.625943 | 0.000000 | -0.024809 | 0.000000 | -0.035554 | 28/0 |
| variable_only_uniform | variable_only_uniform_seed42_layer17_epoch100_k1 | 0.846048 | 0.661497 | 0.024809 | 0.000000 | 0.035554 | 0.000000 | 28/0 |
| mts_mean_query | mts_mean_query_seed42_layer17_epoch100_k1_beta0p75_power0p5_wmin0p25_wmax4_depthlast2 | 0.848843 | 0.664476 | 0.027604 | 0.002795 | 0.038534 | 0.002980 | 28/0 |
| mts_last_window | mts_last_window_seed42_layer17_epoch100_k1_beta0p25_power0p5_wmin0p25_wmax2_depthall | 0.842291 | 0.647577 | 0.021052 | -0.003757 | 0.021634 | -0.013920 | 28/0 |
| attn_scale_mean_query | attn_scale_mean_query_seed42_layer17_epoch100_k1_beta0p25_power0p5_amin0p5_amax1p5_start0p7 | 0.853776 | 0.671241 | 0.032537 | 0.007729 | 0.045298 | 0.009744 | 28/0 |
| attn_scale_last_window | attn_scale_last_window_seed42_layer17_epoch100_k1_beta0p1_power0p5_amin0p5_amax1p5_start0p7 | 0.837684 | 0.647206 | 0.016445 | -0.008364 | 0.021264 | -0.014290 | 28/0 |
| attn_scale_last_window_gate | attn_scale_last_window_gate_seed42_layer17_epoch100_k1_beta0p25_power0p5_amin0p5_amax2_start0p7 | 0.852918 | 0.673235 | 0.031679 | 0.006870 | 0.047293 | 0.011739 | 28/0 |

## Best Candidate

- best by Token Accuracy among attention-scale rows: `attn_scale_mean_query_seed42_layer17_epoch100_k1_beta0p25_power0p5_amin0p5_amax1p5_start0p7`
- method: `attn_scale_mean_query`
- beta=0.25, alpha_min=0.5, alpha_max=1.5, attention_start_ratio=0.7
- Acc=0.853776, BLEU=0.671241
- vs B0: Acc 0.032537, BLEU 0.045298
- vs B0V: Acc 0.007729, BLEU 0.009744

The selected P4 candidate and the same P4 setting with `alpha_max=2.0` are numerically tied in this grid because the observed max alpha stays below 1.5. The stricter `alpha_max=1.5` row is kept as the milder candidate.

## Audit

- completed/failed for best candidate: 28/0
- has_nan: False
- mean alpha: 1.000000
- max alpha: 1.369781
- fixed public alpha sum max: 0.000000
- final BOS alpha max: 0.000000
- raw BOS mass mean: 0.800015
- final last-valid alpha mean: 0.919276

## Current Conclusion

On this Top-1 development set, P4 improves over both B0 and B0V. P6 also has a row above B0/B0V, but P4 is the best by Token Accuracy. This should be treated as a candidate-selection result only, not as stable evidence.

## Suggested Next Checks

- hold out seeds 43/44 with frozen P4 Top-1 candidate;
- run the planned Top-K=10 main setting after Top-1 validation;
- check layer 11/17/19 before any Skytrax-150 expansion.

## P4S-Res Residual Scaling Follow-up

Residual scaling was tested as alpha_res_i = 1 + rho * (alpha_i - 1) under the same strict Top-1 P4S setting. Seed42 compared global rho=0.5 with layer-adaptive rho; layer-adaptive was frozen as layer11=1.0, layer17=0.6, layer19=0.4 before seed43/44.

Across seed42/43/44 and layers 11/17/19, frozen layer-adaptive P4S-Res has mean dAcc vs B0V = 0.012298 and mean dBLEU vs B0V = 0.012630, but mean dAcc vs P4S = -0.000802 and mean dBLEU vs P4S = -0.001269. Thus it remains positive relative to B0V but does not beat the current P4S branch on heldout seeds. Do not present it as a stronger replacement for P4S. Detailed outputs are in 
uns/p4s_residual_top1/.
