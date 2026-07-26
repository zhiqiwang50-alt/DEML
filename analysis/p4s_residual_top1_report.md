# P4S-Res Residual Attention Scaling Report

## Method

P4S-Res keeps the original strict Top-1 P4S attack path and changes only the attention scaling strength. Given the scheduled P4S alpha, it uses residual scaling:

alpha_res_i = 1 + rho * (alpha_i - 1)

The loss remains the attention-scaled activation matching loss:

L = mean_i || alpha_res_i * h_i(x_tilde) - h_i_target ||^2

rho=0 recovers uniform variable-token scaling; rho=1 recovers the current P4S scaling. Fixed public positions and BOS remain excluded from alpha statistics and loss weighting.

## Frozen Settings

- White-box attack, full model parameters plus H_obs.
- Dataset: Skytrax-150, dataset_len=150.
- TinyLlama-1.1B, participant_number=4, attacker_position=4.
- Top-1 only: K=1, Y=1.
- P4S inherited settings: beta=0.25, weight_power=0.5, alpha_min=0.5, alpha_max=1.5, attention schedule 50%-70%, rollout depth=all.
- Residual candidates on seed42: global rho=0.5 and layer-adaptive rho.
- Frozen layer-adaptive rho for heldout: layer11=1.0, layer17=0.6, layer19=0.4.

## Seed42 Candidate Selection

| variant | mean dAcc vs B0V | mean dBLEU vs B0V | mean dAcc vs P4S | mean dBLEU vs P4S |
|---|---:|---:|---:|---:|
| global_rho0p5 | 0.003157 | 0.005704 | -0.004323 | -0.004033 |
| layer_adaptive | 0.010120 | 0.011632 | 0.002640 | 0.001895 |

Layer-adaptive was selected for heldout because it was positive vs both B0V and P4S on seed42. This was frozen before seed43/44; no rho retuning was done on heldout seeds.

## Frozen Layer-Adaptive Results

| seed | layer | rho | Acc | BLEU | dAcc vs B0V | dAcc vs P4S | dBLEU vs B0V | dBLEU vs P4S |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 11 | 1.00 | 0.892051 | 0.818555 | 0.031664 | 0.000458 | 0.035768 | 0.000227 |
| 42 | 17 | 0.60 | 0.929023 | 0.847002 | -0.002192 | 0.003677 | -0.001472 | 0.000876 |
| 42 | 19 | 0.40 | 0.929432 | 0.845591 | 0.000888 | 0.003784 | 0.000600 | 0.004582 |
| 43 | 11 | 1.00 | 0.904316 | 0.834278 | 0.016221 | -0.000358 | 0.016975 | 0.000109 |
| 43 | 17 | 0.60 | 0.922256 | 0.841181 | 0.002426 | -0.005428 | 0.001255 | -0.008688 |
| 43 | 19 | 0.40 | 0.929839 | 0.851420 | -0.002582 | -0.002307 | -0.003422 | -0.001083 |
| 44 | 11 | 1.00 | 0.901999 | 0.829146 | 0.053930 | -0.010788 | 0.058383 | -0.008309 |
| 44 | 17 | 0.60 | 0.930209 | 0.849505 | 0.011490 | 0.006242 | 0.007806 | 0.004307 |
| 44 | 19 | 0.40 | 0.928616 | 0.846891 | -0.001168 | -0.002492 | -0.002223 | -0.003442 |

## Pooled Interpretation

- Across seed42/43/44 and layers 11/17/19, P4S-Res layer-adaptive mean Acc is 0.918638 and BLEU is 0.840397.
- Mean delta vs B0V: Acc 0.012298, BLEU 0.012630.
- Mean delta vs original P4S: Acc -0.000802, BLEU -0.001269.
- Heldout seed43/44 alone stays positive vs B0V but is negative vs P4S on average.
- Therefore P4S-Res should not replace P4S as the best current method. It is useful as a diagnostic/protective residual variant, but not as evidence of a stronger main method.

## Audit

- Completed runs: 9 P4S-Res cells, each 150/0 completed/failed.
- No NaN/failure/max-alpha/fixed-public alpha/BOS alpha audit violations were found in the generated outputs.
- The attack API still does not receive prompt, original_ids, or original_tokens.
