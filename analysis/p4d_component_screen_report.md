# P4D Component Screen Report

## Scope

This report summarizes the follow-up component screen for the strict Top-1 branch after P4D.
The experiment keeps the same white-box attack setting and does not change the original PIA
baseline definition.

Fixed setting:

- Dataset: Skytrax-150
- Model: TinyLlama-1.1B-Chat-v1.0
- Participant number / attacker position: 4 / 4
- Layers: 11, 17, 19 for full validation; 17, 19 for the first component screen
- Seeds: 42 for the first component screen; 42, 43, 44 for full validation
- Epoch: 100
- Top-K / Top-Y: 1 / 1
- Attention source: server-side mean-query rollout
- Attention schedule: start ratio 0.5, full ratio 0.7
- Alpha range: [0.5, 1.5]
- Beta: 0.25

## Component Definitions

P4D uses attention to weight the residual vector before activation matching:

```text
L_P4D = mean_i || alpha_i * (h_i(X_tilde) - h_i_target) ||_2^2
```

Because alpha is multiplied before the square, the effective token weight is alpha_i^2.

Three conservative variants were added:

### P4DL: Linear Attention Residual

```text
L_P4DL = mean_i || sqrt(alpha_i) * (h_i(X_tilde) - h_i_target) ||_2^2
       = mean_i alpha_i * ||h_i(X_tilde) - h_i_target||_2^2
```

This is closest to the weighted activation-matching formula and avoids the quadratic
amplification in P4D.

### P4DR: Residualized Attention Strength

```text
alpha_eff = 1 + rho * (alpha - 1), rho = 0.5
L_P4DR = mean_i || alpha_eff_i * (h_i(X_tilde) - h_i_target) ||_2^2
```

This keeps the P4D loss form but weakens the deviation from uniform weighting.

### P4DG: Uncertainty-Gated Attention Residual

```text
alpha_eff_i = alpha_i if token position i is in the uncertainty set else 1
L_P4DG = mean_i || alpha_eff_i * (h_i(X_tilde) - h_i_target) ||_2^2
```

This only applies attention weights to the most uncertain token positions.

## Component Screen

Command:

```bash
python runs/p4d_component_screen/launch_runner.py
```

Result path:

- `runs/p4d_component_screen/summary.csv`
- `runs/p4d_component_screen/summary.md`

Screen result on seed42, layers 17 and 19:

| Method | n | Mean Acc | Mean BLEU | Mean dAcc vs B0V | Mean dBLEU vs B0V |
|---|---:|---:|---:|---:|---:|
| B0 | 2 | 0.928205 | 0.847470 | -0.001674 | 0.000737 |
| B0V | 2 | 0.929879 | 0.846733 | 0.000000 | 0.000000 |
| P4S | 2 | 0.925497 | 0.843568 | -0.004382 | -0.003165 |
| P4D | 2 | 0.925676 | 0.840868 | -0.004203 | -0.005865 |
| P4DL | 2 | 0.932135 | 0.851783 | 0.002256 | 0.005050 |
| P4DR | 2 | 0.929450 | 0.847430 | -0.000429 | 0.000697 |
| P4DG | 2 | 0.927699 | 0.844912 | -0.002180 | -0.001821 |

Screen conclusion: P4DL is the only component that is positive on both Token Accuracy and BLEU
relative to B0V in the pooled screen. It was therefore expanded to the full 3-seed, 3-layer
validation.

## P4DL Full Validation

Command:

```bash
python runs/p4dl_linear_residual_pia_top1_skytrax150/launch_runner.py
```

Result path:

- `runs/p4dl_linear_residual_pia_top1_skytrax150/summary.csv`
- `runs/p4dl_linear_residual_pia_top1_skytrax150/summary.md`

