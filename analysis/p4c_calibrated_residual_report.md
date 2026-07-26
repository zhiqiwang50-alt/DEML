# P4C Calibration-Aware Residual Weighting Report

## Method

P4C keeps the same residual-matching loss family as P4D. It does not use the P4S-style
`alpha * h_pred` scaling.

```text
L_P4C = mean_i || alpha_i^C * (h_i(X_tilde) - h_i_target) ||_2^2
```

The change is only in how the attention weight `alpha_i^C` is constructed:

```text
alpha_i^C = 1 + rho * (alpha_i - 1), only for uncertainty-gated token positions
alpha_i^C = 1, for other variable token positions
```

Then the active weights are bounded and mean-normalized. In this run:

- `rho = 0.5`
- `uncertainty_fraction = 0.3`
- `alpha_min = 0.5`
- `alpha_max = 1.5`
- attention schedule: start 50%, full 70%
- Top-K / Top-Y: 1 / 1

This means P4C tests whether a calibrated and gated version of the correct P4D residual
formula can improve strict Top-1 reconstruction.

## Verification

Unit tests:

```bash
python -m unittest tests.test_masked_server_attn_mts
```

Result: 32 tests passed.

P4C smoke audit:

- `method = attn_calibrated_weighted_residual_schedule`
- `method_role = attention_weighted_residual`
- `calibrated_rho = 0.5`
- `gate_open_rate = 0.3`
- `mean_alpha = 1.0`
- `fixed_public_alpha_sum = 0.0`
- `final_bos_alpha = 0.0`

## Full Experiment

Command:

```bash
python runs/p4c_calibrated_residual_top1_skytrax150/launch_runner.py
```

Fixed settings:

- Dataset: Skytrax-150
- Seeds: 42, 43, 44
- Layers: 11, 17, 19
- Epoch: 100
- Top-K / Top-Y: 1 / 1
- Baselines reused from existing completed runs

Result files:

- `runs/p4c_calibrated_residual_top1_skytrax150/summary.csv`
- `runs/p4c_calibrated_residual_top1_skytrax150/summary.md`
- `runs/p4c_calibrated_residual_top1_skytrax150/runner_complete.json`

## Results

| Method | n | Mean Acc | Mean BLEU | dAcc vs B0V | dBLEU vs B0V |
|---|---:|---:|---:|---:|---:|
| B0 | 9 | 0.906114 | 0.827937 | -0.000226 | 0.000170 |
| B0V | 9 | 0.906340 | 0.827767 | 0.000000 | 0.000000 |
| P4S | 9 | 0.919439 | 0.841666 | 0.013099 | 0.013899 |
| P4D | 9 | 0.914471 | 0.834990 | 0.008131 | 0.007224 |
| P4DL | 9 | 0.911761 | 0.832775 | 0.005421 | 0.005008 |
| P4C | 9 | 0.907271 | 0.828943 | 0.000931 | 0.001177 |

P4C relative to P4D:

- Token Accuracy: -0.007200
- BLEU: -0.006047

P4C relative to P4S:

- Token Accuracy: -0.012169
- BLEU: -0.012722

P4C is positive against B0V in 5/9 seed-layer configurations for Token Accuracy and 5/9 for BLEU.

## Conclusion

P4C confirms that the implementation can preserve the correct P4D residual formula while adding
calibration-aware attention weighting. However, this specific gated and residualized alpha design
does not improve over P4D or P4S. It only gives a very small pooled gain over B0V.

This should be treated as a negative or weak-positive component result, not as the final best
method. The current best empirical branch remains P4S, while P4D remains stronger than P4C among
the residual-formula methods.
