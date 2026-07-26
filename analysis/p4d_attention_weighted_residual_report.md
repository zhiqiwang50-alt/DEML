# P4D Attention-Weighted Residual Top-1 Report

## Method

P4D keeps the frozen Top-1 P4S setting but changes the continuous activation objective from scaling the predicted activation to scaling the activation residual:

```text
L_P4D = mean_{i in variable positions} || alpha_i * (h_pred_i - h_obs_i) ||_2^2
      = mean_i alpha_i^2 * || h_pred_i - h_obs_i ||_2^2
```

Compared with P4S:

```text
L_P4S = mean_i || alpha_i * h_pred_i - h_obs_i ||_2^2
```

P4D is closer to the intended interpretation of assigning attention-derived importance to the mismatch between predicted and observed activations. The optimum remains `h_pred = h_obs`; attention only changes the optimization emphasis.

## Fixed Setting

- Attack setting: white-box, full model parameters known, no prompt/original token ids passed to attack API.
- Dataset: Skytrax-150.
- Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0.
- Seeds: 42, 43, 44.
- Layers: 11, 17, 19.
- Epoch: 100.
- Top-K/Y: 1/1 strict Top-1 path.
- Attention source: server-side attention rollout, mean-query.
- Rollout: residual on, depth all.
- Alpha: beta=0.25, alpha_min=0.5, alpha_max=1.5.
- Schedule: attention loss starts at 50% epoch and reaches full strength at 70%.
- Baselines reused from the completed Top-1 Skytrax-150 runs.

## Pooled Result

| Method | Mean Acc | Mean BLEU | Mean dAcc vs B0V | Mean dBLEU vs B0V |
|---|---:|---:|---:|---:|
| B0 | 0.906114 | 0.827937 | -0.000226 | 0.000170 |
| B0V | 0.906340 | 0.827767 | 0.000000 | 0.000000 |
| P4S | 0.919439 | 0.841666 | 0.013099 | 0.013899 |
| P4D | 0.914471 | 0.834990 | 0.008131 | 0.007224 |

P4D is positive on the pooled Top-1 Skytrax-150 comparison, but it does not exceed P4S.

## Layer-Level Pattern

| Layer | P4D Acc | dAcc vs B0 | dAcc vs B0V | Note |
|---:|---:|---:|---:|---|
| 11 | 0.891657 | +0.015661 | +0.026140 | Strongest layer for P4D. |
| 17 | 0.922269 | +0.010146 | -0.000985 | Better than B0, slightly below B0V. |
| 19 | 0.929486 | -0.000736 | -0.000763 | Essentially flat/slightly negative. |

## Conclusion

P4D validates that attention-weighting the activation residual is implementable and yields a pooled improvement over B0/B0V under strict Top-1, but the gain is smaller and less stable than P4S. For the current project narrative, P4S remains the better-performing branch; P4D is useful as an ablation showing that the exact residual-weighted formula is not the source of the best observed improvement.

Result directory: `runs/attention_weighted_residual_pia_top1_skytrax150/`
