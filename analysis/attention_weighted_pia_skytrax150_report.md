# Skytrax-150 Attention Weighted PIA Validation

This report freezes B0VAG after the Skytrax-28 development stage and validates it on Skytrax-150 with TinyLlama, seed42, Top-K=10, layers 11/17/19, epoch100. B0 remains the paper-style original PIA reproduction and B0V remains the strong variable-only baseline constraint.

## Frozen Configuration

- Candidate: B0VAG = variable-only activation matching + server-side last-window attention weighting + uncertainty gate.
- Attention settings: beta=0.25, weight_power=0.5, weight_min=0.5, weight_max=2.0, last_window_size=16, rollout_depth=all, residual_rollout=on.
- Schedule: attention weighting enabled after 70% of epochs.
- Gate: embedding-margin uncertainty gate, uncertainty_fraction=0.3.
- Dataset: data/skytrax_150.json, dataset_len=150, seed=42.

## Main Results

| layer | method | Acc | BLEU | dAcc vs B0 | dBLEU vs B0 | dAcc vs B0V | dBLEU vs B0V | done/fail |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 11 | B0 | 0.948995 | 0.934548 | 0.000000 | 0.000000 | 0.012305 | 0.019677 | 150/0 |
| 11 | B0V | 0.936689 | 0.914871 | -0.012305 | -0.019677 | 0.000000 | 0.000000 | 150/0 |
| 11 | B0A | 0.937927 | 0.920772 | -0.011068 | -0.013776 | 0.001237 | 0.005901 | 150/0 |
| 11 | B0VAG | 0.938231 | 0.916781 | -0.010764 | -0.017767 | 0.001541 | 0.001911 | 150/0 |
| 17 | B0 | 0.993061 | 0.987884 | 0.000000 | 0.000000 | -0.003081 | -0.004161 | 150/0 |
| 17 | B0V | 0.996142 | 0.992045 | 0.003081 | 0.004161 | 0.000000 | 0.000000 | 150/0 |
| 17 | B0A | 0.991912 | 0.986127 | -0.001150 | -0.001757 | -0.004231 | -0.005918 | 150/0 |
| 17 | B0VAG | 0.991761 | 0.985291 | -0.001300 | -0.002593 | -0.004381 | -0.006754 | 150/0 |
| 19 | B0 | 0.994753 | 0.989965 | 0.000000 | 0.000000 | -0.001359 | -0.002884 | 150/0 |
| 19 | B0V | 0.996112 | 0.992849 | 0.001359 | 0.002884 | 0.000000 | 0.000000 | 150/0 |
| 19 | B0A | 0.996482 | 0.993012 | 0.001729 | 0.003047 | 0.000370 | 0.000163 | 150/0 |
| 19 | B0VAG | 0.994565 | 0.989764 | -0.000188 | -0.000201 | -0.001547 | -0.003085 | 150/0 |

## Decision

- B0VAG exceeds B0 in Token Accuracy on 0/3 layers and exceeds B0V on 1/3 layers.
- B0VAG has non-negative Token Accuracy delta vs B0 on 0/3 layers and vs B0V on 1/3 layers.
- Therefore the frozen B0VAG candidate does not reproduce a robust improvement on Skytrax-150. It should be reported as a negative validation result, not as a stable improvement.
- Direct formula variant B0A improves over B0 on 1/3 layers and over B0V on 2/3 layers; its gain is concentrated at layer19 and is not stable across layers.

## Run Audit

- Completed samples across all method-layer runs: 1800.
- Failed samples across all method-layer runs: 0.
- NaN metric count in summary columns: 0.
- Output root: runs/attention_weighted_pia_skytrax150/
- Summary files: runs/attention_weighted_pia_skytrax150/reports/stage1_delta_summary.csv and stage1_summary.md.
