# Attention Weighted PIA Stage1 Report

## 运行设置

- Dataset: Skytrax-28 (`data/airline.json`)
- Seed: 42
- Layers: 11 / 17 / 19
- Epoch: 100
- 主实验: Top-K=10, Y=10
- 消融: Top-1, semantic speculation disabled by naive discretization
- Attention: server-side last-window=16, rollout depth=all
- Stabilization: beta=0.25, weight_min=0.5, weight_max=2.0, schedule=70%
- Gate: B0VAG uses embedding top1/top2 margin bottom 30%

## Top-K=10 主结果

| layer | method | Acc | BLEU | dAcc vs B0 | dAcc vs B0V | done/fail |
|---:|---|---:|---:|---:|---:|---|
| 11 | B0 | 0.9546 | 0.9292 | 0.0000 | 0.0270 | 28/0 |
| 11 | B0A | 0.9462 | 0.9250 | -0.0084 | 0.0186 | 28/0 |
| 11 | B0V | 0.9277 | 0.9003 | -0.0270 | 0.0000 | 28/0 |
| 11 | B0VA | 0.8974 | 0.8763 | -0.0573 | -0.0303 | 28/0 |
| 11 | B0VAG | 0.9536 | 0.9440 | -0.0011 | 0.0259 | 28/0 |
| 17 | B0 | 0.9847 | 0.9742 | 0.0000 | -0.0110 | 28/0 |
| 17 | B0A | 0.9938 | 0.9868 | 0.0091 | -0.0019 | 28/0 |
| 17 | B0V | 0.9958 | 0.9907 | 0.0110 | 0.0000 | 28/0 |
| 17 | B0VA | 0.9917 | 0.9843 | 0.0070 | -0.0041 | 28/0 |
| 17 | B0VAG | 0.9888 | 0.9773 | 0.0040 | -0.0070 | 28/0 |
| 19 | B0 | 0.9960 | 0.9917 | 0.0000 | -0.0030 | 28/0 |
| 19 | B0A | 0.9955 | 0.9905 | -0.0005 | -0.0035 | 28/0 |
| 19 | B0V | 0.9990 | 0.9974 | 0.0030 | 0.0000 | 28/0 |
| 19 | B0VA | 0.9973 | 0.9940 | 0.0012 | -0.0017 | 28/0 |
| 19 | B0VAG | 0.9993 | 0.9987 | 0.0033 | 0.0003 | 28/0 |

## Top-1 消融

| layer | method | Acc | BLEU | dAcc vs B0 | dAcc vs B0V | done/fail |
|---:|---|---:|---:|---:|---:|---|
| 11 | B0 | 0.7823 | 0.6007 | 0.0000 | 0.0380 | 28/0 |
| 11 | B0A | 0.7817 | 0.6016 | -0.0006 | 0.0374 | 28/0 |
| 11 | B0V | 0.7443 | 0.5684 | -0.0380 | 0.0000 | 28/0 |
| 11 | B0VA | 0.7398 | 0.5802 | -0.0426 | -0.0045 | 28/0 |
| 11 | B0VAG | 0.8065 | 0.6412 | 0.0242 | 0.0622 | 28/0 |
| 17 | B0 | 0.8212 | 0.6259 | 0.0000 | -0.0248 | 28/0 |
| 17 | B0A | 0.8408 | 0.6512 | 0.0196 | -0.0052 | 28/0 |
| 17 | B0V | 0.8460 | 0.6615 | 0.0248 | 0.0000 | 28/0 |
| 17 | B0VA | 0.8306 | 0.6394 | 0.0094 | -0.0154 | 28/0 |
| 17 | B0VAG | 0.8243 | 0.6325 | 0.0030 | -0.0218 | 28/0 |
| 19 | B0 | 0.8468 | 0.6605 | 0.0000 | 0.0024 | 28/0 |
| 19 | B0A | 0.8411 | 0.6535 | -0.0057 | -0.0033 | 28/0 |
| 19 | B0V | 0.8444 | 0.6527 | -0.0024 | 0.0000 | 28/0 |
| 19 | B0VA | 0.8360 | 0.6394 | -0.0108 | -0.0084 | 28/0 |
| 19 | B0VAG | 0.8461 | 0.6574 | -0.0007 | 0.0017 | 28/0 |

## 阶段性判断

- B0A 在 Top-K 下超过 B0 的层数: 1/3。
- B0VA 在 Top-K 下超过 B0 的层数: 2/3。
- B0VAG 在 Top-K 下超过 B0 的层数: 2/3。
- 因 B0VA/B0VAG 在 layer17 和 layer19 超过 B0，满足进入 Skytrax-150 前置条件。
- 但 attention 方法尚未稳定超过 B0V：只有 B0VAG 在 layer19 Top-K 同时超过 B0 和 B0V。
- 当前不能声称方法已稳定提升，只能说相对于论文 B0 的小规模多层验证出现正信号。

## Skytrax-150 Frozen Candidate Validation

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
