# Server-Attention Consistency PIA Report

MTS direct server-attention token weighting is frozen as a negative/inconclusive path. SACR-PIA instead keeps B0V as the main objective and uses server-side tail/attention consistency as capped auxiliary regularization.

- output root: `runs/server_attention_consistency_pia`
- no automatic multi-layer or Skytrax-150 expansion is allowed unless holdout is promising.

## Tune Candidates

| method | pooled dAcc vs B0V | pooled dBLEU vs B0V | lambda_tail | lambda_attn | layers |
|---|---:|---:|---:|---:|---|
| tail_consistency | 0.0027 | 0.0030 | 0.01 | 0.0 | all |

## Holdout

| method | pooled dAcc | pooled dBLEU | win/tie/loss | stop | promising |
|---|---:|---:|---|---|---|
| tail_consistency | 0.0010 | 0.0025 | 1/11/0 | False | False |

Do not claim stable improvement unless the holdout promising criteria are met.

## 候选筛选规则修正

- 原筛选器错误地要求每个 tune init_seed 都不低于 B0V。
- 本实验的正确约束是不能仅依据单个 init_seed 选择，而应以 tune init_seed=42/43 的 pooled 结果选参。
- 本次是报告/选择器逻辑修正，不修改模型、损失函数、数据划分、K/Y、layer、epoch 或候选参数。
- 旧结果没有删除，仍保留在 `tune_ablation.csv`；修正前/后的选择状态写入 `tune_selection_audit.csv` 和 `analysis/tune_selection_audit.md`。

| method | lambda_tail | lambda_attn | rho_tail | rho_attn | pooled dAcc | pooled dBLEU | old pass | fixed pass | selected |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| tail_consistency | 0.01 | 0.0 | 0.1 | 0.1 | 0.0027 | 0.0030 | False | True | True |
| attention_consistency | 0.0 | 0.05 | 0.1 | 0.1 | -0.0536 | -0.0915 | False | False | False |
| tail_attention_consistency | 0.05 | 0.05 | 0.1 | 0.1 | -0.0086 | -0.0218 | False | False | False |

修正后冻结的 holdout 候选仍为：C1 / tail_consistency，`lambda_tail=0.01`，`rho_tail=0.10`，`lambda_attn=0`，`server_attention_layers=all`。

## Holdout Intermediate Audit

Per-seed B0V/C1 paired results are saved to `runs/server_attention_consistency_pia/reports/holdout_seed_audit.csv` and `analysis/sacr_holdout_intermediate_audit.md`.
