# SACR-PIA Tune Selection Audit / 调参选择审计

This audit preserves all old tune/smoke results and only records the selector/reporting correction. 本审计保留全部旧结果，只记录选择器/报告逻辑修正。

- Old selector: required each tune init seed to be non-negative vs B0V. 原筛选器错误地要求每个 tune init_seed 都不低于 B0V。
- Corrected selector: uses pooled tune init_seed=42/43 results so selection is not based on one seed. 修正后按两个 tune init_seed 的 pooled 结果选参，避免只依据单个 init_seed。
- No model, loss, data split, K/Y, layer, epoch, or frozen candidate parameter was changed. 本次不修改模型、损失函数、数据划分、K/Y、layer、epoch 或冻结候选参数。

| method | lambda_tail | lambda_attn | rho_tail | rho_attn | layers | seed42 dAcc | seed43 dAcc | pooled dAcc | pooled dBLEU | old pass | fixed pass | reason |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|---|
| tail_consistency | 0.01 | 0.0 | 0.05 | 0.1 | all | -0.0360 | -0.0105 | -0.0233 | -0.0432 | False | False | pooled Token Accuracy delta vs B0V < 0; pooled BLEU delta vs B0V < 0 |
| tail_consistency | 0.01 | 0.0 | 0.1 | 0.1 | all | -0.0050 | 0.0104 | 0.0027 | 0.0030 | False | True | old selector rejected because one tune init_seed was below B0V |
| tail_consistency | 0.03 | 0.0 | 0.05 | 0.1 | all | -0.0385 | 0.0077 | -0.0154 | -0.0276 | False | False | pooled Token Accuracy delta vs B0V < 0; pooled BLEU delta vs B0V < 0 |
| tail_consistency | 0.03 | 0.0 | 0.1 | 0.1 | all | -0.0084 | -0.0804 | -0.0444 | -0.0655 | False | False | pooled Token Accuracy delta vs B0V < 0; pooled BLEU delta vs B0V < 0 |
| tail_consistency | 0.05 | 0.0 | 0.05 | 0.1 | all | -0.0196 | 0.0022 | -0.0087 | -0.0199 | False | False | pooled Token Accuracy delta vs B0V < 0; pooled BLEU delta vs B0V < 0 |
| tail_consistency | 0.05 | 0.0 | 0.1 | 0.1 | all | -0.0025 | -0.0050 | -0.0037 | -0.0063 | False | False | pooled Token Accuracy delta vs B0V < 0; pooled BLEU delta vs B0V < 0 |
| attention_consistency | 0.0 | 0.05 | 0.1 | 0.1 | all | -0.0536 | NA | -0.0536 | -0.0915 | False | False | stopped at smoke because Token Accuracy delta vs B0V was below -0.03; seed43 tune not run |
| tail_attention_consistency | 0.05 | 0.05 | 0.1 | 0.1 | all | -0.0086 | NA | -0.0086 | -0.0218 | False | False | not tuned because C3 requires combining selected C1 and C2 configs, and C2 was stopped at smoke |
