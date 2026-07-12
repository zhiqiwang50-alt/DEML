# SAER-PIA Report

## 当前状态

- 已新建 SAER 独立分支文件与纯函数测试。
- 已创建 Skytrax-150 split：prompt_split_seed=20260706，tune=50，holdout=100。
- E0/E1 机制 smoke 完成；E2 在真实 suffix-attention edge 候选评分阶段过慢，未完成第一条样本；E3 未启动。
- 因阶段 0 未完整通过，未进入 tune 或 holdout。
- 不能声称 attention edge routing 有效。

## Smoke 结果

| method | status | completed | Acc | BLEU | override | correct/incorrect | graph empty | mean edges |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | complete | 3 | 0.6027799139821698 | 0.3914214105927989 | 0 | 0/0 | 0 | 35806.333333333336 |
| E1 | complete | 3 | 0.6412885584892762 | 0.4712321591282127 | 7 | 1/6 | 0 | 35806.333333333336 |
| E2 | incomplete_timeout_before_first_prediction |  |  |  |  | / |  |  |

E2 did not complete the first prompt within the local 20 minute command window even after delaying auxiliary scoring until after the calibration-margin gate. E3 was not started. Per the mechanism-smoke rule, tune and holdout were not run.
