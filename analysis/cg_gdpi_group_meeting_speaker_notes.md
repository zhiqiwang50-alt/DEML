# 组会讲稿：面向协同推理 Prompt Inversion 的注意力初始化与置信度门控梯度重排序方法探索

> 说明：本讲稿对应 `analysis/cg_gdpi_group_meeting.pptx`，基于 `runs/cg_gdpi/` 当前实验结果。

## 第 1 页：标题与汇报主线

这一页先明确汇报定位：这是 TinyLlama 白盒 pilot，不是论文大模型完整复现。三点结论要谨慎：中间 activation 有泄露风险；旧全局梯度融合失败；CG-GDPI 相比 seed42 baseline 有提升，但没有超过 attention_context_existing，而且门控没有打开。

## 第 2 页：研究背景

协同推理会把模型按层切分，中间 activation 在参与方之间传递。PIA 的问题是：攻击者能否只凭 activation、公开模型和词表 embedding，恢复原始输入。这个问题关系到协同推理的隐私边界。

## 第 3 页：Baseline 方法

原始 DEML/PIA baseline 可以概括为两步：先在连续 embedding 空间中匹配 activation，再通过 activation calibration 把连续向量离散成 token。这个 baseline 在当前 pilot 中已经很强。

## 第 4 页：实验进度

这里强调当前是 TinyLlama-adapted pilot。主比较已经完成 seed42、epoch100 的六个方法，每个 28 条；还有一个额外 baseline seed43。不要把它表述为完整多 seed 结论。

## 第 5 页：创新动机

方法动机有两条：第一，利用 self-attention 的上下文结构；第二，不再做不受约束的全局梯度融合，因为它会把噪声方向放大。本轮 B4 的结果非常差，是这个动机的实验证据。

## 第 6 页：AIGM-PIA 方法

AIGM 的关键是先用 dummy embedding 拟合 activation，再从模型内部拿到目标层 attention，构造 A_proxy。通过 H_obs 和 A_proxy @ H_obs 的残差融合做初始化。但本轮 B3 单独使用并不稳定。

## 第 7 页：CG-GDPI 方法

CG-GDPI 的核心是把 calibration 放在第一优先级，梯度只做保守重排序。这样设计的目的不是让梯度无条件覆盖校准，而是在梯度方向稳定时利用它；不稳定时不伤害结果。

## 第 8 页：反泄漏验证

这一页说明实验不是通过泄漏真实 token 得到的结果。攻击函数只接收 activation、sequence length、公开模型和词表等信息。prefix/full activation 完全一致，max diff 为 0。注意 gate open rate 为 0，是当前方法不足。

## 第 9 页：主实验对比表

这一页是核心结果。CG-GDPI 相比 seed42 baseline 从 0.9847 提升到 0.9971，但 attention_context_existing 是 0.9979，略高于 CG-GDPI。因此不能声称 CG-GDPI 稳定超过所有 baseline。B4 global gradmatch 只有 0.2524，说明旧梯度融合失败。

## 第 10 页：结果分析

本轮最重要的分析是区分“门控保护”与“梯度有效”。CG-GDPI 没有像 B4 一样崩掉，但 gate open 和 override 都是 0，所以它主要依赖 calibration，而不是梯度重排序。

## 第 11 页：结论与不足

结论要保守：当前证据支持中间 activation 泄露风险，也说明 calibration 很强，CG-GDPI 的门控设计避免了 B4 的灾难性下降。但不足是 gate 没开，实验规模也还只是 pilot。

## 第 12 页：下一步计划

下一步主要围绕如何让梯度稳定参与。可以先放宽 tau_consensus，再做 soft gate 和 position-wise 阈值。最终需要看 gate open 后的 override accuracy，而不是只看总 accuracy。

## 备用说明

- 当前结果应表述为 TinyLlama 白盒 pilot，不是论文 Llama-65B/70B 的完整复现。
- CG-GDPI 相比 seed42 baseline 有提升：0.9971 vs 0.9847。
- CG-GDPI 没有超过 attention_context_existing：0.9971 vs 0.9979。
- 当前 gate open rate = 0、override rate = 0，因此不能把提升归因于梯度重排序已经发挥作用。
- 旧全局梯度匹配 B4 明显失败：token accuracy = 0.2524。
