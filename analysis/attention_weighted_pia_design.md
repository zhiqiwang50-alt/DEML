# Attention Weighted PIA Design

本实验遵循原论文 white-box 设置：攻击者知道完整模型参数，并获得切分层 activation `H_obs`。在原 PIA activation matching 基础上，用 `H_obs` 经过 server suffix 得到的 last-window attention rollout 构造位置权重。

第一阶段固定设置：Skytrax-28、seed42、layer 11/17/19、epoch100、Top-K=10 主实验、Top-1 消融、last-window=16、rollout depth=all、beta=0.25、weight_min=0.5、weight_max=2.0、70% epoch 后启用 attention。B0 是论文主基线，B0V 是强基线约束。

方法：B0=原论文 all-valid baseline；B0V=variable-only 强基线；B0A=B0+attention；B0VA=B0V+attention；B0VAG=B0V+attention+embedding-margin gate。
