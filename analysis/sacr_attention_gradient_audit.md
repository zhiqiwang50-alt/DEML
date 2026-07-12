# SACR Attention Gradient Audit

结论：A. attention loss 梯度链路正确，但方法性能为负。

- attention_used_rows: 372
- attention_skipped_rows: 0
- L_attn: 0.1118006780743599
- ||grad_z(L_attn)||: 0.3193514049053192
- 关闭 attention loss 后的差异: 0.3193514049053192
- 微小扰动后 L_attn 是否变化: True
- fixed public / invalid positions 是否排除: True

本审计没有重跑 C2/C3 大规模实验，也没有修改 SACR 方法。
