# MTS-PIA Held-Out Seed 43 Decision

This held-out seed was not used for parameter selection.

- stop: True
- stop_rule: A
- P2: acc_delta_vs_B0V=-0.0153, BLEU_delta_vs_B0V=-0.0265, win/tie/loss=1/19/8, failed=0, fixed_sum=0.0000, max_weight=1.8062
- P3: acc_delta_vs_B0V=-0.0076, BLEU_delta_vs_B0V=-0.0096, win/tie/loss=1/22/5, failed=0, fixed_sum=0.0000, max_weight=1.2500
- reason: P2 and P3 both have negative mean token accuracy delta, negative mean BLEU delta, and win_count <= loss_count.
