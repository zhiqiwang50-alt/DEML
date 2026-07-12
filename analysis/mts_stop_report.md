# MTS-PIA Stop Report

Stopped after held-out seed 43. Results are kept on disk.

Conclusion: development-set gains could not be safely treated as reproduced on the held-out validation path. This is not evidence that MTS-PIA is effective.

- stop_rule: A
- reason: P2 and P3 both have negative mean token accuracy delta, negative mean BLEU delta, and win_count <= loss_count.
- P2: acc_delta_vs_B0V=-0.0153, BLEU_delta_vs_B0V=-0.0265, win/tie/loss=1/19/8
- P3: acc_delta_vs_B0V=-0.0076, BLEU_delta_vs_B0V=-0.0096, win/tie/loss=1/22/5
