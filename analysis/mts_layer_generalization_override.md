# MTS-PIA Layer Generalization Override Summary

This file summarizes the exploratory layer11/layer19 runs launched after the user explicitly overrode the held-out stop rule. The method and parameters were not retuned: only frozen P3 was compared with B0 and B0V on Skytrax-28 seed42.

| layer | label | method | Acc | BLEU | dAcc vs B0 | dBLEU vs B0 | dAcc vs B0V | dBLEU vs B0V | completed/failed | audit fixed sum max | max weight |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | B0 | original_pia_baseline | 0.9546 | 0.9292 | 0.0000 | 0.0000 | 0.0270 | 0.0289 | 28/0 | 1.0000 | 1.0000 |
| 11 | B0V | variable_only_uniform | 0.9277 | 0.9003 | -0.0270 | -0.0289 | 0.0000 | 0.0000 | 28/0 | 0.0000 | 1.0000 |
| 11 | P3 | mts_last_window | 0.9772 | 0.9672 | 0.0226 | 0.0379 | 0.0495 | 0.0669 | 28/0 | 0.0000 | 1.2500 |
| 19 | B0 | original_pia_baseline | 0.9960 | 0.9917 | 0.0000 | 0.0000 | -0.0030 | -0.0058 | 28/0 | 1.0000 | 1.0000 |
| 19 | B0V | variable_only_uniform | 0.9990 | 0.9974 | 0.0030 | 0.0058 | 0.0000 | 0.0000 | 28/0 | 0.0000 | 1.0000 |
| 19 | P3 | mts_last_window | 0.9944 | 0.9881 | -0.0016 | -0.0035 | -0.0046 | -0.0093 | 28/0 | 0.0000 | 1.2500 |

- Layer 11 P3 paired token accuracy vs B0V: delta=0.0495, win/tie/loss=8/14/6.
- Layer 11 P3 paired BLEU vs B0V: delta=0.0669, win/tie/loss=8/14/6.
- Layer 19 P3 paired token accuracy vs B0V: delta=-0.0046, win/tie/loss=3/17/8.
- Layer 19 P3 paired BLEU vs B0V: delta=-0.0093, win/tie/loss=3/17/8.

Interpretation: layer11 is positive for frozen P3 relative to both B0 and B0V, but layer19 is negative. This is not cross-seed or cross-layer evidence of a stable improvement; it is an exploratory result after an explicit stop-rule override.
