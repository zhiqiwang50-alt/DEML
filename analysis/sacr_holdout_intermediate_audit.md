# SACR-PIA Holdout Intermediate Audit

All rows compare frozen C1 against B0V with the same holdout prompt ids and order for each init seed.

| init_seed | B0V Acc | B0V BLEU | C1 Acc | C1 BLEU | dAcc | dBLEU | win/tie/loss | done/fail B0V | done/fail C1 | mean lambda_eff | cap violations | leakage |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---|
| 101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0/4/0 | 4/0 | 4/0 | 0.0084 | 0 | True |
| 102 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0/4/0 | 4/0 | 4/0 | 0.0086 | 0 | True |
| 103 | 0.9971 | 0.9925 | 1.0000 | 1.0000 | 0.0029 | 0.0075 | 1/3/0 | 4/0 | 4/0 | 0.0082 | 0 | True |
