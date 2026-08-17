# AIR-PTR Development Decision

Experiment: Skytrax-28, seed42, target layer 17, epoch 100, strict Top-1.

Corrected result root: `runs/air_ptr_fixed/`.

## Decision

Do not enter held-out Skytrax-150 seed45.

## Reason

- FINAL Token Accuracy delta vs B0V: -0.008351
- FINAL BLEU delta vs B0V: -0.013921
- FINAL paired win/tie/loss vs B0V: 10 / 3 / 15
- LAR Token Accuracy delta vs B0V: 0.000158
- LAR BLEU delta vs B0V: -0.000019
- LAR paired win/tie/loss vs B0V: 7 / 6 / 15
- LAR_PTR equals LAR in accuracy and BLEU; PTR proposed 223 repairs and accepted 0.

The development gate required a clear FINAL improvement over B0V, BLEU improvement, more paired wins than losses, no failure/NaN/audit violation, and non-negative PTR behavior. The corrected development run does not satisfy those conditions.

AIR-PTR can be described as a corrected negative or inconclusive development result relative to B0V. It should not be reported as a stable improvement.
