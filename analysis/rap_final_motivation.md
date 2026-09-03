# RAP-FINAL Motivation

RAP-v2 seed42/43/44 are now treated as development/diagnostic results, not final held-out evidence, because these seeds have already been inspected while methods were being modified.

## RAP-v2 Pooled Facts

- RAP_V2 vs B0V: Acc +0.001733, BLEU +0.000283, W/T/L 22/18/16.
- RAP_V2 vs B0: Acc +0.010545, BLEU +0.014577, W/T/L 32/6/18.
- RAP_V2 vs SPARSE_PATCH: Acc +0.000518, BLEU +0.000333, W/T/L 20/13/23.
- fix/harm/net_fix vs B0V: 129/90/+39.

## Interpretation

The sparse repair component appears valuable, but the relational contribution is weak and seed-dependent. Seed44 did not stably reproduce the B0V improvement. Therefore, RAP-v2 cannot be used to claim stable relational-attention improvement.

RAP-FINAL is designed to separate uncertainty-aware sparse repair from relational routing. It removes relational min-cut partitioning, uses fixed patches, adds layer-wise relational normalization, and validates accepted repairs on the discrete Top-1 path.

