# ACDR Design

ACDR is a B0-based white-box prompt inversion branch. It keeps the original B0 continuous reconstruction as the first stage and tests whether server-side attention probabilities can act as a discrete consistency verifier after B0.

## Frozen Method

1. Stage1 is B0: all-valid uniform activation matching plus vocabulary prior. No attention enters initialization or gradient optimization.
2. Uncertain positions are selected by B0 nearest-neighbor margin on variable positions only. Attention is not used for localization.
3. Sparse patch search updates only uncertain positions and optimizes L_patch = L_cut_all_valid + lambda_vocab * L_vocab.
4. Candidate states are strict Top-1 discrete token states saved whenever patch Top-1 ids change. No vocab Top-K candidates are used.
5. ACDR captures observed server-side attention probabilities A_obs from H_obs and candidate probabilities A_x from the same server suffix. It compares attention probabilities only with JS divergence.
6. Attention rows are purified by masking fixed/public key positions, preserving causal legality, renormalizing legal keys, removing public-sink-dominated rows, and retaining the middle 80% entropy rows.
7. Query rows are determined by a causal fixed window after the patch, not by attention.
8. Candidate selection is activation-filter first, then attention-consistency reranking. Ground truth is used only after the attack for metrics.

## Stop Rule

ACDR should proceed to heldout only if it exceeds B0, B0_SPARSE, and B0_SHUFFLED_ATTN and satisfies the candidate-pool agreement audit.
