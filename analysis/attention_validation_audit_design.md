# Attention Validation Audit Design

This audit freezes B0-SPARSE as the candidate generator. Activation/min-cut selects `x_sparse`; attention is evaluated only afterward as a verifier diagnostic.

- Candidate generation: B0 Stage1, margin uncertainty, fixed-size sparse patch search, strict Top-1 candidate states.
- Candidate selection: `argmin L_cut_disc(x)` over cut-improving candidates.
- Attention scoring: JS divergence between purified server-side attention probabilities from `H_obs`, before state, and frozen candidate.
- Shuffled control: same before ids, candidate ids, and selected query rows; only observed attention identity is shuffled.
- Ground truth is used only after all candidate and attention scores are computed, for fix/harm/gt_gain labels.
