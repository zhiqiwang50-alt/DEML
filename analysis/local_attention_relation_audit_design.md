# Local Attention Relation Audit Design

This audit freezes B0-SPARSE as the only candidate generator. Activation/min-cut selects `x_sparse`; attention is evaluated afterward only as a diagnostic verifier.

- RNG policy: `set_seed(seed)` is called once through `acdr.load_model_and_data(cfg)` at run start; the prompt loop does not reset the RNG stream.
- Candidate generation: B0 Stage1, margin uncertainty, fixed-size sparse patch search, strict Top-1 candidate states.
- Candidate selection: `argmin L_cut_disc(x)` over activation-improving candidates. Attention does not generate, order, select, or update candidates.
- Changed set `C`: positions whose frozen candidate Top-1 ids differ from before ids; fixed/public keys are excluded.
- Local mass improvement: `delta_mass = E_mass_candidate - E_mass_before`, where each `E_mass` measures absolute mass error on changed keys inside a fixed causal query window.
- LAER: `delta_LAER = LAER_candidate - LAER_before`, averaged over server attention edges with `k in C`, `q > k`, and fixed/public keys excluded.
- Shuffled control: same before ids, candidate ids, changed positions, query rows, layer/head set, and candidate trajectory; only observed key identity is permuted per legal row.
- Ground truth is used only after all candidate and attention scores are computed, for fix/harm/gt_gain labels.
