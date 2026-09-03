# B0-LAER Design

B0-LAER is a controlled end-to-end experiment built on the formal B0/B0-SPARSE path. B0 remains the main baseline.

## Pipeline

1. Stage1 exactly reuses B0 all-valid uniform activation matching with strict Top-1 decoding (`K=1`, `Y=0`, semantic speculation off, calibration off).
2. B0-SPARSE generates candidates using margin uncertainty, fixed sparse patches, sparse continuous search, and strict Top-1 discrete candidates.
3. The sparse candidate is frozen as `x_sparse = argmin L_cut_disc(candidate)` over activation-improving candidates. Attention is not used for generation, ordering, or selection.
4. For changed Top-1 positions `C`, compute `LAER(x; C)=mean |A_x[l,h,q,k]-A_obs[l,h,q,k]|` over legal server attention edges with `k in C`, `q > k`, and fixed/public keys excluded.
5. B0-LAER accepts a frozen sparse candidate iff `cut_candidate < cut_before - eps_cut` and `laer_candidate < laer_before`. Otherwise the patch rolls back.
6. B0-SHUFFLED-LAER uses the same candidate generator and gate budget, but shuffles observed key identity inside each legal row before LAER scoring.

Ground truth is used only after attack completion for evaluation and patch-level mechanism labels.
