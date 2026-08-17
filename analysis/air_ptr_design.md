# AIR-PTR Design

## Scope

AIR-PTR is a white-box TinyLlama prompt-inversion branch. The attacker knows the public model parameters, tokenizer, sequence length, public framing, observed boundary activation `H_obs`, and server-side attention. The attack path never receives prompt text, original token ids, original tokens, ground-truth embeddings, or reference tokens.

Every method uses strict Top-1: `K=1`, `Y=0`, and semantic speculation disabled. Nearest-neighbor projection emits one token per position and no secondary candidate list is created.

## Methods

- B0: original all-valid activation matching plus vocabulary constraint.
- B0V: variable-only uniform activation matching plus vocabulary constraint.
- ASINIT: dummy `A_self` initialization followed by B0V optimization.
- LAR: random/public initialization followed by layer-adaptive residual matching.
- LAR_PTR: LAR plus projection-time repair.
- FINAL: ASINIT plus LAR plus projection-time repair.

## Dummy Attention Initialization

Stage A optimizes dummy embeddings against `H_obs` and extracts a detached dummy-attention proxy `A_self`. It is not the real prompt attention.

```text
D* = argmin_D ||F_prefix(D) - H_obs||^2
H_init = (1 - gamma) H_obs + gamma (A_self @ H_obs)
z0 = E[NN(H_init)]
```

The historical no-gradient-matching pilot freezes `gamma=0.3`.

## Layer-Adaptive Residual Matching

For variable positions `V`:

```text
e_i = h_pred_i - H_obs_i
L_LAR = mean_{i in V} w_i(t,l) ||e_i||^2 + lambda_vocab L_vocab
w_i(t,l) = 1 + s(t) rho_l (r_i - 1)
```

`r_i` is server rollout importance using mean-query, power 0.5, residual rollout on, and all server layers. It is normalized to mean one over variable positions. Fixed/public positions have final weight zero. Weights are bounded to `[0.5, 1.5]` and remain nonnegative.

The schedule is frozen: `s(t)=0` through 50% of optimization, rises linearly to 1 at 70%, then stays at 1. Only `rho_l` is selected on Skytrax-28 seed42 from `{0, 0.25, 0.5, 0.75, 1}`. `rho=0` exactly recovers B0V activation weighting.

The corrected AIR-PTR run uses `runs/air_ptr_fixed/`. Earlier `runs/air_ptr/` rho outputs are retained but should not be used for method selection: an audit found that `residual_alpha_rho` was not applied to `attn_linear_weighted_residual_schedule` in the reused stage-B branch. The fix is covered by `tests.test_air_ptr.LarWeightTests.test_linear_residual_schedule_applies_residual_alpha_rho`.

`layer_adaptive_stats.json` records the base server-attention alpha before step-wise rho shrinkage. It is useful for checking public fixed-position masking and base attention concentration, but it should not be interpreted as the final per-step rho-shrunk loss weight.

## Projection-Time Repair

After continuous optimization, strict Top-1 gives `x0=NN(z*)`. AIR-PTR evaluates the resulting discrete prompt against `H_obs`, computes the B0V activation gradient with respect to its discrete embeddings, and selects high-residual variable positions.

For each selected position, one proposal is generated:

```text
g_i = d L_B0V / d Z_i
z_proposal_i = E[x_i] - eta g_i / (||g_i|| + eps)
x'_i = NN(z_proposal_i)
```

The proposal is accepted only when recomputing the discrete prefix gives `L_candidate < L_current - 1e-6`. Ground truth is not used. Defaults are `eta=0.10`, repair fraction 0.20, at most eight positions per pass, and at most two passes. A position receives at most one NN proposal. A pass with no accepted proposal stops repair.

## Audit and Stopping

Verification covers attack API leakage, manual prefix equivalence, server forward equivalence, dummy-only `A_self`, strict Top-1, `rho=0` equivalence, `eta=0` no-op, and strict activation-loss decrease for accepted repair. Any verification failure blocks formal experiments.

Development success is judged primarily against B0V. FINAL must improve Token Accuracy by at least 0.020, improve BLEU, have more paired wins than losses, contain no failure/NaN/audit violation, and have `PTR_fix >= PTR_harm` before expansion. Dev results are not described as stable or significant.
