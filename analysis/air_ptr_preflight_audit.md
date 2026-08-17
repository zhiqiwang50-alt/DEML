# AIR-PTR Preflight Audit

## Repository State

- Branch: `attn-scaled-activation-matching`
- Audited commit: `318a382 feat: evaluate attention-scaled activation matching`
- Existing dirty files are preserved. AIR-PTR is implemented in new files and does not overwrite historical result directories.
- GPU state at audit time (2026-08-15 11:50 CST): all four RTX 4090D devices had active external processes. Formal runs must re-check free memory and use at most one worker per GPU.

## Strict Top-1 Definition

AIR-PTR requires `K=1`, `Y=0`, and `semantic_speculation=false`. The embedding nearest-neighbor projection produces exactly one token per position. There is no semantic candidate, top-M reranking, beam search, or LLM oracle. B0, B0V, and every AIR-PTR component are rerun under this same definition.

## Existing Baselines

- B0 / `original_pia_baseline`: all-valid activation matching plus the existing vocabulary constraint, followed by strict Top-1 projection. Its activation loss is not changed.
- B0V / `variable_only_uniform`: uniform activation matching only over recoverable variable positions plus the existing vocabulary constraint. Fixed/public positions do not enter the activation loss. B0V is the primary strong baseline.

## Existing Dummy-Attention Initialization

`pia_alpha_gm.py` provides `stage_a_dummy_proxy(...)` and `alpha_nn_initialization(...)`. Stage A optimizes dummy embeddings using only `H_obs`, public model/tokenizer state, sequence length, and public framing. It extracts `A_self` from the dummy forward; this is a dummy-attention proxy, not the real prompt attention.

The residual initialization is:

```text
H_init = (1 - gamma) H_obs + gamma (A_self @ H_obs)
z0 = E[NN(H_init)]
```

The historical no-gradient-matching pilot used `gamma=0.3` for `alpha_nn_init_residual` and obtained the best two-sample result among the alpha initialization variants. AIR-PTR freezes `gamma=0.3`; the `gamma=0.25` fallback is not needed.

## Existing Residual Matching and Schedule

The existing P4D loss is:

```text
mean_i ||alpha_i (h_pred_i - H_obs_i)||^2
= mean_i alpha_i^2 ||h_pred_i - H_obs_i||^2
```

AIR-PTR does not reuse this squared-weight form. It reuses the correct linear residual helper `attention_linear_weighted_residual_loss(...)`:

```text
mean_i w_i ||h_pred_i - H_obs_i||^2
```

The P4S late schedule is retained and frozen: no attention contribution in the first 50% of epochs, linear ramp from 50% to 70%, and full contribution from 70% to 100%.

## Reusable Functions

- `capture_prefix_activation`
- `embedding_candidates`, `naive_discretization`, `nearest_embedding_loss`
- `activation_calibrated_discretization` (strict Top-1 gives one embedding candidate)
- `build_variable_mask`, `variable_mask_audit_payload`
- `server_forward_with_attention`, `server_attention_bundle`, `rollout_from_attentions`
- `stage_a_dummy_proxy`, `alpha_nn_initialization`
- `random_public_embeddings`, `fixed_embedding_tensor`, `enforce_embedding_constraints`
- `stage_b_optimize` with the existing B0, B0V, and linear residual modes

## Historical Negative Components

- Gradient-Direction Token Matching / broad gradient fusion degraded the historical AIGM pilot as gradient weight increased.
- CA-GDM hard gradient reranking selected wrong override tokens in the recorded diagnostic runs.
- MTS direct token-weighted activation loss did not reproduce its development gain on held-out seeds.
- P4C hard uncertainty gating and P4CAR did not provide a stable improvement over B0V.
- P4D applies an effective `alpha^2` weight and is not the intended AIR-PTR formula.
- P4DL/P4LWR linear residual weighting was layer-sensitive and did not establish cross-layer stability.
- SACR and semantic speculation are outside the AIR-PTR path.

AIR-PTR must not reintroduce hard gradient reranking, top-M candidates, P4C gating, MTS direct weighting, SACR, P4DL/P4CAR as named experimental branches, semantic candidates, or any ground-truth-dependent attack decision.

## Threat-Model Audit Rules

The attack entry point may receive `H_obs`, sequence length, model/tokenizer, public parameters and framing, server attention, dummy state, and candidate recovered embeddings. It must not receive or read prompt text, `original_ids`, `original_tokens`, ground-truth ids/embeddings, or reference tokens.

Position 0 and other `fixed_public` positions are public framing positions. Without token ids in the attack API, the code cannot claim token-id-based EOS detection. The audit uses `last_valid_position`; it does not rename the final valid position as EOS.
