# Masked Server Attention PIA Design

This document records the current MTS-PIA branch implemented in `pia_masked_server_attn_pia.py`.

## Method Boundary

- B0 / `original_pia_baseline`: all-valid uniform activation matching, kept as the original baseline definition.
- B0V / `variable_only_uniform`: variable-token-only uniform activation matching.
- P1 / `server_attn_last_raw`: raw last-query server attention rollout, kept as a negative control.
- P2 / `mts_mean_query`: server-side attention rollout with variable mask, power tempering, clipping, beta mixing, and activation calibration.
- P3 / `mts_last_window`: same MTS body, but the attention source is the final variable query window.

P2/P3 do not use dummy attention, alpha initialization, or gradient reranking in this branch.

## Audit Semantics

The attack path receives observed activation, sequence length, public model/tokenizer/config objects, and public framing assumptions. It must not receive prompt text, `original_ids`, `original_tokens`, or token ids.

- Position 0 and other `fixed_public` positions are public framing positions that the attacker can determine.
- If the attack API has no token ids, `token_id_based_special_mask_available=false`; the code must not claim token-id EOS detection.
- `last_valid_position` is only the last valid sequence position. It can be excluded as EOS only when the public protocol explicitly fixes the final token as EOS.
- Audit fields therefore use `raw_last_valid_mass` and `final_last_valid_weight`, not EOS naming.

## MTS Weight Construction

For P2/P3, the server side starts from the captured prefix activation and forwards through public server blocks. Attention matrices are averaged across heads, optionally include residual rollout, and are multiplied across the selected server depth.

The final loss weights are built as:

1. choose query source: mean query for P2, final variable query window for P3;
2. zero all non-variable positions;
3. apply power tempering;
4. clip to `weight_min` / `weight_max`;
5. renormalize variable-token mean weight to 1;
6. mix with uniform variable weights using beta;
7. enforce fixed public final weights as 0.

## Development and Held-Out Protocol

Development set:

- Skytrax-28, seed42, layer17, epoch100, K=10, Y=10.
- P2/P3 grid searched beta, weight power, and weight max on seed42 only.
- Structure ablation also used seed42 only.

Frozen held-out candidates:

- P2: beta 0.75, weight_power 0.50, weight_min 0.25, weight_max 4.0, residual rollout on, server depth last2.
- P3: beta 0.25, weight_power 0.50, weight_min 0.25, weight_max 2.0, residual rollout on, server depth all.

Seeds 43 and 44 are held-out validation seeds and were not used for retuning. The original stop rule was triggered at seed43. After the user explicitly requested to ignore the stop rule, seed44 and exploratory layer11/layer19 runs were completed without changing the method or frozen parameters.

## Current Interpretation

The development result was positive, but held-out seed43/44 did not reproduce the gain over B0V. Exploratory layer runs after the explicit override were mixed: layer11 was positive for frozen P3, while layer19 was negative. This branch should therefore be reported as a useful negative/diagnostic result rather than a stable improvement.
