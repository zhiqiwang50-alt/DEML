# Server-Attention-Weighted Prompt Inversion Design

## Layer mapping

TinyLlama uses 0-based block indices. `capture_prefix_activation(model, target_layer=L, ...)` manually executes blocks `0..L` and returns the output of block `L`. Thus `target_layer=17` is `H^(17)`, and server-side layers are `18..21`.

## Method

SAW-PIA feeds `H_obs` through public server-side layers, extracts each server layer attention, averages heads, applies `RowNorm(I + A)`, and computes rollout in forward order:

`R = A_tilde_last @ ... @ A_tilde_first`

For causal TinyLlama, token importance defaults to the last valid query row: `w_i = R[last_valid_position, i]`, normalized to mean 1 with optional floor.

The activation loss becomes a weighted per-token loss:

`L = sum_i w_i mean_h((F_prefix(z)_i - H_obs_i)^2) / sum_i w_i + lambda_vocab L_vocab`

P1/P2 do not use dummy embeddings or dummy attention. P3 is an explicit contrast that combines alpha initialization with server attention loss.
