# Masked and Tempered Server-Attention Prompt Inversion Design

## Layer mapping

TinyLlama uses 0-based block indices. `capture_prefix_activation(model, target_layer=L, ...)` manually executes blocks `0..L` and returns the output of block `L`. Thus `target_layer=17` is `H^(17)`, and server-side layers are `18..21`.

## Starting point

The existing SAW-PIA result already feeds `H_obs` through public server-side layers, extracts each server layer attention, averages heads, applies `RowNorm(I + A)`, and computes rollout in forward order:

`R = A_tilde_last @ ... @ A_tilde_first`

The leakage verification shows that `H_obs` is the target block output, server layers start at `target_layer + 1`, and manual server forward matches full-model forward.

## Motivation

Full Skytrax-150 top-1 results show that raw last-query weighting degrades strongly at deeper layers. The rollout mass can concentrate on position 0, and the old weighted loss does not explicitly exclude fixed public/special token positions.

## Audit semantics

The attack function receives `H_obs`, sequence length, tokenizer/model/config, and public framing assumptions. It does not receive `original_ids`, `original_tokens`, or prompt text.

- `fixed_public` positions such as position 0 can be determined by the public protocol and are safe to mark as public framing positions.
- When `token_ids=None`, `token_id_based_special_mask_available=false`; the branch cannot claim it identified EOS from token id.
- `last_valid_position` is only the last position allowed by the attention mask. It is not called EOS in the audit.
- Only if a public protocol explicitly fixes the final token as EOS should a caller exclude that position as public framing.
- Weight statistics therefore use `raw_last_valid_mass` and `final_last_valid_weight`, not EOS names.

## Implemented methods

- B0 / `original_pia_baseline`: original PIA baseline with uniform activation loss over all valid tokens.
- B0V / `variable_only_uniform`: new variable-token-only uniform baseline. Fixed public and special positions have final weight 0.
- P1 / `server_attn_last_raw`: old raw last-query rollout, kept only as a negative control.
- P2 / `mts_mean_query`: MTS weights from mean query attention over variable query positions.
- P3 / `mts_last_window`: MTS weights from the final variable query window.

No dummy attention or alpha-initialization combo is used in this branch.

## MTS loss

The branch builds a `variable_mask` from the attention mask and public fixed positions. For P2/P3 it extracts server-side attention by forwarding only from `H_obs` through public server layers, computes attention rollout, and converts rollout mass into token weights:

1. choose the source query set: `mean_query` or `last_window_mean`;
2. zero all non-variable positions;
3. apply power tempering with `--weight-power`;
4. clip to `--weight-min` / `--weight-max`;
5. renormalize variable-token mean weight to 1.0;
6. mix with uniform weights using `--beta`;
7. enforce fixed public/special final weights as 0.

With `--beta 0`, P2/P3 reduce to B0V for the weighted loss and recovered ids in the unit tests.

## Development ablation

`--mode dev-ablation` runs the requested Skytrax-28 seed42 layer17 development grid in the same branch:

- baselines: B0, B0V, P1;
- P2/P3 grid: beta 0.25/0.50/0.75, power 0.25/0.50/1.00, weight_max 2/4;
- fixed settings: epoch 100, K=10, Y=10, residual rollout on, rollout depth all, weight_min 0.25;
- auto-batch is capacity-aware, so `--max-jobs-per-gpu 1` keeps one active task per GPU;
- outputs: `dev_ablation.csv`, `dev_ablation.md`, `dev_weight_summary.csv`, `dev_token_accuracy.png`, and `dev_weight_concentration.png`.

## Resource-aware batching

The new runner adds `--mode plan-batch` and `--auto-batch`. It reads current GPU free memory with `nvidia-smi`, reserves a safety buffer, and schedules concurrent single-prompt optimization workers only on GPUs with enough free memory.

Current conservative defaults:

- `--batch-free-mb-per-job 6144`
- `--batch-reserve-free-mb 2048`
- `--max-jobs-per-gpu 2`
- `--max-parallel-jobs 3`

Under the checked server state, this selects two workers: GPU0 and GPU3. This shortens wall-clock time while avoiding the tighter-memory GPU1/GPU2.

## Required checks

- Unit tests cover beta=0 equivalence, recovered ids under beta=0, fixed/special zero weights, max clipping, last-window variable-only query positions, and B0 all-valid loss.
- Leakage verification checks that the attack API does not receive prompt/original ids, manual server forward matches full-model forward, and no dummy attention is used.
- Current pilot smoke is Skytrax-28, dataset_len=2, seed=42, layer=17, epoch=100, K=10, Y=10, methods B0/B0V/P1/P2/P3.
