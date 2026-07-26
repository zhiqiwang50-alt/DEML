# Attention-Scaled Activation PIA Top-1 Design

## Purpose

This branch tests a Top-1 white-box Prompt Inversion variant that is closer to the attention formula proposed in the discussion image. The primary baseline is B0, the paper-style PIA baseline. B0V is kept as a stronger variable-only constraint.

## Loss Formula

The old MTS branch weights the activation error:

`L_weighted = mean_i alpha_i * ||H_pred_i - H_obs_i||^2`

This branch scales the predicted activation before matching:

`L_scale = mean_{i in variable} ||alpha_i * H_pred_i - H_obs_i||^2`

Only variable positions are included. Public fixed positions such as BOS/framing are excluded from the loss and have final alpha 0.

## Implemented Methods

- P4 / `attn_scale_mean_query`: server attention rollout, mean over variable query positions, alpha-normalized activation scaling.
- P5 / `attn_scale_last_window`: server attention rollout, mean over the last variable query window, alpha-normalized activation scaling.
- P6 / `attn_scale_last_window_gate`: P5 plus an embedding-margin uncertainty gate; non-gated variable positions fall back to alpha=1.

These methods do not use dummy attention, alpha initialization, gradient reranking, prompt text, original token ids, or original tokens.

## Stabilizers

- variable-only mask;
- fixed public positions forced to alpha 0;
- raw attention mass transformed with power tempering;
- alpha clipped to `alpha_min` / `alpha_max`;
- variable-position alpha mean normalized to 1;
- beta mixing with uniform alpha;
- scheduled activation: B0V warmup before `attention_start_ratio`, then attention-scaled activation loss.

## Top-1 Dev Setting

- dataset: Skytrax-28 from `data/airline.json`;
- seed: 42;
- model: TinyLlama/TinyLlama-1.1B-Chat-v1.0, local files only;
- participant number: 4;
- attacker position: 4;
- target layer: 17;
- epoch: 100;
- K/Y: 1/1;
- max token length: 896;
- residual rollout: on;
- rollout depth: all;
- attention start ratio: 0.70;
- GPU scheduling: one active task per GPU on available GPUs.

## Output Files

- `runs/attention_scaled_activation_pia_top1/dev_ablation.csv`
- `runs/attention_scaled_activation_pia_top1/dev_ablation.md`
- `runs/attention_scaled_activation_pia_top1/dev_alpha_summary.csv`
- `runs/attention_scaled_activation_pia_top1/dev_token_accuracy.png`
- `runs/attention_scaled_activation_pia_top1/dev_alpha_concentration.png`

## Interpretation Rule

This is a development-set Top-1 ablation. It can identify candidates for held-out validation, but it is not evidence of stable improvement until held-out seeds, Top-K=10 main setting, and multi-layer checks pass.

## P4S-Res Residual Attention Scaling

P4S-Res keeps the same white-box, strict Top-1 attack setting as P4S. It does not add Top-K reranking, ensemble selection, dummy attention, or projection refinement. The only change is to soften the scheduled P4S alpha before applying the scaled activation loss:

alpha_res_i = 1 + rho * (alpha_i - 1)

L = mean_i || alpha_res_i * h_i(x_tilde) - h_i_target ||^2

Here rho=0 reduces to uniform variable-token scaling, and rho=1 recovers the original P4S attention scaling strength. The seed42 candidate comparison tested a global rho=0.5 and a layer-adaptive setting. The layer-adaptive setting was then frozen as layer11=1.0, layer17=0.6, layer19=0.4 before seed43/44.

Interpretation: residual scaling remains positive relative to B0V in pooled seed42/43/44 results, but it does not outperform the original P4S branch on heldout seed43/44. It should be reported as a diagnostic/protective residual variant rather than the new main method.
