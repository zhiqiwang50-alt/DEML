# Server-Attention Consistency PIA Design

SACR-PIA freezes the negative MTS finding: server attention rollout is no longer used as direct token-loss weights. The main reconstruction objective remains B0V variable-only uniform activation matching.

Auxiliary server information is used only as a consistency regularizer:

- C1 adds server tail hidden-state consistency.
- C2 adds server attention Jensen-Shannon consistency over variable query/key positions.
- C3 combines C1 and C2.

The attack API accepts only model, tokenizer, cfg, observed activation H_obs, sequence length, and device. It does not accept prompt text, original ids/tokens, input ids, ground-truth embeddings, dummy_x, dummy attention, or client-side true attention.

Prompt split and optimization initialization are separated by `--prompt-split-seed` and `--init-seed`. The canonical split is saved under `runs/server_attention_consistency_pia/splits/`.

Auxiliary losses are normalized by their detached step-0 value. Per-step gradient caps enforce `lambda_eff * ||g_aux|| <= rho * ||g_base||`; effective lambdas, gradient norms, cosines, and cap violations are written for every run.
