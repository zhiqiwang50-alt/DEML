# CAR-PTR Strict Top-1 Design Spec

## Safety and Authorization Scope

This work is an authorized academic research experiment on prompt inversion risks in collaborative inference. All experiments are limited to the user's own isolated environment:

- rack-server-codex managed by the user.
- DEML repository controlled by the user.
- Local TinyLlama model files.
- Local Skytrax academic experiment data.
- Offline, isolated model evaluation environment.

This work must not access or test third-party systems, scan networks, exploit vulnerabilities, collect credentials or personal data, elevate privileges, bypass access controls, create malware or remote-control behavior, damage systems, or send attack traffic to external targets.

## Goal

Build and evaluate a white-box prompt inversion branch that uses the server-side attention formula as a controlled weighting signal while preserving a strict Top-1 reconstruction path. The main comparison is against the paper-style PIA baseline B0, with B0V used as a stronger variable-token constraint.

## Threat Model and API Boundary

- White-box attacker knows the complete model parameters and receives the observed intermediate activation `H_obs`.
- The attack API must not receive or read `prompt`, `original_ids`, `original_tokens`, ground-truth text, or ground-truth token ids.
- Public fixed positions such as BOS and fixed framing tokens may be masked if they are protocol-known.
- The method must not infer EOS from token ids unless token ids are public to the attack API. If the protocol does not publicly define the final valid position as EOS, it must be called `last_valid`, not EOS.

## Strict Top-1 Constraint

The new branch must enforce a deterministic single-token projection path:

- `K=1`.
- `Y=0`.
- `semantic_speculation=false`.
- Full-vocabulary nearest-neighbor projection is allowed.
- Top-K candidate lists, beam search, semantic candidates, and multi-candidate reranking are not allowed.

Existing older runs with `K=1`, `Y=1`, and semantic speculation enabled are historical only. They are not strict Top-1 baselines and cannot be used as the final paper comparison.

## Baselines and Methods

- `B0`: original PIA all-valid uniform activation matching plus nearest-neighbor projection. This is the paper-style main baseline and must not be redefined.
- `B0V`: variable-only uniform activation matching plus nearest-neighbor projection. This is a strong constraint baseline, not the paper's main baseline.
- `P4S-T1`: attention-scaled prediction loss, `||alpha h_pred - h_obs||^2`, strict Top-1 projection.
- `P4D-T1`: attention-weighted residual loss, `||alpha (h_pred - h_obs)||^2`, strict Top-1 projection.
- `CAR`: confidence-aware rollout residual matching, strict Top-1 projection.
- `P4D-PTR`: `P4D-T1` plus projected Top-1 residual correction.
- `CAR-PTR`: CAR plus projected Top-1 residual correction. This is the proposed main method.

## Stage 1: CAR, Confidence-Aware Rollout Residual Matching

Collect server-side suffix attentions once and derive two rollout views from the same collection:

- Full suffix rollout: `r_all`.
- Last-2 suffix rollout: `r_last2`.

Build a consensus attention score:

```text
r_con_i = sqrt((r_all_i + eps) * (r_last2_i + eps))
```

Mask public fixed positions and BOS. On the remaining variable-token positions `V`, apply power tempering and mean-one normalization:

```text
tilde_r_i = |V| * r_con_i^0.5 / sum_{j in V} r_con_j^0.5
```

Estimate rollout disagreement:

```text
d_i = |log(r_all_i + eps) - log(r_last2_i + eps)|
c_i = exp(-d_i / (median_{j in V}(d_j) + eps))
```

Convert attention to a bounded residual weight:

```text
w_i = MeanOneClip_[0.5, 1.5](1 + 0.25 * c_i * (tilde_r_i - 1))
alpha_i = sqrt(w_i)
```

Apply a training schedule:

- Epoch 1 to 50: uniform residual matching.
- Epoch 51 to 69: linearly ramp from uniform to CAR.
- Epoch 70 to 100: full CAR weight.

At epoch `t`:

```text
w_i^(t) = MeanOneClip(1 + s_t * (w_i - 1))
alpha_i^(t) = sqrt(w_i^(t))
```

The CAR loss keeps the same residual form as P4D:

```text
L_CAR = mean_i || alpha_i^(t) * (h_pred_i - h_obs_i) ||_2^2 + 0.1 * L_vocab
```

Fixed public positions and BOS have final weight zero. Variable-token weights have mean one after clipping and normalization.

## Stage 2: PTR, Projected Top-1 Residual Correction

PTR is a conservative post-processing step that remains strict Top-1. It starts from the single nearest-neighbor reconstruction:

```text
x0 = NN(z*)
```

PTR computes a proposal loss:

```text
L_prop = sum_i w_i || H_i(E(x)) - H_obs_i ||_2^2
```

It chooses exactly one editable position per proposal:

```text
i* = argmax_i c_i * || grad_{E(x_i)} L_prop ||_2
```

Then it optimizes only that selected position for 5 inner steps with learning rate `0.025`, projects once to the full-vocabulary nearest neighbor, and evaluates the proposal with the unweighted B0V activation loss.

A proposal is accepted only if:

```text
B0V_loss(candidate) < B0V_loss(current) - 1e-6
```

PTR limits:

- Maximum proposals: 10.
- Maximum accepted token changes: 5.
- Early stop after 3 consecutive rejects.
- No semantic candidates.
- No candidate list.
- No ground-truth text or token ids.

For `P4D-PTR`, if no confidence vector is available, `c_i = 1` for all variable-token positions.

## Experiment Plan

Common settings:

- Model: TinyLlama-1.1B.
- Dataset path: `data/airline.json`.
- Participant number: 4.
- Attacker position: 4.
- Epoch: 100.
- Max token length: 896.
- `K=1`.
- `Y=0`.
- Semantic speculation disabled.
- Every method in the same comparison cell uses the same seed, prompt ids, prompt order, target layer, epoch count, and max token length.

### Stage 0: Unit Tests and Smoke

Run unit tests and a tiny smoke test before any real experiment. Stop on any of the following:

- Attack API receives banned ground-truth fields.
- Fixed public/BOS final weight is nonzero.
- Variable-token weight mean differs from one beyond tolerance.
- Final weight exceeds `[0.5, 1.5]`.
- PTR creates more than one projected token per proposal.
- PTR accepts a proposal that does not reduce B0V activation loss by more than `1e-6`.
- NaN appears.
- Any sample fails in smoke.

### Stage 1: Development Check

Run Skytrax-28, seed 42, layer 17, for all seven methods: B0, B0V, P4S-T1, P4D-T1, CAR, P4D-PTR, CAR-PTR.

Continue only if CAR-PTR satisfies all of:

- Token Accuracy greater than P4S-T1.
- BLEU not lower than P4S-T1.
- Token Accuracy and BLEU greater than B0.
- Token Accuracy and BLEU greater than B0V.
- Paired wins greater than paired losses against P4S-T1.
- No audit violation.

### Stage 2: Layer Development Check

Freeze all CAR-PTR parameters and run seed 42 on layers 11, 17, and 19.

Continue only if:

- Pooled Token Accuracy and BLEU are greater than P4S-T1.
- At least two of three layers have nonnegative Token Accuracy and BLEU deltas against P4S-T1.
- CAR-PTR is greater than B0V on every tested layer.
- No audit violation.

### Stage 3: Full Validation

Only after Stages 0 to 2 pass, run Skytrax-150 across seeds 42, 43, and 44 and layers 11, 17, and 19 for all seven methods. This is 63 method/seed/layer cells.

Use the four 4090D GPUs with at most one active job per GPU. Do not retune CAR, PTR, layer, epoch, `K`, `Y`, prompt split, or method parameters during this stage.

## Success Criteria

CAR-PTR may be described as a stable improvement only if all conditions hold:

- Pooled Token Accuracy is greater than P4S-T1.
- Pooled BLEU is greater than P4S-T1.
- For every seed, pooled across layers, Token Accuracy and BLEU deltas against P4S-T1 are nonnegative.
- At least six of nine seed-layer cells have simultaneous nonnegative Token Accuracy and BLEU deltas against P4S-T1.
- Pooled paired wins exceed losses.
- 10,000-sample prompt-level paired bootstrap confidence intervals for both Token Accuracy and BLEU have lower bounds greater than zero.
- There are no NaNs, failed samples, leakage audit failures, fixed-token weight violations, or PTR acceptance violations.

If the confidence interval lower bound is not greater than zero, report the result as an observed improvement or mixed result, not a stable improvement.

## Required Reporting

Reports must include:

- Per-method Token Accuracy and BLEU.
- Deltas against B0, B0V, and P4S-T1.
- Paired win/tie/loss counts.
- 10,000-sample bootstrap confidence intervals.
- Rollout disagreement statistics.
- Confidence statistics.
- Final `w` and `alpha` range.
- PTR proposal count, accept count, reject count, token-change count.
- Pre/post B0V residual loss delta for PTR.
- Completed and failed sample counts.
- NaN count.
- Attack API and leakage-audit status.

Reports must not claim stable improvement unless the full validation success criteria pass.
