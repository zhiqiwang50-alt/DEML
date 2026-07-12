# Confidence-Gated Gradient-Directed Prompt Inversion (CG-GDPI) Design

This document describes the TinyLlama pilot method implemented in `pia_cg_gdpi.py`.
It is not a full reproduction of the original large-model Prompt Inversion Attack
paper.

## Threat Model

The attack stage receives only:

- observed boundary activation;
- sequence length;
- public TinyLlama parameters and tokenizer;
- public vocabulary embedding matrix;
- semantic candidates from the public TinyLlama oracle;
- a dummy-attention proxy learned from attack-side optimization;
- attack-side reconstruction gradients.

The attack API does not receive original prompt text, ground-truth token ids,
ground-truth embeddings, or evaluation references. Original prompt data is used
only to generate the target activation and to compute final metrics.

## Method

CG-GDPI keeps the residual alpha-NN attention initialization from AIGM-PIA and
replaces global min-max gradient score fusion with a calibration-first gate.

Stage A learns continuous dummy embeddings by matching the observed activation
with a nearest-vocabulary constraint. The target-layer self-attention weights are
averaged over heads into `A_proxy`. The implementation records row-sum checks,
causal upper-triangle leakage, entropy, sparsity, and BOS-column mass. The
optional `bos_debias` attention mode zeros BOS-column mass for rows after row 0
and renormalizes rows; it is only an ablation.

Stage B initializes with:

```text
H_init = (1 - gamma) * H_obs + gamma * (A_proxy @ H_obs)
```

Then it maps each position to the nearest public token embedding. This preserves
the residual attention direction without using dummy token ids or ground-truth
tokens.

Stage C optimizes continuous embeddings with activation loss, vocabulary
constraint loss, optional context loss, embedding bounds, and gradient clipping.
At the last five checkpoints it stores per-position reconstruction-gradient
snapshots and derives consensus `kappa`, gradient norm mean/std, and final update
direction.

Stage D builds each token candidate set from nearest optimized embeddings,
nearest residual alpha-NN initialization embeddings, semantic candidates, and the
initial token. It first scores every candidate by activation calibration and keeps
only calibration top-M. Gradient reranking is allowed only inside this top-M set
and only when:

```text
kappa >= tau_consensus
grad_margin >= tau_grad_margin
cal_margin <= prompt_quantile(cal_margins, tau_cal_margin_quantile)
```

If the gate is closed, the method chooses the calibration top-1 token. The
gradient can never select a token outside calibration top-M.

## Baselines

- `B0`: baseline constrained optimization plus adaptive discretization.
- `B1`: existing dummy initialization.
- `B2`: existing attention context objective.
- `B3`: residual alpha-NN initialization.
- `B4`: old alpha-NN global gradient matching baseline with min-max fusion.
- `P1`: CG-GDPI.
- `P2`: CG-GDPI with BOS-debiased attention proxy.

Only `P1` and `P2` are the proposed confidence-gated methods. `B4` is retained
to compare against the old global fusion behavior.

## Verification

`pia_cg_gdpi.py --mode verify` writes `runs/cg_gdpi/leakage_verification.json`
and checks:

- attack API has no reference-token/text inputs;
- replacing evaluation-only prompt text/token ids does not change recovery;
- full-model hook activation and manual prefix activation match;
- `top_m=1` equals pure activation calibration;
- forced closed gate equals pure activation calibration;
- every override stays inside calibration top-M;
- the proposed CG-GDPI reranker does not call the old global min-max fusion.

## Outputs

Each run directory contains:

- `config.json`
- `metrics.json`
- `predictions.jsonl`
- `stdout.log`
- `failures.jsonl`
- `attention_stats.json`
- `initialization_audit.json`
- `gradient_consensus_stats.json`
- `gradient_gate_stats.json`
- `COMPLETE` when finished

Summaries are written under `runs/cg_gdpi/` and `analysis/cg_gdpi_report.md`.
