# CA-GDM white-box Prompt Inversion pilot report

Date: 2026-06-25
Project: `/data/zhiqi/Collaborative_Inference/DEML`
Setup: TinyLlama-1.1B-Chat-v1.0, Skytrax-28 pilot, seed 42, target layer 17, participant 4 / attacker 4, epoch 100, stage-A epoch 100.

## Implementation

This run keeps the previous scripts untouched and adds the CA-GDM path in separate files:

- `pia_cagdm.py`
- `scripts/run_cagdm.sh`
- `tests/test_cagdm_unit.py`
- `analysis/cagdm_design.md`
- `analysis/cagdm_report.md`

The attack does not use original prompt text, original token ids, ground-truth embeddings, or ground-truth gradients during inversion. Ground truth is used only for post-hoc evaluation.

## Method interpretation

The paper-style prompt inversion method and white-box gradient matching are not the same thing. Prompt inversion in collaborative inference can be implemented as activation or hidden-state matching around the observed split activation. White-box gradient matching additionally assumes differentiable access to model internals and tries to align update directions or residual gradients.

In this implementation, CA-GDM currently uses gradient information as a gated reranker over calibration top-m candidates. It is not yet a full continuous gradient-matching objective over prompt embeddings.

For a paper positioned around white-box gradient matching, the safer framing is:

1. Calibration / alpha-self initialization is the primary recovery path.
2. Gradient matching is an auxiliary loss or audited tie-breaker.
3. The gate is reported as an ablation until its override audit becomes positive.

## Full-run results

All runs below completed 28 / 28 samples with 0 failures.

| Run | Token accuracy mean +/- std | BLEU mean +/- std | Mean gate open | Mean override | Override audit |
|---|---:|---:|---:|---:|---|
| B1 full, dummy-init existing | 0.989797 +/- 0.031407 | 0.980254 +/- 0.058356 | n/a | n/a | n/a |
| P1 gate closed full | 0.991473 +/- 0.030045 | 0.985201 +/- 0.046375 | 0.000000 | 0.000000 | no overrides |
| P1 balanced full, cosine | 0.983762 +/- 0.027026 | 0.966321 +/- 0.045018 | 0.009696 | 0.008060 | selected 0 / 36 correct; calibration 32 / 36 correct; delta -0.888889 |
| P2 balanced full, dot | 0.963821 +/- 0.045486 | 0.925868 +/- 0.074208 | 0.032656 | 0.027338 | selected 0 / 147 correct; calibration 127 / 147 correct; delta -0.863946 |
| P3 balanced full, BOS-debiased | 0.987334 +/- 0.012354 | 0.972075 +/- 0.027527 | 0.011107 | 0.009017 | selected 0 / 47 correct; calibration 46 / 47 correct; delta -0.978723 |

## Smoke controls

| Run | Token accuracy | BLEU | Note |
|---|---:|---:|---|
| B0 smoke | 1.000000 | 1.000000 | baseline smoke passed |
| B1 smoke | 1.000000 | 1.000000 | dummy-init smoke passed |
| B2 smoke | 1.000000 | 1.000000 | attention-context smoke passed |
| B3 smoke | 0.996269 | 0.990511 | alpha residual close to exact |
| B4 smoke | 0.242060 | 0.044589 | global gradient-matching control failed |
| P1 smoke from terminal | 1.000000 | 1.000000 | gate stayed closed |
| P2 smoke | 1.000000 | 1.000000 | gate stayed closed |
| P3 smoke | 1.000000 | 1.000000 | gate stayed closed |
| P1 relaxed smoke | 0.115164 | 0.009199 | forced-open style gate collapses recovery |

## Gate diagnosis

The current gradient-directed gate is harmful when it opens:

- P1 cosine: 36 overrides, gradient-selected token correct 0 times; calibration token correct 32 times.
- P2 dot: 147 overrides, gradient-selected token correct 0 times; calibration token correct 127 times.
- P3 BOS-debias: 47 overrides, gradient-selected token correct 0 times; calibration token correct 46 times.

Dominant closed reasons were `kappa_below_tau`, `grad_margin_below_tau`, and `calibration_confident`. The gate mostly suppresses the gradient signal, but the few accepted overrides are still wrong.

## Conclusion

The current evidence does not support claiming that gradient-directed reranking improves prompt inversion. The strongest full run is P1 with the gradient gate closed. This slightly exceeds the B1 comparator on this single seed and dataset, but multi-seed and layer-depth repeats are still required.

Recommended paper revision:

- Use calibration / alpha-self initialization as the main method.
- Move gradient matching into a continuous auxiliary objective or a learned confidence gate.
- Keep `--force-gate-closed` as the default reported setting for now.
- Present the current CA-GDM gate as a negative ablation showing that naive gradient-direction token reranking is anti-correlated with correctness in this setup.
