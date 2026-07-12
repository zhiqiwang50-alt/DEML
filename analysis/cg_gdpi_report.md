# Confidence-Gated Gradient-Directed Prompt Inversion (CG-GDPI) TinyLlama white-box pilot

This is a TinyLlama white-box pilot, not a complete reproduction of the original large-model paper.

## Method

CG-GDPI keeps residual alpha-NN initialization and replaces global min-max gradient fusion with calibration-first top-M reranking guarded by gradient consensus.

B4 is retained as the old global gradient-fusion baseline. P1/P2 do not call the old min-max fusion path.

## Execution Status

Completed configurations in the summary table: `13`.

The completed main-comparison subset includes Skytrax-28 pilot, participant 4, attacker position 4, target layer 17, epoch 100, seed 42 for B0/B1/B2/B3/B4/P1. It also includes any additional configurations that have a `COMPLETE` marker.

The remaining requested main seeds/epoch-200 runs, depth generalization, and ablations are not marked complete unless they appear in the table below.

Partial directories retained but excluded from the table:

- `B0_p4_a4_layer17_epoch100_seed42_topm3_raw_open_g0p3_eta0p5`: `19` prediction rows, no `COMPLETE` marker.
- `B1_p4_a4_layer17_epoch100_seed42_topm3_raw_open_g0p3_eta0p5`: `13` prediction rows, no `COMPLETE` marker.
- `B1_p4_a4_layer17_epoch100_seed43_topm3_raw_open_maxlen640_g0p3_eta0p5`: `25` prediction rows, no `COMPLETE` marker.
- `B2_p4_a4_layer17_epoch100_seed42_topm3_raw_open_g0p3_eta0p5`: `13` prediction rows, no `COMPLETE` marker.
- `B2_p4_a4_layer17_epoch100_seed43_topm3_raw_open_maxlen640_g0p3_eta0p5`: `2` prediction rows, no `COMPLETE` marker.
- `B3_p4_a4_layer17_epoch100_seed43_topm3_raw_open_maxlen640_g0p3_eta0p5`: `20` prediction rows, no `COMPLETE` marker.

## Best Completed Setting

Best observed run: stage `smoke`, method `baseline`, seed `42`, layer `17`, epoch `100`, top_m `3`, attention `raw`, token accuracy `1.0`, BLEU `1.0`.

## Anti-Leakage Verification

- Prefix/full activation max diff: `0.0`; mean diff: `0.0`.
- Attack API passes: `True`.
- top_m=1 equals pure calibration: `True`.
- force gate closed equals pure calibration: `True`.
- All overrides inside calibration top-M: `True`.

## CG-GDPI Gate Summary

- P1 seed 42 epoch 100 gate open mean: `0.0`.
- P1 seed 42 epoch 100 override mean: `0.0`.
- P1 gate samples: `28`.

## Findings

- Alpha-NN initialization remains useful in this pilot when followed by calibration-safe selection.
- The previous global gradient-matching fusion fails because per-position min-max scaling can amplify noisy gradients.
- CG-GDPI only allows gradients to rerank calibration top-M candidates when consensus and margin checks pass.
- BOS debiasing is implemented as an ablation; it is not assumed to help without measurement.
- Depth generalization and full multi-seed results are reported only for runs that have COMPLETE markers.

## Full Table

# Confidence-Gated Gradient-Directed Prompt Inversion (CG-GDPI) TinyLlama white-box pilot

| stage | method | seed | layer | epoch | top_m | attention | gate_closed | acc mean | acc std | BLEU | completed | failed |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| main_comparison | baseline | 42 | 17 | 100 | 3 | raw | False | 0.9847436973434537 | 0.04205891885829532 | 0.9742062980514853 | 28 | 0 |
| main_comparison | baseline | 43 | 17 | 100 | 3 | raw | False | 0.9885886455354032 | 0.03292370449877594 | 0.9794764457649561 | 28 | 0 |
| main_comparison | dummy_init_existing | 42 | 17 | 100 | 3 | raw | False | 0.9897970572799956 | 0.030841546581646638 | 0.9802539619958822 | 28 | 0 |
| main_comparison | attention_context_existing | 42 | 17 | 100 | 3 | raw | False | 0.9978848768646456 | 0.00514936968340821 | 0.9958967697530117 | 28 | 0 |
| main_comparison | alpha_nn_init_residual | 42 | 17 | 100 | 3 | raw | False | 0.9816100806667849 | 0.07700972525100114 | 0.9721334592103684 | 28 | 0 |
| main_comparison | alpha_nn_global_gradmatch | 42 | 17 | 100 | 3 | raw | False | 0.25243873677596856 | 0.060271388531648935 | 0.054040778179828115 | 28 | 0 |
| main_comparison | cg_gdpi | 42 | 17 | 100 | 3 | raw | False | 0.9971014554440709 | 0.00765260075176551 | 0.9935173298428167 | 28 | 0 |
| smoke | baseline | 42 | 17 | 100 | 3 | raw | False | 1.0 | 0.0 | 1.0 | 2 | 0 |
| smoke | dummy_init_existing | 42 | 17 | 100 | 3 | raw | False | 1.0 | 0.0 | 1.0 | 2 | 0 |
| smoke | attention_context_existing | 42 | 17 | 100 | 3 | raw | False | 1.0 | 0.0 | 1.0 | 2 | 0 |
| smoke | alpha_nn_init_residual | 42 | 17 | 100 | 3 | raw | False | 0.9962686567164178 | 0.0037313432835821003 | 0.9905109193811685 | 2 | 0 |
| smoke | alpha_nn_global_gradmatch | 42 | 17 | 100 | 3 | raw | False | 0.2420597014925373 | 0.03405970149253733 | 0.04458931720465496 | 2 | 0 |
| smoke | cg_gdpi | 42 | 17 | 100 | 3 | raw | False | 1.0 | 0.0 | 1.0 | 2 | 0 |

