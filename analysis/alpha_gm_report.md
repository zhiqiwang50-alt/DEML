# Attention-Initialized Gradient-Matched Prompt Inversion (AIGM-PIA) TinyLlama pilot

This is a TinyLlama pilot and must not be described as a full reproduction of the paper's large-model results.

## Threat Model

White-box attack using only observed boundary activation, sequence length, public TinyLlama prefix parameters, tokenizer, public vocabulary embeddings, and public semantic candidates.

The attack entry point is `invert_observed(model, tokenizer, cfg, observed_activation, seq_len, device)`. The outer runner uses prompt text and evaluation token ids only to produce observed activation and to score final outputs.

## Method

Stage A optimizes continuous dummy embeddings `D` to match `H_obs` and extracts a detached dummy-attention proxy `A_self = MeanHead(Attention_target_layer(D*))`. This proxy is not true prompt attention.

Stage B alpha-NN initialization computes `H_alpha = A_self @ H_obs`, then `t0_i = NN_W(H_alpha_i)` and `z0_i = E(t0_i)`. The residual version uses `H_alpha_residual = (1 - gamma) * H_obs + gamma * (A_self @ H_obs)`.

`H_obs @ A_self` is not used because `H_obs` is `[seq_len, hidden_dim]` and the attention matrix applies over sequence positions, so the valid left multiplication is `[seq_len, seq_len] @ [seq_len, hidden_dim]`.

Gradient-Direction Token Matching scores candidate directions `E(w) - z_i` against `-grad(L_rec, z_i)`. This is an attack-side recovery gradient from the reconstruction objective, not a recovered ground-truth training gradient.

## Existing Attention-Guided Baselines

The previous `dummy_init` used Stage-A optimized dummy embeddings directly as Stage-B initialization; it was not alpha-NN initialization. The previous `attention_context` used `A_dummy @ activation` as a context loss, not as initialization. The previous `attention_gradient` was gradient fusion, not Gradient-Direction Token Matching. The current `full` method from `pia_attention_guided.py` is not renamed or treated as AIGM-PIA.

## Results

`runs/alpha_gm_pilot/comparison.csv` and `.md` contain every completed configuration.

## Best Smoke Setting

Best 2-sample smoke setting: method `attention_context_existing`, seed `42`, layer `17`, epoch `100`, gamma `0.3`, eta `0.5`, accuracy `0.9962686567164178`.

## Best Gamma/Eta Sweep Setting

Best 24-sample sweep setting: method `alpha_nn_gradmatch_context`, seed `42`, layer `17`, epoch `200`, gamma `1.0`, eta `0.0`, accuracy `0.990452647240886`, BLEU `0.9849491554253162`, completed `22`, failed `2`.

The sweep shows `eta=0.0` is consistently strongest; increasing Gradient-Direction Token Matching weight degrades this pilot, so it should not be presented as a guaranteed improvement.

## Run Status

- Smoke comparison completed for 7 methods on 2 prompts, seed 42, layer 17, epoch 100.
- Gamma/eta sweep completed for `alpha_nn_gradmatch_context` on 24 requested prompts, seed 42, layer 17, epoch 200.
- Formal multi-seed/multi-layer comparison was not run in this session; use `scripts/run_alpha_gm_pilot.sh --mode comparison --resume`.

## Anti-Leakage Verification

- Full model vs manual prefix activation diff: max `0.0`, mean `0.0`.
- Attack API passes: `True`; banned signature hits: `[]`; banned global references: `[]`.
- Replacing evaluation-only prompt/token references leaves recovery unchanged: `True`.
- Alpha direction: `A_self @ H_obs`; z0 lookup max abs diff: `0.0`.
- Attention proxy shape `[135, 135]`, row-sum range `[0.9999368786811829, 1.0000474452972412]`, upper-triangular max `0.0`.

## Method Summary

| method | runs | mean acc across runs | mean BLEU across runs |
| --- | ---: | ---: | ---: |
| baseline | 1 | 0.9624179104477613 | 0.9427134048578516 |
| dummy_init_existing | 1 | 0.9736119402985075 | 0.9498991407743753 |
| attention_context_existing | 1 | 0.9962686567164178 | 0.9905109193811685 |
| alpha_nn_init_direct | 1 | 0.9850746268656716 | 0.9617986943720438 |
| alpha_nn_init_residual | 1 | 0.9962686567164178 | 0.9905109193811685 |
| alpha_nn_gradmatch | 1 | 0.4414626865671642 | 0.16015812403183063 |
| alpha_nn_gradmatch_context | 26 | 0.5149492636501096 | 0.3451048801631898 |

## Long vs Short Text

Prompt-level grouping is stored in each `predictions.jsonl`. The pilot summary currently aggregates by run; long/short prompt analysis can be recomputed directly from prompt token counts in those files.

## Attention BOS Concentration

Each `attention_stats.json` records `bos_column_mass_mean` and `bos_column_mass_max` to identify whether the dummy-attention proxy collapses onto BOS.

## Failure Cases

Failures are saved per run in `failures.jsonl`. With the current Skytrax-28 pilot, prompts exceeding `max_token_len` are skipped rather than silently truncated.

## Full Table

# Attention-Initialized Gradient-Matched Prompt Inversion (AIGM-PIA) TinyLlama pilot

| method | seed | layer | epoch | gamma | eta | acc mean | acc std | BLEU | completed | failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 42 | 17 | 100 | 0.3 | 0.5 | 0.9624179104477613 | 0.02958208955223879 | 0.9427134048578516 | 2 | 0 |
| dummy_init_existing | 42 | 17 | 100 | 0.3 | 0.5 | 0.9736119402985075 | 0.018388059701492543 | 0.9498991407743753 | 2 | 0 |
| attention_context_existing | 42 | 17 | 100 | 0.3 | 0.5 | 0.9962686567164178 | 0.0037313432835821003 | 0.9905109193811685 | 2 | 0 |
| alpha_nn_init_direct | 42 | 17 | 100 | 0.3 | 0.5 | 0.9850746268656716 | 0.014925373134328346 | 0.9617986943720438 | 2 | 0 |
| alpha_nn_init_residual | 42 | 17 | 100 | 0.3 | 0.5 | 0.9962686567164178 | 0.0037313432835821003 | 0.9905109193811685 | 2 | 0 |
| alpha_nn_gradmatch | 42 | 17 | 100 | 0.3 | 0.5 | 0.4414626865671642 | 0.0734626865671642 | 0.16015812403183063 | 2 | 0 |
| alpha_nn_gradmatch_context | 42 | 17 | 100 | 0.3 | 0.5 | 0.42522388059701494 | 0.014776119402985066 | 0.13310920260281095 | 2 | 0 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 0.0 | 0.0 | 0.9900947464390504 | 0.011180266490402687 | 0.9857726758126912 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 0.0 | 0.25 | 0.7667594408387617 | 0.06272490361655911 | 0.5599161044902846 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 0.0 | 0.5 | 0.4553875190813178 | 0.05189339900374816 | 0.16430643483348484 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 0.0 | 0.75 | 0.23301846385259034 | 0.04561068502825195 | 0.024423269951034395 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 0.0 | 1.0 | 0.1353070347773953 | 0.04234179723096104 | 0.010179548286763438 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 0.1 | 0.0 | 0.9904131247767762 | 0.012108246146333814 | 0.9874543213269198 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 0.1 | 0.25 | 0.7783582242994876 | 0.05105464008524525 | 0.5880014794176337 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 0.1 | 0.5 | 0.4507592820075913 | 0.0901706263714043 | 0.1585946568385878 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 0.1 | 0.75 | 0.22459364048779715 | 0.052429646773742136 | 0.030316296059311087 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 0.1 | 1.0 | 0.13955094582634855 | 0.0444963731832989 | 0.015151450710132604 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 0.3 | 0.0 | 0.9903169741378732 | 0.011303280809605527 | 0.9845344512194427 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 0.3 | 0.25 | 0.7723032013107485 | 0.06250598390376295 | 0.569643903467121 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 0.3 | 0.5 | 0.47848010420947723 | 0.05323524109689012 | 0.17749888102523062 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 0.3 | 0.75 | 0.2367607712616069 | 0.04607031503380928 | 0.03918436430033285 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 0.3 | 1.0 | 0.14131891807938196 | 0.04323127245840692 | 0.013773499256132695 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 0.5 | 0.0 | 0.9902812639032129 | 0.014919560459189678 | 0.9834559573306685 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 0.5 | 0.25 | 0.7891956880628083 | 0.06057186498823645 | 0.5824679436019256 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 0.5 | 0.5 | 0.4580985090167301 | 0.048914941325595845 | 0.14449652203562244 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 0.5 | 0.75 | 0.23904704881643704 | 0.04271909993298257 | 0.03485998617829503 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 0.5 | 1.0 | 0.13405490315189672 | 0.03799344679844836 | 0.011437408519945048 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 1.0 | 0.0 | 0.990452647240886 | 0.01683420029762493 | 0.9849491554253162 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 1.0 | 0.25 | 0.7667869734332091 | 0.10163316420948136 | 0.5719909969816751 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 1.0 | 0.5 | 0.47209909868025196 | 0.0715653422696232 | 0.18258079062615648 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 1.0 | 0.75 | 0.21801026113643868 | 0.05591879619429407 | 0.025721788754042747 | 22 | 2 |
| alpha_nn_gradmatch_context | 42 | 17 | 200 | 1.0 | 1.0 | 0.12200818947776121 | 0.04241253080440356 | 0.008905795191372797 | 22 | 2 |

