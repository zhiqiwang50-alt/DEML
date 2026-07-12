# Masked Server Attention PIA Report

This report summarizes the completed MTS-PIA experiments for `pia_masked_server_attn_pia.py`. It separates development-set tuning from held-out validation and from the later user-requested stop-rule override.

## Setup

- Dataset: Skytrax-28 from `data/airline.json`
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Participant number / attacker position: 4 / 4
- Main development layer: 17
- Epoch: 100
- K / Y: 10 / 10
- Max token length: 896
- Attack API audit: no prompt text, `original_ids`, `original_tokens`, or token ids are passed into the attack function.

## Development Result: Seed42 Layer17

Seed42 was used for development and parameter selection only.

| method | Acc | BLEU | dAcc vs B0 | dAcc vs B0V | dBLEU vs B0 | dBLEU vs B0V | done/fail |
|---|---:|---:|---:|---:|---:|---:|---|
| B0 | 0.9847 | 0.9742 | 0.0000 | -0.0110 | 0.0000 | -0.0165 | 28/0 |
| B0V | 0.9958 | 0.9907 | 0.0110 | 0.0000 | 0.0165 | 0.0000 | 28/0 |
| P1 | 0.9500 | 0.9140 | -0.0347 | -0.0457 | -0.0602 | -0.0767 | 28/0 |
| best P2 | 0.9983 | 0.9960 | 0.0135 | 0.0025 | 0.0218 | 0.0054 | 28/0 |
| best P3 | 0.9986 | 0.9963 | 0.0138 | 0.0028 | 0.0221 | 0.0057 | 28/0 |

Best development parameters:

| method | beta | power | weight_max | residual | depth | fixed sum max | max observed weight |
|---|---:|---:|---:|---|---|---:|---:|
| P2 | 0.75 | 0.50 | 4.0 | on | last2 | 0.0000 | 1.8062 |
| P3 | 0.25 | 0.50 | 2.0 | on | all | 0.0000 | 1.2500 |

Note: the P2 structural ablation selected `last2` even though the initial development grid used `all`.

## Held-Out Validation

Seeds 43 and 44 were not used for tuning.

| seed | method | dAcc vs B0V | dBLEU vs B0V | dAcc vs B0 | dBLEU vs B0 | win/tie/loss vs B0V | bootstrap Acc CI | bootstrap BLEU CI | raw BOS | final BOS | fixed sum | max weight | done/fail |
|---:|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---|
| 43 | P2 | -0.0153 | -0.0265 | -0.0069 | -0.0114 | 1/19/8 | [-0.0350, -0.0026] | [-0.0585, -0.0052] | 0.5335 | 0.0000 | 0.0000 | 1.8062 | 28/0 |
| 43 | P3 | -0.0076 | -0.0096 | 0.0009 | 0.0055 | 1/22/5 | [-0.0213, -0.0000] | [-0.0258, -0.0001] | 0.7661 | 0.0000 | 0.0000 | 1.2500 | 28/0 |
| 44 | P2 | -0.0013 | -0.0015 | -0.0018 | -0.0022 | 4/18/6 | [-0.0049, 0.0019] | [-0.0076, 0.0042] | 0.5335 | 0.0000 | 0.0000 | 1.8062 | 28/0 |
| 44 | P3 | -0.0083 | -0.0104 | -0.0088 | -0.0110 | 6/18/4 | [-0.0291, 0.0036] | [-0.0390, 0.0068] | 0.7661 | 0.0000 | 0.0000 | 1.2500 | 28/0 |

Cross held-out summary:

| method | pooled dAcc vs B0V | pooled dBLEU vs B0V | win/tie/loss | audit ok | continue layers by original rule |
|---|---:|---:|---|---|---|
| P2 | -0.0083 | -0.0140 | 5/37/14 | true | false |
| P3 | -0.0079 | -0.0100 | 7/40/9 | true | false |

The original stop rule was triggered on seed43. Seed44 was run only because the user explicitly requested to ignore the previous stopping rule.

## Exploratory Layer Override

After the explicit override, frozen P3 was used as the least-negative candidate for an exploratory layer11/layer19 check on seed42. No parameters were retuned.

| layer | label | method | Acc | BLEU | dAcc vs B0 | dBLEU vs B0 | dAcc vs B0V | dBLEU vs B0V | done/fail | fixed sum max | max weight |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 11 | B0 | original_pia_baseline | 0.9546 | 0.9292 | 0.0000 | 0.0000 | 0.0270 | 0.0289 | 28/0 | 1.0000 | 1.0000 |
| 11 | B0V | variable_only_uniform | 0.9277 | 0.9003 | -0.0270 | -0.0289 | 0.0000 | 0.0000 | 28/0 | 0.0000 | 1.0000 |
| 11 | P3 | mts_last_window | 0.9772 | 0.9672 | 0.0226 | 0.0379 | 0.0495 | 0.0669 | 28/0 | 0.0000 | 1.2500 |
| 19 | B0 | original_pia_baseline | 0.9960 | 0.9917 | 0.0000 | 0.0000 | -0.0030 | -0.0058 | 28/0 | 1.0000 | 1.0000 |
| 19 | B0V | variable_only_uniform | 0.9990 | 0.9974 | 0.0030 | 0.0058 | 0.0000 | 0.0000 | 28/0 | 0.0000 | 1.0000 |
| 19 | P3 | mts_last_window | 0.9944 | 0.9881 | -0.0016 | -0.0035 | -0.0046 | -0.0093 | 28/0 | 0.0000 | 1.2500 |

Layer11 is positive for frozen P3, but layer19 is negative. This is not evidence of cross-layer generalization.

## Actual Commands

Held-out seed44:

```bash
python pia_masked_server_attn_pia.py --mode heldout-seed --output-root runs/masked_server_attn_pia --dataset-name Skytrax-28 --dataset-path data/airline.json --dataset-len 28 --seed 44 --participant-number 4 --attacker-position 4 --target-layer 17 --epoch 100 --k 10 --y 10 --max-token-len 896 --weight-min 0.25 --max-jobs-per-gpu 1 --max-parallel-jobs 4 --batch-free-mb-per-job 4096 --batch-reserve-free-mb 2048 --local-files-only --bootstrap-iters 10000
```

Cross-seed summary:

```bash
python pia_masked_server_attn_pia.py --mode summarize-heldout --output-root runs/masked_server_attn_pia --seeds 43 44 --bootstrap-iters 10000
```

Exploratory layer override:

```bash
python pia_masked_server_attn_pia.py --mode multi-layer --auto-batch --methods B0 B0V P3 --target-layers 11 19 --output-root runs/masked_server_attn_pia --dataset-name Skytrax-28 --dataset-path data/airline.json --dataset-len 28 --seed 42 --participant-number 4 --attacker-position 4 --epoch 100 --k 10 --y 10 --max-token-len 896 --beta 0.25 --weight-power 0.5 --weight-min 0.25 --weight-max 2.0 --server-rollout-depth all --max-jobs-per-gpu 1 --max-parallel-jobs 4 --batch-free-mb-per-job 4096 --batch-reserve-free-mb 2048 --local-files-only
```

## Output Files

- `runs/masked_server_attn_pia/dev_ablation.csv`
- `runs/masked_server_attn_pia/structure_ablation.csv`
- `runs/masked_server_attn_pia/heldout_seed43/paired_summary.csv`
- `runs/masked_server_attn_pia/heldout_seed44/paired_summary.csv`
- `runs/masked_server_attn_pia/heldout_summary.csv`
- `runs/masked_server_attn_pia/layer_generalization_override.csv`
- `analysis/mts_heldout_seed43_decision.md`
- `analysis/mts_heldout_seed44_decision.md`
- `analysis/mts_heldout_report.md`
- `analysis/mts_layer_generalization_override.md`

## Conclusion

MTS-PIA gave a small positive result on the development seed, but the improvement did not reproduce on held-out seeds. The later layer check was mixed. The correct conclusion is a staged negative or inconclusive result, not a stable improvement.
