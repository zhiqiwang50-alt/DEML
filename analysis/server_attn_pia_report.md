# Server Attention PIA Report

This is a TinyLlama white-box pilot for server-side attention rollout weighted activation matching.

## Layer Mapping

`capture_prefix_activation(target_layer=L)` returns the output of 0-based TinyLlama block `L`. Therefore `target_layer=17` corresponds to `H^(17)`, and server-side layers start from block `18`.

## Top-k Setting

All requested runs use embedding candidate `--k 1` (top-1). Semantic candidates are controlled separately by `--y`.

## Experimental Setup

- Output root: `runs/server_attn_pia_skytrax150_top1_len896`
- Dataset: `data/skytrax_150.json` (150 Skytrax review prompts)
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Layers: 11, 17, 19
- Methods: B0 original baseline, P1 last-query server rollout, P2 mean-query server rollout, P3 alpha init plus server rollout
- Epochs: 20
- Max token length: 896
- Completed prompt-runs: 1800; failed prompt-runs: 0

## Results

| method | layer | token_acc | delta_vs_B0 | BLEU | n | fail |
|---|---:|---:|---:|---:|---:|---:|
| alpha_init_plus_server_attn | 11 | 0.4172 | -0.0948 | 0.3171 | 150 | 0 |
| original_pia_baseline | 11 | 0.5119 | 0.0000 | 0.4154 | 150 | 0 |
| server_attn_weighted_last | 11 | 0.4478 | -0.0641 | 0.3413 | 150 | 0 |
| server_attn_weighted_mean | 11 | 0.4698 | -0.0421 | 0.3588 | 150 | 0 |
| alpha_init_plus_server_attn | 17 | 0.2798 | -0.2819 | 0.1139 | 150 | 0 |
| original_pia_baseline | 17 | 0.5617 | 0.0000 | 0.3843 | 150 | 0 |
| server_attn_weighted_last | 17 | 0.2849 | -0.2768 | 0.1201 | 150 | 0 |
| server_attn_weighted_mean | 17 | 0.5193 | -0.0424 | 0.3332 | 150 | 0 |
| alpha_init_plus_server_attn | 19 | 0.1132 | -0.4436 | 0.0248 | 150 | 0 |
| original_pia_baseline | 19 | 0.5568 | 0.0000 | 0.3564 | 150 | 0 |
| server_attn_weighted_last | 19 | 0.1071 | -0.4497 | 0.0232 | 150 | 0 |
| server_attn_weighted_mean | 19 | 0.5408 | -0.0160 | 0.3382 | 150 | 0 |

## Main Observation

On the full Skytrax-150 top-1 pilot, the original PIA baseline remains the best row. Server-attention weighting does not produce a stable improvement over B0.

P2 (`server_attn_weighted_mean`) is the least damaging server-attention variant, especially at layer 19, but it is still below the same-layer baseline. P1 (`last_query`) is strongly hurt at layers 17 and 19. P3 shows that adding alpha initialization to the server-attention loss does not rescue the weighted objective in this run.

## Best Observed Row

{
  "stage": "multi_layer",
  "method": "original_pia_baseline",
  "seed": 42,
  "target_layer": 17,
  "epoch": 20,
  "max_token_len": 896,
  "top_k_embedding": 1,
  "top_y_semantic": 10,
  "token_accuracy_mean": 0.5617290196072724,
  "token_accuracy_std": 0.17238485591450106,
  "bleu_mean": 0.3842719531489516,
  "bleu_std": 0.18072972614180996,
  "completed_sample_count": 150,
  "failed_sample_count": 0,
  "metrics_path": "runs/server_attn_pia_skytrax150_top1_len896/multi_layer/original_pia_baseline_seed42_layer17_epoch20_k1/metrics.json"
}

## Caution

These are single-seed epoch-20 pilot results. They support a negative or inconclusive conclusion for the current SAW-PIA weighting design, not a stable improvement claim.