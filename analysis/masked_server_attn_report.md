# Masked Server Attention PIA Report

The earlier `pia_masked_server_attn_pia.py` branch was effectively SAW-PIA plus resource-aware batching. This version completes the masked and tempered server-attention (MTS) body.

## Layer Mapping

`capture_prefix_activation(target_layer=L)` returns the output of 0-based TinyLlama block `L`. Therefore `target_layer=17` corresponds to `H^(17)`, and server-side layers start from block `18`.

## Method Definitions

- B0 / `original_pia_baseline`: original PIA activation matching over all valid tokens.
- B0V / `variable_only_uniform`: variable-token-only uniform activation matching.
- P1 / `server_attn_last_raw`: raw last-query server rollout negative control.
- P2 / `mts_mean_query`: variable-mask MTS weights from mean query attention.
- P3 / `mts_last_window`: variable-mask MTS weights from the last query window.

P2/P3 exclude fixed public positions and any token-id special positions only when token ids are available. In the attack path token ids are not passed, so the audit reports `last_valid_position` rather than claiming EOS detection.

## Experimental Setup

- Output root: `runs/p4lwr_last_window_residual_top1`
- Dataset names: Skytrax-150
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Layers: 11, 17, 19
- Epochs: 100, 5
- Embedding top-k: 1
- Semantic top-y: 0
- Max token length: 896
- Completed prompt-runs: 1352; failed prompt-runs: 0

## Results

| method | layer | token_acc | delta_vs_B0 | BLEU | n | fail |
|---|---:|---:|---:|---:|---:|---:|
| attn_last_window_linear_residual_schedule | 11 | 0.7624 | -0.0404 | 0.6027 | 150 | 0 |
| original_pia_baseline | 11 | 0.8028 | 0.0000 | 0.6412 | 150 | 0 |
| variable_only_uniform | 11 | 0.7836 | -0.0192 | 0.6145 | 150 | 0 |
| attn_last_window_linear_residual_schedule | 17 | 0.8309 | -0.0039 | 0.6384 | 150 | 0 |
| attn_last_window_linear_residual_schedule | 17 | 0.0000 | -0.8348 | 0.0014 | 2 | 0 |
| original_pia_baseline | 17 | 0.8348 | 0.0000 | 0.6445 | 150 | 0 |
| variable_only_uniform | 17 | 0.8425 | 0.0078 | 0.6530 | 150 | 0 |
| attn_last_window_linear_residual_schedule | 19 | 0.8388 | 0.0013 | 0.6474 | 150 | 0 |
| original_pia_baseline | 19 | 0.8375 | 0.0000 | 0.6451 | 150 | 0 |
| variable_only_uniform | 19 | 0.8358 | -0.0017 | 0.6435 | 150 | 0 |

## Weight Audit

| method | source | variable_n | mean_w | std_w | min_w | max_w | raw_bos | bos_w_max | last_valid_w | fixed_sum_max | query_positions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| attn_last_window_linear_residual_schedule | last_window_mean | 169.1 | 1.0000 | 0.1974 | 0.5972 | 1.5000 | 0.9714 | NA | NA | NA | 123..130 (n=8) |
| original_pia_baseline | all_valid_uniform | NA | 1.0000 | 0.0000 | 1.0000 | 1.0000 | NA | 1.0000 | 1.0000 | 1.0000 | NA |
| variable_only_uniform | variable_uniform | 169.1 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | NA | 0.0000 | 1.0000 | 0.0000 | NA |
| attn_last_window_linear_residual_schedule | last_window_mean | 169.1 | 1.0000 | 0.2325 | 0.6129 | 1.5000 | 0.7747 | NA | NA | NA | 123..130 (n=8) |
| attn_last_window_linear_residual_schedule | last_window_mean | 129.5 | 1.0000 | 0.2247 | 0.6150 | 1.5000 | 0.7662 | NA | NA | NA | 127..134 (n=8) |
| original_pia_baseline | all_valid_uniform | NA | 1.0000 | 0.0000 | 1.0000 | 1.0000 | NA | 1.0000 | 1.0000 | 1.0000 | NA |
| variable_only_uniform | variable_uniform | 169.1 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | NA | 0.0000 | 1.0000 | 0.0000 | NA |
| attn_last_window_linear_residual_schedule | last_window_mean | 169.1 | 1.0000 | 0.2394 | 0.6329 | 1.5000 | 0.5114 | NA | NA | NA | 123..130 (n=8) |
| original_pia_baseline | all_valid_uniform | NA | 1.0000 | 0.0000 | 1.0000 | 1.0000 | NA | 1.0000 | 1.0000 | 1.0000 | NA |
| variable_only_uniform | variable_uniform | 169.1 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | NA | 0.0000 | 1.0000 | 0.0000 | NA |

## Main Observation

No MTS row is above the same-layer B0 row in this summary.

Interpret this as a pilot result only. Matching or exceeding B0 on a two-sample smoke run is useful for debugging, but it is not evidence of stable improvement.

## Best Observed Row

{
  "stage": "seed42_skytrax150_layers11_17_19",
  "method": "variable_only_uniform",
  "seed": 42,
  "target_layer": 17,
  "dataset_name": "Skytrax-150",
  "dataset_path": "data/skytrax_150.json",
  "epoch": 100,
  "max_token_len": 896,
  "top_k_embedding": 1,
  "top_y_semantic": 0,
  "residual_alpha_rho": 0.6,
  "token_accuracy_mean": 0.8425440949105719,
  "token_accuracy_std": 0.05849227427365609,
  "bleu_mean": 0.6530267294944091,
  "bleu_std": 0.10150189933139517,
  "completed_sample_count": 150,
  "failed_sample_count": 0,
  "metrics_path": "runs/p4lwr_last_window_residual_top1/seed42_skytrax150_layers11_17_19/B0V_seed42_layer17_epoch100_k1_y0/metrics.json"
}

## Caution

Do not claim stable improvement unless MTS beats both B0 and B0V under the planned multi-seed and multi-layer checks.