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

- Output root: `runs/masked_server_attn_pia`
- Dataset names: Skytrax-150, Skytrax-28
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Layers: 11, 17, 19
- Epochs: 1, 100
- Embedding top-k: 1, 10
- Semantic top-y: 10
- Max token length: 896
- Completed prompt-runs: 1720; failed prompt-runs: 0

## Results

| method | layer | token_acc | delta_vs_B0 | BLEU | n | fail |
|---|---:|---:|---:|---:|---:|---:|
| mts_last_window | 11 | 0.9772 | 0.0226 | 0.9672 | 28 | 0 |
| original_pia_baseline | 11 | 0.9546 | 0.0000 | 0.9292 | 28 | 0 |
| variable_only_uniform | 11 | 0.9277 | -0.0270 | 0.9003 | 28 | 0 |
| mts_last_window | 17 | 0.9952 | -0.0048 | 0.9904 | 28 | 0 |
| mts_last_window | 17 | 0.9963 | -0.0037 | 0.9925 | 28 | 0 |
| mts_last_window | 17 | 0.9986 | -0.0014 | 0.9963 | 28 | 0 |
| mts_last_window | 17 | 0.9910 | -0.0090 | 0.9828 | 28 | 0 |
| mts_last_window | 17 | 0.9975 | -0.0025 | 0.9944 | 28 | 0 |
| mts_last_window | 17 | 0.9891 | -0.0109 | 0.9788 | 28 | 0 |
| mts_last_window | 17 | 0.9944 | -0.0056 | 0.9876 | 28 | 0 |
| mts_last_window | 17 | 0.9951 | -0.0049 | 0.9894 | 28 | 0 |
| mts_last_window | 17 | 0.9920 | -0.0080 | 0.9858 | 28 | 0 |
| mts_last_window | 17 | 0.9951 | -0.0049 | 0.9903 | 28 | 0 |
| mts_last_window | 17 | 0.9882 | -0.0118 | 0.9778 | 28 | 0 |
| mts_last_window | 17 | 0.9896 | -0.0104 | 0.9800 | 28 | 0 |
| mts_last_window | 17 | 0.9940 | -0.0060 | 0.9873 | 28 | 0 |
| mts_last_window | 17 | 0.9968 | -0.0032 | 0.9930 | 28 | 0 |
| mts_last_window | 17 | 0.9899 | -0.0101 | 0.9829 | 28 | 0 |
| mts_last_window | 17 | 0.9952 | -0.0048 | 0.9913 | 28 | 0 |
| mts_last_window | 17 | 0.9957 | -0.0043 | 0.9909 | 28 | 0 |
| mts_last_window | 17 | 0.9933 | -0.0067 | 0.9865 | 28 | 0 |
| mts_last_window | 17 | 0.9895 | -0.0105 | 0.9850 | 28 | 0 |
| mts_last_window | 17 | 0.9878 | -0.0122 | 0.9811 | 28 | 0 |
| mts_last_window | 17 | 0.9925 | -0.0075 | 0.9867 | 2 | 0 |
| mts_last_window | 17 | 0.9973 | -0.0027 | 0.9940 | 28 | 0 |
| mts_last_window | 17 | 0.9959 | -0.0041 | 0.9909 | 28 | 0 |
| mts_last_window | 17 | 0.9986 | -0.0014 | 0.9963 | 28 | 0 |
| mts_last_window | 17 | 0.9884 | -0.0116 | 0.9796 | 28 | 0 |
| mts_mean_query | 17 | 0.9665 | -0.0335 | 0.9548 | 28 | 0 |
| mts_mean_query | 17 | 0.9665 | -0.0335 | 0.9548 | 28 | 0 |
| mts_mean_query | 17 | 0.9949 | -0.0051 | 0.9887 | 28 | 0 |
| mts_mean_query | 17 | 0.9822 | -0.0178 | 0.9693 | 28 | 0 |
| mts_mean_query | 17 | 0.9947 | -0.0053 | 0.9896 | 28 | 0 |
| mts_mean_query | 17 | 0.9954 | -0.0046 | 0.9897 | 28 | 0 |
| mts_mean_query | 17 | 0.9911 | -0.0089 | 0.9833 | 28 | 0 |
| mts_mean_query | 17 | 0.9911 | -0.0089 | 0.9833 | 28 | 0 |
| mts_mean_query | 17 | 0.9750 | -0.0250 | 0.9645 | 28 | 0 |
| mts_mean_query | 17 | 0.9962 | -0.0038 | 0.9915 | 28 | 0 |
| mts_mean_query | 17 | 0.9964 | -0.0036 | 0.9917 | 28 | 0 |
| mts_mean_query | 17 | 0.9939 | -0.0061 | 0.9866 | 28 | 0 |
| mts_mean_query | 17 | 0.9906 | -0.0094 | 0.9832 | 28 | 0 |
| mts_mean_query | 17 | 0.9906 | -0.0094 | 0.9832 | 28 | 0 |
| mts_mean_query | 17 | 0.9972 | -0.0028 | 0.9933 | 28 | 0 |
| mts_mean_query | 17 | 0.9983 | -0.0017 | 0.9960 | 28 | 0 |
| mts_mean_query | 17 | 0.9937 | -0.0063 | 0.9870 | 28 | 0 |
| mts_mean_query | 17 | 0.9853 | -0.0147 | 0.9744 | 28 | 0 |
| mts_mean_query | 17 | 0.9817 | -0.0183 | 0.9680 | 28 | 0 |
| mts_mean_query | 17 | 0.9948 | -0.0052 | 0.9900 | 28 | 0 |
| mts_mean_query | 17 | 1.0000 | 0.0000 | 1.0000 | 2 | 0 |
| mts_mean_query | 17 | 0.9976 | -0.0024 | 0.9945 | 28 | 0 |
| mts_mean_query | 17 | 0.9846 | -0.0154 | 0.9739 | 28 | 0 |
| mts_mean_query | 17 | 0.9983 | -0.0017 | 0.9960 | 28 | 0 |
| mts_mean_query | 17 | 0.9988 | -0.0012 | 0.9973 | 28 | 0 |
| original_pia_baseline | 17 | 0.0615 | -0.9385 | 0.0074 | 1 | 0 |
| original_pia_baseline | 17 | 0.9847 | -0.0153 | 0.9742 | 28 | 0 |
| original_pia_baseline | 17 | 0.9886 | -0.0114 | 0.9795 | 28 | 0 |
| original_pia_baseline | 17 | 0.9966 | -0.0034 | 0.9922 | 28 | 0 |
| original_pia_baseline | 17 | 1.0000 | 0.0000 | 1.0000 | 2 | 0 |
| server_attn_last_raw | 17 | 0.9500 | -0.0500 | 0.9140 | 28 | 0 |
| server_attn_last_raw | 17 | 0.9701 | -0.0299 | 0.9592 | 2 | 0 |
| server_attn_weighted_mean | 17 | 0.0385 | -0.9615 | 0.0052 | 1 | 0 |
| variable_only_uniform | 17 | 0.9958 | -0.0042 | 0.9907 | 28 | 0 |
| variable_only_uniform | 17 | 0.9970 | -0.0030 | 0.9946 | 28 | 0 |
| variable_only_uniform | 17 | 0.9960 | -0.0040 | 0.9916 | 28 | 0 |
| variable_only_uniform | 17 | 0.9920 | -0.0080 | 0.9858 | 2 | 0 |
| mts_last_window | 19 | 0.9944 | -0.0016 | 0.9881 | 28 | 0 |
| original_pia_baseline | 19 | 0.9960 | 0.0000 | 0.9917 | 28 | 0 |
| variable_only_uniform | 19 | 0.9990 | 0.0030 | 0.9974 | 28 | 0 |

## Weight Audit

| method | source | variable_n | mean_w | std_w | min_w | max_w | raw_bos | bos_w_max | last_valid_w | fixed_sum_max | query_positions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.0972 | 0.8125 | 1.2500 | 0.9695 | 0.0000 | 1.0394 | 0.0000 | 127..134 (n=8) |
| original_pia_baseline | all_valid_uniform | NA | 1.0000 | 0.0000 | 1.0000 | 1.0000 | NA | 1.0000 | 1.0000 | 1.0000 | NA |
| variable_only_uniform | variable_uniform | 161.2 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | NA | 0.0000 | 1.0000 | 0.0000 | NA |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.0743 | 0.8490 | 1.2500 | 0.7661 | 0.0000 | 1.2036 | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.0767 | 0.8489 | 1.4563 | 0.7661 | 0.0000 | 1.2152 | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.1166 | 0.8125 | 1.2500 | 0.7661 | 0.0000 | 1.2459 | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.1727 | 0.8125 | 1.7500 | 0.7661 | 0.0000 | 1.5263 | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.1664 | 0.8125 | 1.2500 | 0.7661 | 0.0000 | 1.2500 | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.2695 | 0.8125 | 1.7500 | 0.7661 | 0.0000 | 1.7164 | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.1486 | 0.6980 | 1.5000 | 0.7661 | 0.0000 | 1.4073 | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.1534 | 0.6978 | 1.9127 | 0.7661 | 0.0000 | 1.4303 | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.2331 | 0.6250 | 1.5000 | 0.7661 | 0.0000 | 1.4917 | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.3453 | 0.6250 | 2.5000 | 0.7661 | 0.0000 | 2.0527 | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.3329 | 0.6250 | 1.5000 | 0.7661 | 0.0000 | 1.5000 | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.5389 | 0.6250 | 2.5000 | 0.7661 | 0.0000 | 2.4328 | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.2230 | 0.5470 | 1.7500 | 0.7661 | 0.0000 | 1.6109 | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.2300 | 0.5467 | 2.3690 | 0.7661 | 0.0000 | 1.6455 | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.3497 | 0.4375 | 1.7500 | 0.7661 | 0.0000 | 1.7376 | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.5180 | 0.4375 | 3.2500 | 0.7661 | 0.0000 | 2.5790 | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.4993 | 0.4375 | 1.7500 | 0.7661 | 0.0000 | 1.7500 | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.8084 | 0.4375 | 3.2500 | 0.7661 | 0.0000 | 3.1492 | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.1166 | 0.8125 | 1.2500 | 0.7661 | 0.0000 | 1.2459 | 0.0000 | 79..86 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.1166 | 0.8125 | 1.2500 | 0.7661 | 0.0000 | 1.2459 | 0.0000 | 179..186 (n=8) |
| mts_last_window | last_window_mean | 129.5 | 1.0000 | 0.3486 | 0.6250 | 2.5000 | 0.7662 | 0.0000 | NA | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.1043 | 0.8125 | 1.2500 | 0.9811 | 0.0000 | 0.8148 | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.1093 | 0.8125 | 1.2500 | 0.8364 | 0.0000 | 1.1135 | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.1166 | 0.8125 | 1.2500 | 0.7661 | 0.0000 | 1.2459 | 0.0000 | 127..134 (n=8) |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.1249 | 0.8125 | 1.2500 | 0.5004 | 0.0000 | 1.2500 | 0.0000 | 127..134 (n=8) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.0283 | 0.9434 | 1.1467 | 0.8000 | 0.0000 | 0.9570 | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.0283 | 0.9434 | 1.1467 | 0.8000 | 0.0000 | 0.9570 | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.0589 | 0.8973 | 1.2500 | 0.8000 | 0.0000 | 0.9194 | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.0594 | 0.8972 | 1.3698 | 0.8000 | 0.0000 | 0.9193 | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.1053 | 0.8357 | 1.2500 | 0.8000 | 0.0000 | 0.8632 | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.1313 | 0.8310 | 1.7500 | 0.8000 | 0.0000 | 0.8590 | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.0567 | 0.8867 | 1.2935 | 0.8000 | 0.0000 | 0.9139 | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.0567 | 0.8867 | 1.2935 | 0.8000 | 0.0000 | 0.9139 | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.1178 | 0.7946 | 1.5000 | 0.8000 | 0.0000 | 0.8387 | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.1188 | 0.7943 | 1.7396 | 0.8000 | 0.0000 | 0.8386 | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.2107 | 0.6714 | 1.5000 | 0.8000 | 0.0000 | 0.7263 | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.2627 | 0.6620 | 2.5000 | 0.8000 | 0.0000 | 0.7179 | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.0850 | 0.8301 | 1.4402 | 0.8000 | 0.0000 | 0.8709 | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.0850 | 0.8301 | 1.4402 | 0.8000 | 0.0000 | 0.8709 | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.1766 | 0.6919 | 1.7500 | 0.8000 | 0.0000 | 0.7581 | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.1782 | 0.6915 | 2.1093 | 0.8000 | 0.0000 | 0.7578 | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.3160 | 0.5072 | 1.7500 | 0.8000 | 0.0000 | 0.5895 | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.3940 | 0.4930 | 3.2500 | 0.8000 | 0.0000 | 0.5769 | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.1094 | 0.8173 | 1.8062 | 0.5335 | 0.0000 | 0.8716 | 0.0000 | 1..86 (n=86) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.1094 | 0.8173 | 1.8062 | 0.5335 | 0.0000 | 0.8716 | 0.0000 | 1..186 (n=186) |
| mts_mean_query | mean_query | 129.5 | 1.0000 | 0.1171 | 0.8137 | 1.4337 | 0.8002 | 0.0000 | NA | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.4865 | 0.4375 | 3.2500 | 0.9883 | 0.0000 | 0.4375 | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.3336 | 0.4375 | 3.0608 | 0.8686 | 0.0000 | 0.5212 | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.1782 | 0.6915 | 2.1093 | 0.8000 | 0.0000 | 0.7578 | 0.0000 | 1..134 (n=134) |
| mts_mean_query | mean_query | 161.2 | 1.0000 | 0.1094 | 0.8173 | 1.8062 | 0.5335 | 0.0000 | 0.8716 | 0.0000 | 1..134 (n=134) |
| original_pia_baseline | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| original_pia_baseline | all_valid_uniform | NA | 1.0000 | 0.0000 | 1.0000 | 1.0000 | NA | 1.0000 | 1.0000 | 1.0000 | NA |
| original_pia_baseline | all_valid_uniform | NA | 1.0000 | 0.0000 | 1.0000 | 1.0000 | NA | 1.0000 | 1.0000 | 1.0000 | NA |
| original_pia_baseline | all_valid_uniform | NA | 1.0000 | 0.0000 | 1.0000 | 1.0000 | NA | 1.0000 | 1.0000 | 1.0000 | NA |
| original_pia_baseline | all_valid_uniform | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| server_attn_last_raw | last_query | NA | 1.0000 | 8.9470 | 0.0497 | 363.4860 | 0.7369 | 363.4860 | 13.8229 | 363.4860 | NA |
| server_attn_last_raw | last_query | NA | 1.0000 | 8.2588 | 0.0499 | 96.1469 | NA | NA | NA | NA | NA |
| server_attn_weighted_mean | mean_query | NA | 1.0000 | 9.1929 | 0.0793 | 105.8066 | NA | NA | NA | NA | NA |
| variable_only_uniform | variable_uniform | 161.2 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | NA | 0.0000 | 1.0000 | 0.0000 | NA |
| variable_only_uniform | variable_uniform | 161.2 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | NA | 0.0000 | 1.0000 | 0.0000 | NA |
| variable_only_uniform | variable_uniform | 161.2 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | NA | 0.0000 | 1.0000 | 0.0000 | NA |
| variable_only_uniform | variable_uniform | 129.5 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | NA | 0.0000 | NA | 0.0000 | NA |
| mts_last_window | last_window_mean | 161.2 | 1.0000 | 0.1249 | 0.8125 | 1.2500 | 0.5004 | 0.0000 | 1.2500 | 0.0000 | 127..134 (n=8) |
| original_pia_baseline | all_valid_uniform | NA | 1.0000 | 0.0000 | 1.0000 | 1.0000 | NA | 1.0000 | 1.0000 | 1.0000 | NA |
| variable_only_uniform | variable_uniform | 161.2 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | NA | 0.0000 | 1.0000 | 0.0000 | NA |

## Main Observation

At least one MTS row is above the same-layer B0 row in this summary: mts_last_window Tied MTS rows: mts_mean_query.

Interpret this as a pilot result only. Matching or exceeding B0 on a two-sample smoke run is useful for debugging, but it is not evidence of stable improvement.

## Best Observed Row

{
  "stage": "smoke",
  "method": "mts_mean_query",
  "seed": 42,
  "target_layer": 17,
  "dataset_name": "Skytrax-28",
  "dataset_path": "data/airline.json",
  "epoch": 100,
  "max_token_len": 896,
  "top_k_embedding": 10,
  "top_y_semantic": 10,
  "residual_alpha_rho": null,
  "token_accuracy_mean": 1.0,
  "token_accuracy_std": 0.0,
  "bleu_mean": 1.0,
  "bleu_std": 0.0,
  "completed_sample_count": 2,
  "failed_sample_count": 0,
  "metrics_path": "runs/masked_server_attn_pia/smoke/multi_layer/mts_mean_query_seed42_layer17_epoch100_k10/metrics.json"
}

## Caution

Do not claim stable improvement unless MTS beats both B0 and B0V under the planned multi-seed and multi-layer checks.