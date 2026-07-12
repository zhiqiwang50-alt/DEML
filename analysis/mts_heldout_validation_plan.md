# MTS-PIA Held-Out Validation Plan

Seed42 is treated only as the development set used for parameter selection. Seeds 43 and 44 are held-out validation seeds.

## Frozen Candidates

- P2: {'method': 'mts_mean_query', 'beta': 0.75, 'weight_power': 0.5, 'weight_min': 0.25, 'weight_max': 4.0, 'residual_rollout': True, 'server_rollout_depth': 'last2'}
- P3: {'method': 'mts_last_window', 'beta': 0.25, 'weight_power': 0.5, 'weight_min': 0.25, 'weight_max': 2.0, 'residual_rollout': True, 'server_rollout_depth': 'all'}

## Pre-Run Checks

- dev_ablation.csv rows: 39
- structure_ablation.csv rows: 8
- P2 frozen structure row found: True
- P3 frozen structure row found: True
- leakage verification passes: True
- B0 remains all-valid uniform loss via `uniform_all_token_activation_loss`.
- B0V remains variable-only uniform loss via `weighted_variable_activation_loss(..., weights=None)`.
- P2/P3 keep server-side attention rollout, variable mask, tempering, clipping, beta mixing, and activation calibration.
- The attack API does not receive prompt, original_ids, original_tokens, or token_ids.
- Last valid position is recorded as a sequence boundary, not as EOS.

No parameter search is allowed on seed43 or seed44.
