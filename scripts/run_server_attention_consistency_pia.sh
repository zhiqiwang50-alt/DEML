#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python pia_server_attention_consistency_pia.py \
  --output-root runs/server_attention_consistency_pia \
  --dataset-name Skytrax-28 \
  --dataset-path data/airline.json \
  --dataset-len 28 \
  --prompt-split-seed 20260705 \
  --participant-number 4 \
  --attacker-position 4 \
  --target-layer 17 \
  --epoch 100 \
  --k 10 \
  --y 10 \
  --max-token-len 896 \
  --local-files-only \
  "$@"
