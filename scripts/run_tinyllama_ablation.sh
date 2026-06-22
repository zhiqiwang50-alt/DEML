#!/usr/bin/env bash
set -euo pipefail

RESUME=""
if [[ "${1:-}" == "--resume" ]]; then
  RESUME="--resume"
fi

mkdir -p runs/tinyllama_full/ablations
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" timeout 6h conda run -n deml python pia_tinyllama.py \
  --mode ablation \
  ${RESUME} \
  --dataset-name Skytrax \
  --dataset-path data/airline.json \
  --dataset-len "${DATASET_LEN:-150}" \
  --epoch "${EPOCH:-2000}" \
  --lr 0.1 \
  --lambda-coeff 0.1 \
  --k 10 \
  --y 10 \
  2>&1 | tee "runs/tinyllama_full/ablations/stdout.log"
