#!/usr/bin/env bash
set -euo pipefail

RESUME=""
if [[ "${1:-}" == "--resume" ]]; then
  RESUME="--resume"
fi

OUT_DIR="runs/tinyllama_full/smoke"
mkdir -p "${OUT_DIR}"

conda run -n deml python -c 'import os, torch; print("conda_env=" + str(os.environ.get("CONDA_DEFAULT_ENV"))); print("cuda_available=" + str(torch.cuda.is_available())); print("cuda_device_count=" + str(torch.cuda.device_count()))' 2>&1 | tee "${OUT_DIR}/env.log"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" timeout 6h conda run -n deml python pia_tinyllama.py \
  --mode smoke \
  ${RESUME} \
  --dataset-name Skytrax \
  --dataset-path data/airline.json \
  --dataset-len 1 \
  --target-layer 3 \
  --epoch 50 \
  --lr 0.1 \
  --lambda-coeff 0.1 \
  --k 5 \
  --y 5 \
  2>&1 | tee "${OUT_DIR}/stdout.log"

test -s "${OUT_DIR}/metrics.json"
test -s "${OUT_DIR}/predictions.jsonl"
