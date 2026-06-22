#!/usr/bin/env bash
set -euo pipefail

RESUME=""
if [[ "${1:-}" == "--resume" ]]; then
  RESUME="--resume"
fi

mkdir -p runs/tinyllama_full/hparams

for LAMBDA in 0 0.0005 0.001 0.0015 0.002 0.0025 0.005 0.01; do
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" timeout 6h conda run -n deml python pia_tinyllama.py --mode smoke ${RESUME} --output-dir "runs/tinyllama_full/hparams/lambda_${LAMBDA}" --dataset-len "${DATASET_LEN:-5}" --epoch "${EPOCH:-500}" --target-layer 17 --lambda-coeff "${LAMBDA}" 2>&1 | tee "runs/tinyllama_full/hparams/lambda_${LAMBDA}.log"
done

for LR in 0.01 0.02 0.05 0.1 0.2 0.5 1.0; do
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" timeout 6h conda run -n deml python pia_tinyllama.py --mode smoke ${RESUME} --output-dir "runs/tinyllama_full/hparams/lr_${LR}" --dataset-len "${DATASET_LEN:-5}" --epoch "${EPOCH:-500}" --target-layer 17 --lr "${LR}" 2>&1 | tee "runs/tinyllama_full/hparams/lr_${LR}.log"
done

for ITER in 500 1000 2000 3000 5000; do
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" timeout 6h conda run -n deml python pia_tinyllama.py --mode smoke ${RESUME} --output-dir "runs/tinyllama_full/hparams/iter_${ITER}" --dataset-len "${DATASET_LEN:-5}" --epoch "${ITER}" --target-layer 17 2>&1 | tee "runs/tinyllama_full/hparams/iter_${ITER}.log"
done

for K in 0 10 20; do
  for Y in 0 10 20; do
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" timeout 6h conda run -n deml python pia_tinyllama.py --mode smoke ${RESUME} --output-dir "runs/tinyllama_full/hparams/k_${K}_y_${Y}" --dataset-len "${DATASET_LEN:-5}" --epoch "${EPOCH:-500}" --target-layer 17 --k "${K}" --y "${Y}" 2>&1 | tee "runs/tinyllama_full/hparams/k_${K}_y_${Y}.log"
  done
done
