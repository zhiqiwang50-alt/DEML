#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="runs/bert_smoke"
mkdir -p "${OUT_DIR}"

conda run -n deml python -c 'import os, torch; print("conda_env=" + str(os.environ.get("CONDA_DEFAULT_ENV"))); print("cuda_available=" + str(torch.cuda.is_available())); print("cuda_device_count=" + str(torch.cuda.device_count()))' 2>&1 | tee "${OUT_DIR}/env.log"

CUDA_VISIBLE_DEVICES=2 timeout 30m conda run -n deml python invert_small_models.py \
  --base-model-name google-bert/bert-base-uncased \
  --dataset-path experiments/prompts_en.json \
  --dataset-len 1 \
  --num-invert-layers 3 \
  --epoch 50 \
  --top-k-cos 5 \
  --disable-perplexity \
  --output-dir "${OUT_DIR}" \
  2>&1 | tee "${OUT_DIR}/run.log"

test -s "${OUT_DIR}/results.jsonl"
