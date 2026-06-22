#!/usr/bin/env bash
set -euo pipefail

mkdir -p runs/tinyllama_batch runs/gpt2_batch runs/bert_batch

conda run -n deml python -c 'import os, torch; print("conda_env=" + str(os.environ.get("CONDA_DEFAULT_ENV"))); print("cuda_available=" + str(torch.cuda.is_available())); print("cuda_device_count=" + str(torch.cuda.device_count()))' 2>&1 | tee "runs/batch_env.log"

CUDA_VISIBLE_DEVICES=0 timeout 30m conda run -n deml python invert_small_models.py \
  --base-model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset-path experiments/prompts_en.json \
  --dataset-len 5 \
  --num-invert-layers 10 \
  --epoch 200 \
  --top-k-cos 5 \
  --top-k-ppl 5 \
  --output-dir runs/tinyllama_batch \
  2>&1 | tee "runs/tinyllama_batch/run.log"

CUDA_VISIBLE_DEVICES=1 timeout 30m conda run -n deml python invert_small_models.py \
  --base-model-name openai-community/gpt2 \
  --dataset-path experiments/prompts_en.json \
  --dataset-len 5 \
  --num-invert-layers 8 \
  --epoch 200 \
  --top-k-cos 5 \
  --top-k-ppl 5 \
  --output-dir runs/gpt2_batch \
  2>&1 | tee "runs/gpt2_batch/run.log"

CUDA_VISIBLE_DEVICES=2 timeout 30m conda run -n deml python invert_small_models.py \
  --base-model-name google-bert/bert-base-uncased \
  --dataset-path experiments/prompts_en.json \
  --dataset-len 5 \
  --num-invert-layers 8 \
  --epoch 200 \
  --top-k-cos 5 \
  --disable-perplexity \
  --output-dir runs/bert_batch \
  2>&1 | tee "runs/bert_batch/run.log"

test -s runs/tinyllama_batch/results.jsonl
test -s runs/gpt2_batch/results.jsonl
test -s runs/bert_batch/results.jsonl
