#!/usr/bin/env bash
set -euo pipefail

OUTROOT="${OUTROOT:-runs/attention_guided_pilot}"
DATASET_PATH="${DATASET_PATH:-data/airline.json}"
DATASET_NAME="${DATASET_NAME:-Skytrax-28-pilot}"
EPOCH="${EPOCH:-200}"
STAGE_A_EPOCH="${STAGE_A_EPOCH:-200}"
LR="${LR:-0.1}"
LAMBDA_VOCAB="${LAMBDA_VOCAB:-0.1}"
LAMBDA_ATTN="${LAMBDA_ATTN:-0.1}"
K="${K:-10}"
Y="${Y:-10}"
DATASET_LEN="${DATASET_LEN:-28}"
PARTICIPANTS="${PARTICIPANTS:-4}"
ATTACKER_POSITION="${ATTACKER_POSITION:-4}"
TARGET_LAYER="${TARGET_LAYER:-17}"
GPUS_CSV="${GPUS:-0,1,2,3}"

IFS=',' read -r -a GPUS_ARRAY <<< "$GPUS_CSV"
METHODS=(baseline dummy_init attention_context attention_gradient full)
SEEDS=(42 43 44)
JOBS=()

for method in "${METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    JOBS+=("${method}:${seed}")
  done
done

mkdir -p "$OUTROOT"

run_job() {
  local gpu="$1"
  local job="$2"
  local method="${job%%:*}"
  local seed="${job##*:}"
  local out="$OUTROOT/${method}_seed${seed}"
  mkdir -p "$out"
  if [[ -s "$out/COMPLETE" ]]; then
    echo "$(date -Is) gpu=$gpu skip complete ${method}_seed${seed}"
    return 0
  fi
  echo "$(date -Is) gpu=$gpu start ${method}_seed${seed}"
  set -o pipefail
  CUDA_VISIBLE_DEVICES="$gpu" timeout 6h conda run --no-capture-output -n deml python -u pia_attention_guided.py \
    --mode single \
    --resume \
    --method "$method" \
    --output-root "$OUTROOT" \
    --output-dir "$out" \
    --run-name "${method}_seed${seed}" \
    --dataset-name "$DATASET_NAME" \
    --dataset-path "$DATASET_PATH" \
    --dataset-len "$DATASET_LEN" \
    --participant-number "$PARTICIPANTS" \
    --attacker-position "$ATTACKER_POSITION" \
    --target-layer "$TARGET_LAYER" \
    --epoch "$EPOCH" \
    --stage-a-epoch "$STAGE_A_EPOCH" \
    --seed "$seed" \
    --lr "$LR" \
    --lambda-vocab "$LAMBDA_VOCAB" \
    --lambda-attn "$LAMBDA_ATTN" \
    --k "$K" \
    --y "$Y" 2>&1 | tee "$out/stdout.log"
  test -s "$out/metrics.json"
  test -s "$out/predictions.jsonl"
  test -s "$out/attention_stats.json"
  test -s "$out/COMPLETE"
  echo "$(date -Is) gpu=$gpu done ${method}_seed${seed}"
}

worker() {
  local worker_id="$1"
  local gpu="${GPUS_ARRAY[$worker_id]}"
  local job_count="${#JOBS[@]}"
  local gpu_count="${#GPUS_ARRAY[@]}"
  local idx
  for ((idx = worker_id; idx < job_count; idx += gpu_count)); do
    run_job "$gpu" "${JOBS[$idx]}"
  done
}

for worker_id in "${!GPUS_ARRAY[@]}"; do
  worker "$worker_id" &
done

wait

conda run -n deml python pia_attention_guided.py --mode summarize --output-root "$OUTROOT"
