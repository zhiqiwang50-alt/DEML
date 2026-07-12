#!/usr/bin/env bash
set -uo pipefail

ROOT="runs/alpha_gm_pilot"
DATASET_NAME="Skytrax-28-pilot"
DATASET_PATH="data/airline.json"
MODE="all"
RESUME="--resume"
MAX_JOBS=4
GPUS_CSV="0,1,2,3"
SEED=42
DATASET_LEN=24
SMOKE_LEN=2
LR=0.1
LAMBDA_VOCAB=0.1
LAMBDA_DUMMY=0.1
LAMBDA_CONTEXT=0.1
K=10
Y=10
GAMMA=0.3
ETA=0.5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --output-root) ROOT="$2"; shift 2 ;;
    --dataset-path) DATASET_PATH="$2"; shift 2 ;;
    --dataset-name) DATASET_NAME="$2"; shift 2 ;;
    --dataset-len) DATASET_LEN="$2"; shift 2 ;;
    --gamma) GAMMA="$2"; shift 2 ;;
    --eta) ETA="$2"; shift 2 ;;
    --max-jobs) MAX_JOBS="$2"; shift 2 ;;
    --gpus) GPUS_CSV="$2"; shift 2 ;;
    --no-resume) RESUME=""; shift ;;
    --resume) RESUME="--resume"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$ROOT"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "no GPUs configured" >&2
  exit 2
fi

norm_float() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  echo "$value"
}

run_config() {
  local gpu="$1"
  local stage="$2"
  local method="$3"
  local seed="$4"
  local layer="$5"
  local epoch="$6"
  local stage_a_epoch="$7"
  local gamma="$8"
  local eta="$9"
  local dataset_len="${10}"
  local gtag etag out status
  gtag="$(norm_float "$gamma")"
  etag="$(norm_float "$eta")"
  out="$ROOT/$stage/${method}_layer${layer}_epoch${epoch}_seed${seed}_g${gtag}_eta${etag}"
  mkdir -p "$out"
  echo "[alpha-gm] gpu=$gpu stage=$stage method=$method seed=$seed layer=$layer epoch=$epoch gamma=$gamma eta=$eta out=$out"
  CUDA_VISIBLE_DEVICES="$gpu" timeout 6h conda run --no-capture-output -n deml python -u pia_alpha_gm.py \
    --mode single \
    $RESUME \
    --output-root "$ROOT" \
    --output-dir "$out" \
    --run-name "$(basename "$out")" \
    --dataset-name "$DATASET_NAME" \
    --dataset-path "$DATASET_PATH" \
    --dataset-len "$dataset_len" \
    --method "$method" \
    --seed "$seed" \
    --participant-number 4 \
    --attacker-position 4 \
    --target-layer "$layer" \
    --epoch "$epoch" \
    --stage-a-epoch "$stage_a_epoch" \
    --lr "$LR" \
    --lambda-vocab "$LAMBDA_VOCAB" \
    --lambda-dummy "$LAMBDA_DUMMY" \
    --lambda-context "$LAMBDA_CONTEXT" \
    --k "$K" \
    --y "$Y" \
    --gamma "$gamma" \
    --eta "$eta" \
    2>&1 | tee "$out/stdout.log"
  status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    echo "$status" > "$out/FAILED"
    echo "[alpha-gm] failed status=$status out=$out" >&2
  fi
  return "$status"
}

TASKS=()

queue_config() {
  TASKS+=("$*")
}

run_queued() {
  local worker_count idx gpu status
  worker_count="$MAX_JOBS"
  if [[ "$worker_count" -gt "${#GPUS[@]}" ]]; then
    worker_count="${#GPUS[@]}"
  fi
  if [[ "$worker_count" -lt 1 ]]; then
    worker_count=1
  fi
  for ((idx = 0; idx < worker_count; idx++)); do
    gpu="${GPUS[$idx]}"
    (
      local task_i task stage method seed layer epoch stage_a_epoch gamma eta dataset_len
      for ((task_i = idx; task_i < ${#TASKS[@]}; task_i += worker_count)); do
        task="${TASKS[$task_i]}"
        read -r stage method seed layer epoch stage_a_epoch gamma eta dataset_len <<< "$task"
        run_config "$gpu" "$stage" "$method" "$seed" "$layer" "$epoch" "$stage_a_epoch" "$gamma" "$eta" "$dataset_len" || true
      done
    ) &
  done
  status=0
  for job in $(jobs -p); do
    wait "$job" || status=1
  done
  TASKS=()
  return "$status"
}

run_verify() {
  local out="$ROOT/leakage_verification"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" conda run --no-capture-output -n deml python -u pia_alpha_gm.py \
    --mode verify \
    --output-root "$ROOT" \
    --dataset-name "$DATASET_NAME" \
    --dataset-path "$DATASET_PATH" \
    --dataset-len 1 \
    --participant-number 4 \
    --attacker-position 4 \
    --target-layer 17 \
    --epoch 20 \
    --stage-a-epoch 20 \
    --lr "$LR" \
    --lambda-vocab "$LAMBDA_VOCAB" \
    --lambda-dummy "$LAMBDA_DUMMY" \
    --lambda-context "$LAMBDA_CONTEXT" \
    --k "$K" \
    --y "$Y" \
    --gamma "$GAMMA" \
    --eta "$ETA" \
    2>&1 | tee "$out/stdout.log"
}

run_smoke() {
  local methods=(
    baseline
    dummy_init_existing
    attention_context_existing
    alpha_nn_init_direct
    alpha_nn_init_residual
    alpha_nn_gradmatch
    alpha_nn_gradmatch_context
  )
  for method in "${methods[@]}"; do
    queue_config "smoke $method $SEED 17 100 100 $GAMMA $ETA $SMOKE_LEN"
  done
  run_queued
}

run_sweep() {
  local gammas=(0.0 0.1 0.3 0.5 1.0)
  local etas=(0.0 0.25 0.5 0.75 1.0)
  local gamma eta
  for gamma in "${gammas[@]}"; do
    for eta in "${etas[@]}"; do
      queue_config "gamma_eta_sweep alpha_nn_gradmatch_context $SEED 17 200 200 $gamma $eta $DATASET_LEN"
    done
  done
  run_queued
}

run_comparison() {
  local methods=(
    baseline
    dummy_init_existing
    attention_context_existing
    alpha_nn_init_direct
    alpha_nn_init_residual
    alpha_nn_gradmatch
    alpha_nn_gradmatch_context
  )
  local seeds=(42 43 44)
  local layers=(17 19)
  local epochs=(100 200)
  local method seed layer epoch
  for layer in "${layers[@]}"; do
    for epoch in "${epochs[@]}"; do
      for seed in "${seeds[@]}"; do
        for method in "${methods[@]}"; do
          queue_config "seed_comparison $method $seed $layer $epoch $epoch $GAMMA $ETA $DATASET_LEN"
        done
      done
    done
  done
  run_queued
}

case "$MODE" in
  verify)
    run_verify ;;
  smoke)
    run_verify
    run_smoke ;;
  sweep)
    run_sweep ;;
  comparison)
    run_comparison ;;
  all)
    run_verify
    run_smoke
    run_sweep
    run_comparison ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2 ;;
esac

conda run --no-capture-output -n deml python pia_alpha_gm.py --mode summarize --output-root "$ROOT"
