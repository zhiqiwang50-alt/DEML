#!/usr/bin/env bash
set -uo pipefail

ROOT="runs/cg_gdpi"
DATASET_NAME="Skytrax-28-pilot"
DATASET_PATH="data/airline.json"
MODE="smoke"
RESUME="--resume"
LOCAL_ONLY="--local-files-only"
MAX_JOBS=2
GPUS_CSV="0,1"
DATASET_LEN=28
SMOKE_LEN=2
LR=0.1
LAMBDA_VOCAB=0.1
LAMBDA_DUMMY=0.1
LAMBDA_CONTEXT=0.1
K=10
Y=10
GAMMA=0.3
ETA=0.5
TOP_M=3
MAX_TOKEN_LEN=640
TAU_CONSENSUS=0.80
TAU_GRAD_MARGIN=0.05
TAU_CAL_MARGIN_QUANTILE=0.30

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --output-root) ROOT="$2"; shift 2 ;;
    --dataset-path) DATASET_PATH="$2"; shift 2 ;;
    --dataset-name) DATASET_NAME="$2"; shift 2 ;;
    --dataset-len) DATASET_LEN="$2"; shift 2 ;;
    --smoke-len) SMOKE_LEN="$2"; shift 2 ;;
    --max-jobs) MAX_JOBS="$2"; shift 2 ;;
    --gpus) GPUS_CSV="$2"; shift 2 ;;
    --gamma) GAMMA="$2"; shift 2 ;;
    --eta) ETA="$2"; shift 2 ;;
    --top-m) TOP_M="$2"; shift 2 ;;
    --max-token-len) MAX_TOKEN_LEN="$2"; shift 2 ;;
    --resume) RESUME="--resume"; shift ;;
    --no-resume) RESUME=""; shift ;;
    --allow-download) LOCAL_ONLY=""; shift ;;
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
  local participants="$5"
  local attacker="$6"
  local layer="$7"
  local epoch="$8"
  local stage_a_epoch="$9"
  local dataset_len="${10}"
  local top_m="${11}"
  local attention_mode="${12}"
  local gate_mode="${13}"
  local gtag etag out status gate_args
  gtag="$(norm_float "$GAMMA")"
  etag="$(norm_float "$ETA")"
  gate_args=""
  if [[ "$gate_mode" == "closed" ]]; then
    gate_args="--force-gate-closed"
  fi
  out="$ROOT/$stage/${method}_p${participants}_a${attacker}_layer${layer}_epoch${epoch}_seed${seed}_topm${top_m}_${attention_mode}_${gate_mode}_maxlen${MAX_TOKEN_LEN}_g${gtag}_eta${etag}"
  mkdir -p "$out"
  echo "[cg-gdpi] gpu=$gpu stage=$stage method=$method seed=$seed p=$participants a=$attacker layer=$layer epoch=$epoch top_m=$top_m attention=$attention_mode gate=$gate_mode out=$out"
  CUDA_VISIBLE_DEVICES="$gpu" timeout 6h conda run --no-capture-output -n deml python -u pia_cg_gdpi.py \
    --mode single \
    $RESUME \
    $LOCAL_ONLY \
    --output-root "$ROOT" \
    --output-dir "$out" \
    --dataset-name "$DATASET_NAME" \
    --dataset-path "$DATASET_PATH" \
    --dataset-len "$dataset_len" \
    --method "$method" \
    --seed "$seed" \
    --participant-number "$participants" \
    --attacker-position "$attacker" \
    --target-layer "$layer" \
    --epoch "$epoch" \
    --stage-a-epoch "$stage_a_epoch" \
    --lr "$LR" \
    --lambda-vocab "$LAMBDA_VOCAB" \
    --lambda-dummy "$LAMBDA_DUMMY" \
    --lambda-context "$LAMBDA_CONTEXT" \
    --k "$K" \
    --y "$Y" \
    --gamma "$GAMMA" \
    --eta "$ETA" \
    --top-m "$top_m" \
    --max-token-len "$MAX_TOKEN_LEN" \
    --tau-consensus "$TAU_CONSENSUS" \
    --tau-grad-margin "$TAU_GRAD_MARGIN" \
    --tau-cal-margin-quantile "$TAU_CAL_MARGIN_QUANTILE" \
    --attention-mode "$attention_mode" \
    $gate_args \
    2>&1 | tee "$out/stdout.log"
  status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    echo "$status" > "$out/FAILED"
    echo "[cg-gdpi] failed status=$status out=$out" >&2
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
      local task_i task stage method seed participants attacker layer epoch stage_a_epoch dataset_len top_m attention_mode gate_mode
      for ((task_i = idx; task_i < ${#TASKS[@]}; task_i += worker_count)); do
        task="${TASKS[$task_i]}"
        read -r stage method seed participants attacker layer epoch stage_a_epoch dataset_len top_m attention_mode gate_mode <<< "$task"
        run_config "$gpu" "$stage" "$method" "$seed" "$participants" "$attacker" "$layer" "$epoch" "$stage_a_epoch" "$dataset_len" "$top_m" "$attention_mode" "$gate_mode" || true
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
  echo "[cg-gdpi] verification out=$out"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" conda run --no-capture-output -n deml python -u pia_cg_gdpi.py \
    --mode verify \
    $LOCAL_ONLY \
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
    --top-m "$TOP_M" \
    --max-token-len "$MAX_TOKEN_LEN" \
    2>&1 | tee "$out/stdout.log"
}

run_smoke() {
  local methods=(B0 B1 B2 B3 B4 P1)
  local method
  for method in "${methods[@]}"; do
    queue_config "smoke $method 42 4 4 17 100 100 $SMOKE_LEN $TOP_M raw open"
  done
  run_queued
}

run_main() {
  local methods=(B0 B1 B2 B3 B4 P1)
  local seeds=(42 43 44)
  local epochs=(100 200)
  local method seed epoch
  for epoch in "${epochs[@]}"; do
    for seed in "${seeds[@]}"; do
      for method in "${methods[@]}"; do
        queue_config "main_comparison $method $seed 4 4 17 $epoch $epoch $DATASET_LEN $TOP_M raw open"
      done
    done
  done
  run_queued
}

run_depth() {
  local methods=(B0 B2 B3 P1)
  local seeds=(42 43 44)
  local settings=("4 3 11" "4 4 17" "5 5 19")
  local method seed setting participants attacker layer
  for setting in "${settings[@]}"; do
    read -r participants attacker layer <<< "$setting"
    for seed in "${seeds[@]}"; do
      for method in "${methods[@]}"; do
        queue_config "depth_generalization $method $seed $participants $attacker $layer 200 200 $DATASET_LEN $TOP_M raw open"
      done
    done
  done
  run_queued
}

run_ablations() {
  queue_config "ablations P1 42 4 4 17 200 200 $DATASET_LEN 2 raw open"
  queue_config "ablations P1 42 4 4 17 200 200 $DATASET_LEN 3 raw open"
  queue_config "ablations P1 42 4 4 17 200 200 $DATASET_LEN 5 raw open"
  queue_config "ablations P2 42 4 4 17 200 200 $DATASET_LEN 3 bos_debias open"
  queue_config "ablations P1 42 4 4 17 200 200 $DATASET_LEN 3 raw closed"
  run_queued
}

case "$MODE" in
  verify)
    run_verify ;;
  smoke)
    run_verify
    run_smoke ;;
  main)
    run_main ;;
  depth)
    run_depth ;;
  ablations)
    run_ablations ;;
  summarize)
    ;;
  all)
    run_verify
    run_smoke
    run_main
    run_depth
    run_ablations ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2 ;;
esac

conda run --no-capture-output -n deml python pia_cg_gdpi.py --mode summarize --output-root "$ROOT"
