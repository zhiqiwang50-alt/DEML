#!/usr/bin/env bash
set -euo pipefail

cd /data/zhiqi/Collaborative_Inference/DEML

PYTHON_BIN="${PYTHON_BIN:-/home/zhiqi/miniconda3/envs/deml/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/air_ptr}"
LOG_ROOT="${OUTPUT_ROOT}/logs/rho_ablation"
mkdir -p "${LOG_ROOT}"

if (( $# > 0 )); then
  GPUS=("$@")
else
  GPUS=(1 2 3)
fi
LAYERS=(11 17 19)
RHOS=(0.0 0.25 0.50 0.75 1.0)
TASKS=()
for layer in "${LAYERS[@]}"; do
  for rho in "${RHOS[@]}"; do
    TASKS+=("${layer}:${rho}")
  done
done

run_gpu_queue() {
  local gpu="$1"
  local slot="$2"
  local stride="$3"
  local index task layer rho tag out log
  for ((index=slot; index<${#TASKS[@]}; index+=stride)); do
    task="${TASKS[$index]}"
    layer="${task%%:*}"
    rho="${task##*:}"
    tag="${rho//./p}"
    out="${OUTPUT_ROOT}/rho_ablation/layer${layer}/rho${tag}"
    log="${LOG_ROOT}/layer${layer}_rho${tag}_gpu${gpu}.log"
    if [[ -f "${out}/COMPLETE" ]]; then
      echo "SKIP gpu=${gpu} layer=${layer} rho=${rho}"
      continue
    fi
    echo "START $(date '+%F %T') gpu=${gpu} layer=${layer} rho=${rho}"
    GPU_INDEX="${gpu}" PYTHON_BIN="${PYTHON_BIN}" scripts/run_air_ptr.sh single \
      --method LAR \
      --dataset-len 28 \
      --target-layer "${layer}" \
      --epoch 100 \
      --stage-a-epoch 100 \
      --rho "${rho}" \
      --output-dir "${out}" \
      > "${log}" 2>&1
    echo "DONE  $(date '+%F %T') gpu=${gpu} layer=${layer} rho=${rho}"
  done
}

worker_count="${#GPUS[@]}"
pids=()
for ((slot=0; slot<worker_count; slot++)); do
  run_gpu_queue "${GPUS[$slot]}" "${slot}" "${worker_count}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
