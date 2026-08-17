#!/usr/bin/env bash
set -euo pipefail

cd /data/zhiqi/Collaborative_Inference/DEML

PYTHON_BIN="${PYTHON_BIN:-/home/zhiqi/miniconda3/envs/deml/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/air_ptr_fixed}"
LOG_ROOT="${OUTPUT_ROOT}/logs/dev"
mkdir -p "${LOG_ROOT}"

if (( $# > 0 )); then
  GPUS=("$@")
else
  GPUS=(1 2 3)
fi

METHODS=(B0 B0V ASINIT LAR LAR_PTR FINAL)
RHO="${RHO:-0.5}"

run_gpu_queue() {
  local gpu="$1"
  local slot="$2"
  local stride="$3"
  local index method out log
  for ((index=slot; index<${#METHODS[@]}; index+=stride)); do
    method="${METHODS[$index]}"
    out="${OUTPUT_ROOT}/dev/${method}"
    log="${LOG_ROOT}/${method}_gpu${gpu}.log"
    if [[ -f "${out}/COMPLETE" ]]; then
      echo "SKIP gpu=${gpu} method=${method}"
      continue
    fi
    echo "START $(date '+%F %T') gpu=${gpu} method=${method}"
    GPU_INDEX="${gpu}" PYTHON_BIN="${PYTHON_BIN}" OUTPUT_ROOT="${OUTPUT_ROOT}" scripts/run_air_ptr.sh single \
      --method "${method}" \
      --dataset-len 28 \
      --target-layer 17 \
      --epoch 100 \
      --stage-a-epoch 100 \
      --rho "${RHO}" \
      --output-dir "${out}" \
      > "${log}" 2>&1
    echo "DONE  $(date '+%F %T') gpu=${gpu} method=${method}"
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
