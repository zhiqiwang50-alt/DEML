#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-verify}"
shift || true

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/air_ptr}"
DATASET_PATH="${DATASET_PATH:-data/airline.json}"
GPU_INDEX="${GPU_INDEX:-}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/home/zhiqi/data/hf_cache/transformers}"

select_gpu() {
  if [[ -n "${GPU_INDEX}" ]]; then
    printf '%s\n' "${GPU_INDEX}"
    return
  fi
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr \
    | head -n1 \
    | cut -d, -f1 \
    | tr -d ' '
}

GPU="$(select_gpu)"
COMMON=(
  --output-root "${OUTPUT_ROOT}"
  --dataset-name Skytrax-28
  --dataset-path "${DATASET_PATH}"
  --seed 42
  --k 1
  --y 0
  --disable-semantic-speculation
  --local-files-only
  --max-token-len 896
  --gamma 0.3
  --ptr-eta 0.10
  --repair-fraction 0.20
  --max-positions-per-pass 8
  --max-passes 2
  --epsilon-accept 1e-6
)

case "${MODE}" in
  verify)
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" pia_air_ptr.py \
      --mode verify --dataset-len 1 --target-layer 17 --epoch 2 --stage-a-epoch 2 \
      "${COMMON[@]}" "$@"
    ;;
  smoke)
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" pia_air_ptr.py \
      --mode smoke --dataset-len 2 --target-layer 17 --epoch 20 --stage-a-epoch 20 --resume \
      "${COMMON[@]}" "$@"
    ;;
  rho)
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" pia_air_ptr.py \
      --mode rho-ablation --dataset-len 28 --target-layers 11 17 19 --epoch 100 --stage-a-epoch 100 --resume \
      --rhos 0.0 0.25 0.50 0.75 1.0 "${COMMON[@]}" "$@"
    ;;
  dev)
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" pia_air_ptr.py \
      --mode dev --dataset-len 28 --target-layer 17 --epoch 100 --stage-a-epoch 100 --resume \
      "${COMMON[@]}" "$@"
    ;;
  single)
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" pia_air_ptr.py \
      --mode single --resume "${COMMON[@]}" "$@"
    ;;
  summarize)
    "${PYTHON_BIN}" pia_air_ptr.py --mode summarize --output-root "${OUTPUT_ROOT}" \
      --k 1 --y 0 --disable-semantic-speculation "$@"
    ;;
  *)
    echo "usage: scripts/run_air_ptr.sh {verify|smoke|rho|dev|single|summarize} [args...]" >&2
    exit 2
    ;;
esac
