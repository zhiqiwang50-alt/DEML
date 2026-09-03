#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="verify"
METHOD="RAP_FINAL"
DATASET_LEN="28"
EPOCH="100"
SEED="42"
TARGET_LAYER="17"
OUTPUT_ROOT="runs/rap_final"
DATASET_PATH="data/airline.json"
SUBDIR=""
UNCERTAINTY_DETECTOR="margin"
RESUME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --dataset-len) DATASET_LEN="$2"; shift 2 ;;
    --epoch) EPOCH="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --target-layer) TARGET_LAYER="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dataset-path) DATASET_PATH="$2"; shift 2 ;;
    --subdir) SUBDIR="--subdir $2"; shift 2 ;;
    --uncertainty-detector) UNCERTAINTY_DETECTOR="$2"; shift 2 ;;
    --resume) RESUME="--resume"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$MODE" in
  verify|smoke|dev|prepare-heldout|summarize|audit-uncertainty|single)
    python pia_rap_final.py \
      --mode "$MODE" \
      --method "$METHOD" \
      --dataset-path "$DATASET_PATH" \
      --dataset-len "$DATASET_LEN" \
      --epoch "$EPOCH" \
      --seed "$SEED" \
      --target-layer "$TARGET_LAYER" \
      --k 1 \
      --y 0 \
      --disable-semantic-speculation \
      --no-adaptive-discretization \
      --local-files-only \
      --output-root "$OUTPUT_ROOT" \
      --patch-size 16 \
      --min-patch-size 16 \
      --max-patch-size 16 \
      --uncertainty-fraction 0.20 \
      --top-r 4 \
      --max-repair-passes 2 \
      --lr 0.08 \
      --lambda-vocab 0.1 \
      --uncertainty-detector "$UNCERTAINTY_DETECTOR" \
      $SUBDIR \
      $RESUME
    ;;
  full-reference)
    RUN_NAME="pia_full_reference_seed${SEED}_layer${TARGET_LAYER}"
    OUT_DIR="${OUTPUT_ROOT}/heldout_seed${SEED}/pia_full_reference"
    python pia_attention_guided.py \
      --mode single \
      --method baseline \
      --dataset-name Skytrax \
      --dataset-path "$DATASET_PATH" \
      --dataset-len "$DATASET_LEN" \
      --seed "$SEED" \
      --target-layer "$TARGET_LAYER" \
      --epoch "$EPOCH" \
      --stage-a-epoch "$EPOCH" \
      --k 10 \
      --y 10 \
      --max-token-len 896 \
      --output-root "$OUTPUT_ROOT" \
      --output-dir "$OUT_DIR" \
      --run-name "$RUN_NAME" \
      --local-files-only \
      $RESUME
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
