#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="single"
METHOD="B0_ACDR"
DATASET_LEN="28"
EPOCH="100"
SEED="42"
TARGET_LAYER="17"
OUTPUT_ROOT="runs/acdr"
DATASET_PATH="data/airline.json"
SUBDIR=""
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
    --resume) RESUME="--resume"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

python pia_b0_attention_consistency_repair.py \
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
  --uncertainty-fraction 0.20 \
  --max-repair-passes 2 \
  --lr 0.08 \
  --lambda-vocab 0.1 \
  --query-window 64 \
  --attn-row-fraction 0.20 \
  --max-attn-rows 128 \
  --entropy-middle-fraction 0.80 \
  --public-sink-threshold 0.50 \
  $SUBDIR \
  $RESUME
