#!/usr/bin/env bash
set -euo pipefail

cd /data/zhiqi/Collaborative_Inference/DEML
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deml

OUTROOT="runs/p4lwr_last_window_residual_top1"
RUNROOT="$OUTROOT/seed42_skytrax150_layers11_17_19"
LOGDIR="$RUNROOT/logs"
mkdir -p "$LOGDIR"

export TRANSFORMERS_CACHE=/home/zhiqi/data/hf_cache/transformers
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="data/skytrax_150.json"
COMMON=(
  --mode single
  --dataset-path "$DATASET"
  --dataset-len 150
  --seed 42
  --participant-number 4
  --attacker-position 4
  --epoch 100
  --max-token-len 896
  --k 1
  --y 0
  --disable-semantic-speculation
  --local-files-only
  --attention-start-ratio 0.7
  --attention-full-ratio 0.7
  --alpha-min 0.5
  --alpha-max 1.5
  --output-root "$OUTROOT"
)

run_one() {
  local method="$1"
  local layer="$2"
  local rho="$3"
  local out="$RUNROOT/${method}_seed42_layer${layer}_epoch100_k1_y0"
  local log="$LOGDIR/${method}_seed42_layer${layer}.log"
  if [ -f "$out/COMPLETE" ]; then
    echo "SKIP method=$method layer=$layer output=$out"
    return 0
  fi
  echo "START $(date '+%F %T') method=$method layer=$layer rho=$rho gpu=$CUDA_VISIBLE_DEVICES output=$out log=$log"
  python pia_masked_server_attn_pia.py "${COMMON[@]}" \
    --method "$method" \
    --target-layer "$layer" \
    --residual-alpha-rho "$rho" \
    --output-dir "$out" \
    > "$log" 2>&1
  echo "DONE  $(date '+%F %T') method=$method layer=$layer"
}

for layer in 17 11 19; do
  case "$layer" in
    11) rho="1.0" ;;
    17) rho="0.6" ;;
    19) rho="0.4" ;;
    *) echo "unexpected layer=$layer" >&2; exit 2 ;;
  esac
  run_one B0 "$layer" "$rho"
  run_one B0V "$layer" "$rho"
  run_one P4LWR "$layer" "$rho"
done

echo "ALL_DONE $(date '+%F %T')"
