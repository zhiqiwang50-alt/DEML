#!/usr/bin/env bash
set -euo pipefail

RESUME="--resume"
ROOT="runs/tinyllama_full/main_results"
mkdir -p "${ROOT}"

CONFIGS=(
  "3 2 42" "3 2 43" "3 2 44"
  "3 3 42" "3 3 43" "3 3 44"
  "4 2 42" "4 2 43" "4 2 44"
  "4 3 42" "4 3 43" "4 3 44"
  "4 4 42" "4 4 43" "4 4 44"
  "5 2 42" "5 2 43" "5 2 44"
  "5 3 42" "5 3 43" "5 3 44"
  "5 4 42" "5 4 43" "5 4 44"
  "5 5 42" "5 5 43" "5 5 44"
)

run_one() {
  local participants="$1"
  local position="$2"
  local seed="$3"
  local gpu="$4"
  local name="p${participants}_a${position}_seed${seed}"
  local out="${ROOT}/${name}"
  mkdir -p "${out}"
  if [[ -f "${out}/COMPLETE" ]]; then
    echo "$(date -Is) skip complete ${name}"
    return 0
  fi
  echo "$(date -Is) start ${name} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" timeout 6h conda run --no-capture-output -n deml python -u pia_tinyllama.py \
    --mode single \
    ${RESUME} \
    --run-name "${name}" \
    --output-dir "${out}" \
    --dataset-name Skytrax-28-pilot \
    --dataset-path data/airline.json \
    --dataset-len 28 \
    --participant-number "${participants}" \
    --attacker-position "${position}" \
    --seed "${seed}" \
    --epoch 2000 \
    --lr 0.1 \
    --lambda-coeff 0.1 \
    --k 10 \
    --y 10 \
    > "${out}/stdout.log" 2>&1
  echo "$(date -Is) done ${name}"
}

idx=0
while [[ "${idx}" -lt "${#CONFIGS[@]}" ]]; do
  pids=()
  for gpu in 0 1 2 3; do
    if [[ "${idx}" -ge "${#CONFIGS[@]}" ]]; then
      break
    fi
    read -r participants position seed <<< "${CONFIGS[$idx]}"
    run_one "${participants}" "${position}" "${seed}" "${gpu}" &
    pids+=("$!")
    idx=$((idx + 1))
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "$(date -Is) one or more configs failed in this batch; continuing to next batch"
  fi
done

conda run -n deml python pia_tinyllama.py --mode table2
