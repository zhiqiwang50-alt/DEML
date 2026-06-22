#!/usr/bin/env bash
set -euo pipefail

RESUME=""
if [[ "${1:-}" == "--resume" ]]; then
  RESUME="--resume"
fi

bash scripts/run_tinyllama_smoke.sh ${RESUME}
bash scripts/run_tinyllama_main.sh ${RESUME}
bash scripts/run_tinyllama_ablation.sh ${RESUME}
bash scripts/run_tinyllama_defense.sh ${RESUME}
bash scripts/run_tinyllama_baselines.sh ${RESUME}
conda run -n deml python pia_tinyllama.py --mode report
