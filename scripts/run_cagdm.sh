#!/usr/bin/env bash
set -euo pipefail
/home/zhiqi/miniconda3/bin/conda run -n deml python pia_cagdm.py --output-root runs/cagdm "$@"
