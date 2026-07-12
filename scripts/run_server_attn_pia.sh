#!/usr/bin/env bash
set -euo pipefail
/home/zhiqi/miniconda3/bin/conda run --no-capture-output -n deml python pia_server_attn_pia.py --output-root runs/server_attn_pia "$@"
