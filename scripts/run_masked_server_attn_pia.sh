#!/usr/bin/env bash
set -euo pipefail
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/home/zhiqi/data/hf_cache/transformers}"
/home/zhiqi/miniconda3/bin/conda run --no-capture-output -n deml python pia_masked_server_attn_pia.py "$@"
