#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/zhiqi/miniconda3/envs/deml/bin/python}"
"$PYTHON_BIN" pia_attention_weighted_pia.py "$@"
