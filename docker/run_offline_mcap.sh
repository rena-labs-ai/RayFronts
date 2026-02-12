#!/usr/bin/env bash
set -euo pipefail

MCAP_PATH="${1:-/home/keisuke/Downloads/experiment-2/experiment-2_0.mcap}"
QUERY="${2:-door}"

EXTRA_ARGS=()
if [ "$#" -gt 2 ]; then
  EXTRA_ARGS=("${@:3}")
fi

python3 scripts/semantic_query_cli.py \
  --source mcap \
  --mcap-path "${MCAP_PATH}" \
  --object "${QUERY}" \
  "${EXTRA_ARGS[@]}"
