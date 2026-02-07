#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Record a ROS2 bag (MCAP + zstd) from live topics using the RayFronts Docker image.

Usage:
  docker/run_record_mcap.sh /host/output_dir/bag_name [--all] [--] [extra ros2 bag record args...]

Examples:
  docker/run_record_mcap.sh /home/keisuke/Downloads/experiment-3
  docker/run_record_mcap.sh /home/keisuke/Downloads/all --all
  docker/run_record_mcap.sh /home/keisuke/Downloads/run1 -- --max-cache-size 0

Env vars:
  IMAGE_TAG     Docker image tag (default: rayfronts:desktop)
  ROS_DOMAIN_ID ROS2 domain id (must match your sim/robot)
  CACHE_DIR     Host cache dir to persist models (default: ~/.cache/rayfronts_docker)
  CACHE_VOLUME  Docker named volume for model cache (default: rayfronts_cache; best with RUN_AS_USER=0)
  DISABLE_CACHE Set to 1 to disable cache mounting
  RUN_AS_USER   Set to 1 to run container as your UID:GID (default: 1)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

OUT="${1:-}"
if [[ -z "$OUT" ]]; then
  usage
  exit 2
fi
shift 1

IMAGE_TAG="${IMAGE_TAG:-rayfronts:desktop}"
CACHE_VOLUME="${CACHE_VOLUME:-rayfronts_cache}"
DISABLE_CACHE="${DISABLE_CACHE:-0}"
RUN_AS_USER="${RUN_AS_USER:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_DIR="$SCRIPT_DIR/offline"

OUT_DIR="$(dirname -- "$OUT")"
mkdir -p "$OUT_DIR"

CACHE_ARGS=()
if [[ "$DISABLE_CACHE" != "1" ]]; then
  if [[ "$RUN_AS_USER" == "1" ]]; then
    HOST_USER="${SUDO_USER:-$USER}"
    HOST_UID="$(id -u "$HOST_USER")"
    HOST_GID="$(id -g "$HOST_USER")"
    HOST_HOME="$(getent passwd "$HOST_USER" | cut -d: -f6)"
    if [[ -z "$HOST_HOME" ]]; then
      echo "Failed to resolve home directory for user: $HOST_USER" >&2
      exit 2
    fi
    CACHE_DIR="${CACHE_DIR:-$HOST_HOME/.cache/rayfronts_docker}"
    if [[ "$EUID" -eq 0 ]]; then
      install -d -m 700 -o "$HOST_UID" -g "$HOST_GID" "$CACHE_DIR"
    else
      mkdir -p "$CACHE_DIR"
    fi
    CACHE_ARGS+=(
      -v "$CACHE_DIR:/tmp/.cache:rw"
      -e XDG_CACHE_HOME=/tmp/.cache
      -e HF_HOME=/tmp/.cache/huggingface
      -e TORCH_HOME=/tmp/.cache/torch
    )
  else
    CACHE_ARGS+=(-v "${CACHE_VOLUME}:/root/.cache")
  fi
fi

USER_ARGS=()
if [[ "$RUN_AS_USER" == "1" ]]; then
  HOST_USER="${SUDO_USER:-$USER}"
  HOST_UID="$(id -u "$HOST_USER")"
  HOST_GID="$(id -g "$HOST_USER")"
  USER_ARGS+=(
    --user "${HOST_UID}:${HOST_GID}"
    -e HOME=/tmp
    -e XDG_RUNTIME_DIR="/tmp/xdg-runtime-${HOST_UID}"
  )
fi

docker run --rm -it \
  "${USER_ARGS[@]}" \
  --network host --ipc host \
  -e ROS_DOMAIN_ID \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  "${CACHE_ARGS[@]}" \
  -v "$OFFLINE_DIR":/opt/rayfronts_offline:ro \
  -v "$OUT_DIR":/data/out \
  "$IMAGE_TAG" \
  /opt/rayfronts_offline/run_record_inside_container.sh --out "/data/out/$(basename -- "$OUT")" "$@"
