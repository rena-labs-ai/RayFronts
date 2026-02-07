#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run RayFronts (docker image) offline on a rosbag2 MCAP bag.

Usage:
  docker/run_offline_mcap.sh /path/to/bag_dir [target_frame] [-- hydra overrides...]

Examples:
  docker/run_offline_mcap.sh /home/keisuke/Downloads/experiment-2
  docker/run_offline_mcap.sh /home/keisuke/Downloads/experiment-2 odom
  docker/run_offline_mcap.sh /home/keisuke/Downloads/experiment-2 odom -- encoder.model_version=radio_v2.5-b

Env vars:
  IMAGE_TAG   Docker image tag (default: rayfronts:desktop)
  RATE        ros2 bag play rate (default: 1.0)
  ROS_DOMAIN_ID  (Optional) ROS 2 domain id to isolate playback
  CACHE_DIR    Host cache dir to persist models (default: ~/.cache/rayfronts_docker)
  CACHE_VOLUME Docker named volume for model cache (default: rayfronts_cache; best with RUN_AS_USER=0)
  DISABLE_CACHE Set to 1 to disable cache mounting
  WITH_X11     Set to 1 to forward X11 for GUI apps like Rerun (default: 0)
  MOUNT_PATCHES Set to 0 to not mount local patches into the container (default: 1)
  RUN_AS_USER  Set to 1 to run container as your UID:GID (default: 1)

Notes (X11):
  - If WITH_X11=1 and RUN_AS_USER=1, you typically need:
      xhost +SI:localuser:$USER
  - If WITH_X11=1 and RUN_AS_USER=0 (root in container), you typically need:
      xhost +SI:localuser:root
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

BAG_DIR="${1:-/home/keisuke/Downloads/experiment-2}"
shift $(( $# >= 1 ? 1 : 0 ))

TARGET_FRAME="odom"
if [[ $# -gt 0 && "${1:-}" != "--" ]]; then
  TARGET_FRAME="$1"
  shift 1
fi

EXTRA=()
if [[ $# -gt 0 && "${1:-}" == "--" ]]; then
  shift 1
  EXTRA=("$@")
fi

if [[ ! -d "$BAG_DIR" ]]; then
  echo "Bag directory not found: $BAG_DIR" >&2
  exit 2
fi

IMAGE_TAG="${IMAGE_TAG:-rayfronts:desktop}"
RATE="${RATE:-1.0}"
CACHE_VOLUME="${CACHE_VOLUME:-rayfronts_cache}"
DISABLE_CACHE="${DISABLE_CACHE:-0}"
WITH_X11="${WITH_X11:-0}"
MOUNT_PATCHES="${MOUNT_PATCHES:-1}"
RUN_AS_USER="${RUN_AS_USER:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_DIR="$SCRIPT_DIR/offline"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

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

    # Persist HF + torch hub caches across `--rm` runs (host bind mount).
    CACHE_ARGS+=(
      -v "$CACHE_DIR:/tmp/.cache:rw"
      -e XDG_CACHE_HOME=/tmp/.cache
      -e HF_HOME=/tmp/.cache/huggingface
      -e TORCH_HOME=/tmp/.cache/torch
    )
  else
    # Root-in-container mode: a named volume works well.
    CACHE_ARGS+=(-v "${CACHE_VOLUME}:/root/.cache")
  fi
fi

X11_ARGS=()
if [[ "$WITH_X11" == "1" ]]; then
  if [[ -z "${DISPLAY:-}" ]]; then
    echo "WITH_X11=1 but DISPLAY is not set." >&2
    echo "Try: sudo DISPLAY=\\$DISPLAY WITH_X11=1 $0 \"$BAG_DIR\" \"$TARGET_FRAME\" -- ..." >&2
    exit 2
  fi
  # Basic X11 forwarding (local unix socket). You may need to allow root:
  #   xhost +local:root
  X11_ARGS+=(
    -e DISPLAY
    -e QT_X11_NO_MITSHM=1
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw
  )
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

SRC_ARGS=()
if [[ "$MOUNT_PATCHES" != "0" ]]; then
  # Mount targeted local patches without hiding compiled extensions under
  # `/workspace/RayFronts/rayfronts/csrc/build` (mounting the whole `rayfronts/`
  # tree can mask `rayfronts_cpp`).
  if [[ -f "$REPO_ROOT/rayfronts/image_encoders/naradio.py" ]]; then
    SRC_ARGS+=(-v "$REPO_ROOT/rayfronts/image_encoders/naradio.py:/workspace/RayFronts/rayfronts/image_encoders/naradio.py:ro")
  fi
fi

docker run --rm -it \
  "${USER_ARGS[@]}" \
  --gpus all --runtime=nvidia \
  --network host --ipc host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e ROS_DOMAIN_ID \
  "${X11_ARGS[@]}" \
  "${CACHE_ARGS[@]}" \
  "${SRC_ARGS[@]}" \
  -v "$BAG_DIR":/data/bag:ro \
  -v "$OFFLINE_DIR":/opt/rayfronts_offline:ro \
  "$IMAGE_TAG" \
  bash -lc "/opt/rayfronts_offline/run_inside_container.sh --bag /data/bag --target-frame \"$TARGET_FRAME\" --rate \"$RATE\" -- ${EXTRA[*]}"
