#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run RayFronts live in Docker against ROS2 topics (simulation or real robot).

Usage:
  docker/run_live_ros2.sh [--image rayfronts:desktop] [--] [args for run_live_inside_container.sh...]

Examples:
  Direct (no bridge):
    docker/run_live_ros2.sh \
      -- \
      dataset=ros2zedx \
      dataset.rgb_topic:=/camera/color/image_raw \
      +dataset.depth_topic:=/camera/depth/image_rect \
      dataset.pose_topic:=/camera/pose \
      dataset.intrinsics_topic:=/camera/color/camera_info \
      dataset.disparity_topic:=null \
      dataset.src_coord_system:=rdf \
      mapping=semantic_ray_frontiers_map

  With bridge (compressed + TF):
    docker/run_live_ros2.sh \
      --with-bridge \
      --rgb-in /base/camera/color/image_raw/compressed \
      --depth-in /base/camera/aligned_depth_to_color/image_raw/compressedDepth \
      --camera-info /base/camera/color/camera_info \
      --target-frame odom \
      --static-tf-identity base_camera_link camera_link \
      --drop-tf-edge camera_init map \
      -- \
      dataset=ros2zedx dataset.disparity_topic:=null dataset.src_coord_system:=rdf mapping=semantic_ray_frontiers_map

Env vars:
  IMAGE_TAG     Docker image tag (default: rayfronts:desktop)
  ROS_DOMAIN_ID ROS2 domain id (must match your sim/robot)
  CACHE_DIR     Host cache dir to persist models (default: ~/.cache/rayfronts_docker)
  CACHE_VOLUME  Docker named volume for model cache (default: rayfronts_cache; best with RUN_AS_USER=0)
  DISABLE_CACHE Set to 1 to disable cache mounting
  WITH_X11      Set to 1 to forward X11 for GUI apps like Rerun (default: 0)
  MOUNT_PATCHES Set to 0 to not mount local patches into the container (default: 1)
  RUN_AS_USER   Set to 1 to run container as your UID:GID (default: 1)

Notes (X11):
  - If WITH_X11=1 and RUN_AS_USER=1, you typically need:
      xhost +SI:localuser:$USER
  - If WITH_X11=1 and RUN_AS_USER=0 (root in container), you typically need:
      xhost +SI:localuser:root
EOF
}

IMAGE_TAG="${IMAGE_TAG:-rayfronts:desktop}"
CACHE_VOLUME="${CACHE_VOLUME:-rayfronts_cache}"
DISABLE_CACHE="${DISABLE_CACHE:-0}"
WITH_X11="${WITH_X11:-0}"
MOUNT_PATCHES="${MOUNT_PATCHES:-1}"
RUN_AS_USER="${RUN_AS_USER:-1}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

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

X11_ARGS=()
if [[ "$WITH_X11" == "1" ]]; then
  if [[ -z "${DISPLAY:-}" ]]; then
    echo "WITH_X11=1 but DISPLAY is not set." >&2
    echo "Try: sudo DISPLAY=\\$DISPLAY WITH_X11=1 $0 -- ..." >&2
    exit 2
  fi
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
  -v "$OFFLINE_DIR":/opt/rayfronts_offline:ro \
  "$IMAGE_TAG" \
  /opt/rayfronts_offline/run_live_inside_container.sh "$@"
