#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run RayFronts live against ROS2 topics (simulation or real robot).

Usage:
  run_live_inside_container.sh [--with-bridge] [bridge args...] -- [hydra overrides...]

Modes:
  - Direct (no bridge): RayFronts subscribes directly to your topics.
    Provide topics via Hydra overrides after `--`.

  - With bridge: Starts `rosbag2_rayfronts_bridge.py` to convert:
      * CompressedImage RGB + compressedDepth -> raw Image + depth meters
      * TF -> PoseStamped
    Then RayFronts subscribes to `/rayfronts/{rgb,depth,pose}`.

Examples:
  Direct:
    run_live_inside_container.sh -- \
      dataset=ros2zedx \
      dataset.rgb_topic:=/camera/color/image_raw \
      +dataset.depth_topic:=/camera/depth/image_rect \
      dataset.pose_topic:=/camera/pose \
      dataset.intrinsics_topic:=/camera/color/camera_info \
      dataset.disparity_topic:=null \
      dataset.src_coord_system:=rdf \
      mapping=semantic_ray_frontiers_map

  With bridge (compressed + TF):
    run_live_inside_container.sh --with-bridge \
      --rgb-in /base/camera/color/image_raw/compressed \
      --depth-in /base/camera/aligned_depth_to_color/image_raw/compressedDepth \
      --camera-info /base/camera/color/camera_info \
      --target-frame odom \
      --static-tf-identity base_camera_link camera_link \
      --drop-tf-edge camera_init map \
      -- \
      dataset=ros2zedx dataset.disparity_topic:=null dataset.src_coord_system:=rdf mapping=semantic_ray_frontiers_map

Notes:
  - ROS env is sourced automatically.
  - Use `--network host` when running the container.
EOF
}

WITH_BRIDGE="0"
BRIDGE_ARGS=()
EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-bridge) WITH_BRIDGE="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_OVERRIDES+=("$@"); break ;;
    *)
      if [[ "$WITH_BRIDGE" == "1" ]]; then
        BRIDGE_ARGS+=("$1")
        shift 1
      else
        echo "Unknown arg (did you mean to pass it after -- ?): $1" >&2
        usage
        exit 2
      fi
      ;;
  esac
done

source /opt/ros/humble/setup.bash
export PYTHONUNBUFFERED=1

if [[ -n "${XDG_RUNTIME_DIR:-}" ]]; then
  mkdir -p "$XDG_RUNTIME_DIR" || true
  chmod 700 "$XDG_RUNTIME_DIR" || true
fi

cleanup() {
  if [[ -n "${BRIDGE_PID:-}" ]]; then
    kill "$BRIDGE_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "$WITH_BRIDGE" == "1" ]]; then
  BRIDGE_SYNC_SLOP="${BRIDGE_SYNC_SLOP:-0.5}"
  BRIDGE_TF_TIMEOUT="${BRIDGE_TF_TIMEOUT:-1.0}"
  BRIDGE_POSE_FALLBACK="${BRIDGE_POSE_FALLBACK:-last}"
  BRIDGE_STATS_PERIOD="${BRIDGE_STATS_PERIOD:-30}"

  echo "[live] Starting bridge (sync_slop=$BRIDGE_SYNC_SLOP tf_timeout=$BRIDGE_TF_TIMEOUT pose_fallback=$BRIDGE_POSE_FALLBACK)..." >&2
  python3 /opt/rayfronts_offline/rosbag2_rayfronts_bridge.py \
    --sync-slop "$BRIDGE_SYNC_SLOP" \
    --tf-timeout "$BRIDGE_TF_TIMEOUT" \
    --pose-fallback "$BRIDGE_POSE_FALLBACK" \
    --stats-period "$BRIDGE_STATS_PERIOD" \
    "${BRIDGE_ARGS[@]}" \
    2>&1 | stdbuf -oL -eL tee /tmp/rayfronts_bridge.log | stdbuf -oL -eL sed 's/^/[bridge] /' &
  BRIDGE_PID=$!

  sleep 0.5
  if ! kill -0 "$BRIDGE_PID" >/dev/null 2>&1; then
    echo "[live] Bridge process exited early. Last 200 lines of bridge log:" >&2
    tail -n 200 /tmp/rayfronts_bridge.log >&2 || true
    exit 2
  fi
fi

RUN_DIR="/tmp/rayfronts_run"
mkdir -p "$RUN_DIR"
export PYTHONPATH="/workspace/RayFronts:${PYTHONPATH:-}"
cd "$RUN_DIR"

echo "[live] Starting RayFronts mapping_server..." >&2
CMD=(python3 -m rayfronts.mapping_server)
CMD+=("${EXTRA_OVERRIDES[@]}")

exec "${CMD[@]}"
