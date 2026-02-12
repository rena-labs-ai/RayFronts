#!/usr/bin/env bash
set -e

usage() {
  cat <<'EOF'
Run RayFronts offline on a rosbag2 MCAP (.mcap[.zstd]) recording.

Usage:
  run_inside_container.sh --bag /data/bag [--target-frame odom] [--rate 1.0] [--no-copy-bag] [--] [hydra overrides...]

Notes:
  - By default the bag is copied to `/tmp` before playback to avoid rosbag2
    writing decompressed artifacts back into the mounted directory.
  - Default topic names match `/home/keisuke/Downloads/experiment-2`.
  - This script sets `compile=False` by default to reduce VRAM usage. Override
    with `-- compile=True` if you have enough GPU memory.
EOF
}

BAG_DIR="/data/bag"
TARGET_FRAME="odom"
RATE="1.0"
COPY_BAG="1"

RGB_IN="/base/camera/color/image_raw/compressed"
DEPTH_IN="/base/camera/aligned_depth_to_color/image_raw/compressedDepth"
CAMERA_INFO="/base/camera/color/camera_info"

BRIDGE_SYNC_SLOP="${BRIDGE_SYNC_SLOP:-0.5}"
BRIDGE_TF_TIMEOUT="${BRIDGE_TF_TIMEOUT:-0.2}"
BRIDGE_POSE_FALLBACK="${BRIDGE_POSE_FALLBACK:-last}"
BRIDGE_STATS_PERIOD="${BRIDGE_STATS_PERIOD:-30}"

EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bag) BAG_DIR="$2"; shift 2 ;;
    --target-frame) TARGET_FRAME="$2"; shift 2 ;;
    --rate) RATE="$2"; shift 2 ;;
    --no-copy-bag) COPY_BAG="0"; shift 1 ;;
    --rgb-in) RGB_IN="$2"; shift 2 ;;
    --depth-in) DEPTH_IN="$2"; shift 2 ;;
    --camera-info) CAMERA_INFO="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_OVERRIDES+=("$@"); break ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! -d "$BAG_DIR" ]]; then
  echo "Bag directory not found: $BAG_DIR" >&2
  exit 2
fi

source /opt/ros/humble/setup.bash

# Helps reduce fragmentation-related OOMs in long runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Make sure Python logs show up immediately even when piped through `tee`.
export PYTHONUNBUFFERED=1

# Some GUI libs (including Rerun viewer) expect XDG_RUNTIME_DIR. When running
# as a non-root UID inside the container, this may not be set by default.
if [[ -n "${XDG_RUNTIME_DIR:-}" ]]; then
  mkdir -p "$XDG_RUNTIME_DIR" || true
  chmod 700 "$XDG_RUNTIME_DIR" || true
fi

# Copy the bag to a writable location (prevents root-owned `.mcap` artifacts on the host mount).
PLAY_BAG_DIR="$BAG_DIR"
if [[ "$COPY_BAG" == "1" ]]; then
  PLAY_BAG_DIR="/tmp/rayfronts_bag"
  rm -rf "$PLAY_BAG_DIR"
  mkdir -p "$PLAY_BAG_DIR"
  cp -a "$BAG_DIR"/. "$PLAY_BAG_DIR"/
fi

# Print bag stats (duration/message counts). Useful for verifying we're replaying what you expect.
echo "==== ros2 bag info: $PLAY_BAG_DIR ===="
ros2 bag info "$PLAY_BAG_DIR" | tee /tmp/rayfronts_baginfo.log || true
echo "====================================="

# rosbag2 plugins (MCAP + zstd)
if [[ ! -f /opt/ros/humble/lib/librosbag2_storage_mcap.so ]] || [[ ! -f /opt/ros/humble/lib/librosbag2_compression_zstd.so ]]; then
  if [[ "$(id -u)" == "0" ]]; then
    apt-get update
    apt-get install -y ros-humble-rosbag2-storage-mcap ros-humble-rosbag2-compression-zstd
  else
    echo "Missing rosbag2 MCAP/zstd plugins inside the image." >&2
    echo "Rebuild `rayfronts:desktop` after installing:" >&2
    echo "  ros-humble-rosbag2-storage-mcap ros-humble-rosbag2-compression-zstd" >&2
    echo "Or run once with RUN_AS_USER=0 to auto-install (root in container)." >&2
    exit 2
  fi
fi

cleanup() {
  echo "Stopping..." >&2
  if [[ -n "${BAG_PID:-}" ]]; then
    kill "$BAG_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${BRIDGE_PID:-}" ]]; then
    kill "$BRIDGE_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "[offline] Starting bridge (sync_slop=$BRIDGE_SYNC_SLOP tf_timeout=$BRIDGE_TF_TIMEOUT pose_fallback=$BRIDGE_POSE_FALLBACK)..." >&2
python3 /opt/rayfronts_offline/rosbag2_rayfronts_bridge.py \
  --rgb-in "$RGB_IN" \
  --depth-in "$DEPTH_IN" \
  --rgb-out /rayfronts/rgb \
  --depth-out /rayfronts/depth \
  --pose-out /rayfronts/pose \
  --camera-info "$CAMERA_INFO" \
  --target-frame "$TARGET_FRAME" \
  --sync-slop "$BRIDGE_SYNC_SLOP" \
  --tf-timeout "$BRIDGE_TF_TIMEOUT" \
  --pose-fallback "$BRIDGE_POSE_FALLBACK" \
  --stats-period "$BRIDGE_STATS_PERIOD" \
  --static-tf-identity base_camera_link camera_link \
  --drop-tf-edge camera_init map \
  --ros-args -p use_sim_time:=true \
  2>&1 | stdbuf -oL -eL tee /tmp/rayfronts_bridge.log | stdbuf -oL -eL sed 's/^/[bridge] /' &
BRIDGE_PID=$!

echo "[offline] Starting bag playback (rate=$RATE)..." >&2
stdbuf -oL -eL ros2 bag play "$PLAY_BAG_DIR" --clock --rate "$RATE" --disable-keyboard-controls \
  2>&1 | stdbuf -oL -eL tee /tmp/rayfronts_bagplay.log | stdbuf -oL -eL sed 's/^/[bagplay] /' &
BAG_PID=$!

# Fail fast if the bridge crashes immediately (common during dev).
sleep 0.5
if ! kill -0 "$BRIDGE_PID" >/dev/null 2>&1; then
  echo "[offline] Bridge process exited early. Last 200 lines of bridge log:" >&2
  tail -n 200 /tmp/rayfronts_bridge.log >&2 || true
  exit 2
fi

RUN_DIR="/tmp/rayfronts_run"
mkdir -p "$RUN_DIR"
export PYTHONPATH="/workspace/RayFronts:${PYTHONPATH:-}"
cd "$RUN_DIR"

echo "[offline] Starting RayFronts mapping_server..." >&2
CMD=(
  python3 -m rayfronts.mapping_server
  dataset=ros2zedx
  dataset.rgb_topic=/rayfronts/rgb
  dataset.pose_topic=/rayfronts/pose
  dataset.intrinsics_topic="$CAMERA_INFO"
  dataset.disparity_topic=null
  +dataset.depth_topic=/rayfronts/depth
  dataset.src_coord_system=rdf
  mapping=semantic_ray_frontiers_map
  interp_mode=nearest-exact
  compile=False
)
CMD+=("${EXTRA_OVERRIDES[@]}")

exec "${CMD[@]}"
