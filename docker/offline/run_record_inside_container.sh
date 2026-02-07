#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Record a ROS2 bag (MCAP + zstd) from live topics.

Usage:
  run_record_inside_container.sh --out /data/out/bag_name [--all] [--topic /foo]... [--] [ros2 bag record args...]

Examples:
  Record default topics (camera + TF):
    run_record_inside_container.sh --out /data/out/experiment-3

  Record everything:
    run_record_inside_container.sh --out /data/out/all --all

  Record a custom set of topics:
    run_record_inside_container.sh --out /data/out/run1 \
      --topic /tf --topic /tf_static \
      --topic /camera/color/image_raw --topic /camera/depth/image_rect \
      --topic /camera/color/camera_info

  Pass additional rosbag2 args:
    run_record_inside_container.sh --out /data/out/run1 -- --max-cache-size 0

Notes:
  - ROS env is sourced automatically.
  - Uses: `ros2 bag record -s mcap --compression-mode file --compression-format zstd`.
EOF
}

OUT=""
ALL="0"
TOPICS=()
EXTRA_RECORD_ARGS=()

DEFAULT_TOPICS=(
  /tf
  /tf_static
  /base/camera/color/image_raw/compressed
  /base/camera/aligned_depth_to_color/image_raw/compressedDepth
  /base/camera/color/camera_info
  /base/camera/depth/camera_info
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --all) ALL="1"; shift 1 ;;
    --topic) TOPICS+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_RECORD_ARGS+=("$@"); break ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$OUT" ]]; then
  echo "--out is required" >&2
  usage
  exit 2
fi

mkdir -p "$(dirname -- "$OUT")"

source /opt/ros/humble/setup.bash

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

CMD=(
  ros2 bag record
  -s mcap
  --compression-mode file
  --compression-format zstd
  -o "$OUT"
)

if [[ "$ALL" == "1" ]]; then
  if [[ ${#TOPICS[@]} -gt 0 ]]; then
    echo "Cannot combine --all with --topic" >&2
    exit 2
  fi
  CMD+=(-a)
else
  if [[ ${#TOPICS[@]} -gt 0 ]]; then
    CMD+=("${TOPICS[@]}")
  else
    CMD+=("${DEFAULT_TOPICS[@]}")
  fi
fi

CMD+=("${EXTRA_RECORD_ARGS[@]}")

exec "${CMD[@]}"
