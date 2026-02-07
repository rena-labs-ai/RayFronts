# Offline rosbag2 MCAP (Docker)

This folder contains scripts to run RayFronts offline on a recorded ROS2 bag
(MCAP `.mcap` / `.mcap.zstd`), tested with `/home/keisuke/Downloads/experiment-2`.

## Quick start (your `experiment-2` bag)

1) Build the desktop image:

```bash
cd docker
docker build . -f desktop.Dockerfile -t rayfronts:desktop
```

2) Run offline mapping:

```bash
docker/run_offline_mcap.sh /home/keisuke/Downloads/experiment-2
```

Optionally specify a TF world frame as the second argument:

```bash
docker/run_offline_mcap.sh /home/keisuke/Downloads/experiment-2 odom
```

## Caching model downloads (HuggingFace / torch hub)

By default, the wrapper scripts persist model caches so weights aren’t
re-downloaded every run (even with `--rm`).

- Default (non-root containers, `RUN_AS_USER=1`): bind-mount host cache dir at `/tmp/.cache`
  - Default host cache dir: `~/.cache/rayfronts_docker` (override with `CACHE_DIR=/path`)

- Root-in-container (`RUN_AS_USER=0`): mount a named Docker volume at `/root/.cache`
  - Default volume name: `rayfronts_cache` (override with `CACHE_VOLUME=...`)

- Disable: `DISABLE_CACHE=1 docker/run_offline_mcap.sh ...`

## Non-root containers

The docker wrapper scripts run the container as your host `UID:GID` by default
(`RUN_AS_USER=1`). This avoids running as root inside the container and keeps
any files created in bind mounts owned by you.

If you need root-in-container (e.g. to auto-install missing ROS packages at
runtime), run:

```bash
RUN_AS_USER=0 docker/run_offline_mcap.sh /path/to/bag_dir
```

## Live operation

Run RayFronts against live ROS2 topics (simulation or robot) using:

```bash
docker/run_live_ros2.sh -- <hydra overrides...>
```

If your simulation publishes `CompressedImage` + TF (and not raw `Image` + `PoseStamped`),
use the bridge mode:

```bash
docker/run_live_ros2.sh -- \
  --with-bridge \
  --rgb-in /base/camera/color/image_raw/compressed \
  --depth-in /base/camera/aligned_depth_to_color/image_raw/compressedDepth \
  --camera-info /base/camera/color/camera_info \
  --target-frame odom \
  -- \
  dataset=ros2zedx dataset.disparity_topic:=null dataset.src_coord_system:=rdf mapping=semantic_ray_frontiers_map
```

## Record + replay

Record live topics to an MCAP+zstd bag:

```bash
docker/run_record_mcap.sh /path/to/output_bag_name
```

Then replay offline:

```bash
docker/run_offline_mcap.sh /path/to/output_bag_name
```

## Troubleshooting: CUDA OOM

If you see a CUDA out-of-memory error inside the encoder (often `naradio` / `RADIO`):

- Keep `compile=False` (the offline script already sets this by default).
- Use `interp_mode=nearest-exact` (the offline script already sets this by default).
- Use a smaller RADIO model:

```bash
docker/run_offline_mcap.sh /path/to/bag_dir odom -- encoder.model_version=radio_v2.5-b
```

- Downscale input (reduces VRAM further):

```bash
docker/run_offline_mcap.sh /path/to/bag_dir odom -- dataset.rgb_resolution=[352,640] dataset.depth_resolution=[352,640]
```

## Troubleshooting: Rerun shows only one frame / stable_time is 0

If the Rerun viewer opens but you only see a single frame (or `stable_time` has
zero duration), RayFronts likely isn't receiving a continuous RGBD+pose stream.

Things to check:
- The terminal should show `[bridge] Published ... pairs ...` periodically.
- `ros2 bag info` (printed at start) should show a non-trivial duration and
  message counts for your image topics.
- In the Rerun UI, make sure you are viewing the `stable_time` timeline (not
  `log_time`), then hit play.

Bridge tuning knobs (env vars, used by offline + live bridge mode):
- `BRIDGE_SYNC_SLOP` (default `0.5`)
- `BRIDGE_TF_TIMEOUT` (default `1.0`)
- `BRIDGE_POSE_FALLBACK` (default `last`, options: `drop|last|identity`)
- `BRIDGE_STATS_PERIOD` (default `30`)

If you see `TF lookup failed (... source_frame does not exist)`, it often means
the bridge did not receive TF due to QoS mismatch. The bridge subscribes to
`/tf` and `/tf_static` with BEST_EFFORT to match rosbag2 playback.

## What the scripts do

- `docker/run_offline_mcap.sh`
  - Starts `rayfronts:desktop` and mounts the bag at `/data/bag` (read-only).
  - Mounts this folder at `/opt/rayfronts_offline`.
- `docker/offline/run_inside_container.sh`
  - Copies the bag to `/tmp/rayfronts_bag` by default (prevents root-owned
    decompressed artifacts on the host).
  - Starts `docker/offline/rosbag2_rayfronts_bridge.py`.
  - Plays the bag with `ros2 bag play --clock`.
  - Launches `python3 -m rayfronts.mapping_server` with `dataset=ros2zedx`
    and the bridged topics.

Logs:
- `/tmp/rayfronts_bridge.log`
- `/tmp/rayfronts_bagplay.log`

## Topic expectations

Defaults match the provided bag:
- RGB: `/base/camera/color/image_raw/compressed` (`sensor_msgs/msg/CompressedImage`)
- Depth: `/base/camera/aligned_depth_to_color/image_raw/compressedDepth` (`sensor_msgs/msg/CompressedImage`)
- Intrinsics: `/base/camera/color/camera_info` (`sensor_msgs/msg/CameraInfo`)
- TF: `/tf` and `/tf_static`

The bridge republishes:
- `/rayfronts/rgb` (`sensor_msgs/msg/Image`, `bgr8`)
- `/rayfronts/depth` (`sensor_msgs/msg/Image`, `32FC1` meters)
- `/rayfronts/pose` (`geometry_msgs/msg/PoseStamped`)

## TF notes (specific to `experiment-2`)

This bag has:
- A TF loop: `map <-> camera_init`
- Camera driver frames (`camera_link`, `camera_color_optical_frame`, …) not
  connected to the robot URDF camera mount frame (`base_camera_link`).

To make TF lookups work, the bridge:
- Drops the TF edge `camera_init -> map` (breaks the loop).
- Inserts an identity static transform `base_camera_link -> camera_link`.

If your setup uses different frame names or non-identity extrinsics, run the
bridge with custom static TF(s) (see `--static-tf` / `--static-tf-identity` in
`docker/offline/rosbag2_rayfronts_bridge.py`).
