# Semantic Query CLI

`scripts/semantic_query_cli.py` runs RayFronts mapping on an RGBD stream and
returns an estimated object position from semantic query scores.

Estimation rule:
- select the most salient `top_k` voxels (default `5`) by query similarity
- output position = coordinate-wise median of those voxels
- output distance = median Euclidean distance from the latest robot pose

## Quick Start (MCAP)

```bash
python3 scripts/semantic_query_cli.py \
  --source mcap \
  --mcap-path /home/keisuke/Downloads/experiment-2/experiment-2_0.mcap \
  --object "door" \
  --top-k 5 \
  --max-frames 120 \
  --encoder-class your_package.your_encoder.YourEncoder \
  --encoder-kwargs '{"device": "cuda"}' \
  --mapper-class rayfronts.mapping.semantic_voxel_map.SemanticVoxelMap \
  --mapper-kwargs '{"vox_size": 0.3, "max_pts_per_frame": 1500, "vox_accum_period": 4}'
```

Shortcut wrapper:

```bash
docker/run_offline_mcap.sh /home/keisuke/Downloads/experiment-2/experiment-2_0.mcap door \
  --max-frames 120 \
  --encoder-class your_package.your_encoder.YourEncoder \
  --encoder-kwargs '{"device": "cuda"}'
```

## Live ROS2 Example

```bash
python3 scripts/semantic_query_cli.py \
  --source live \
  --object "trash can" \
  --rgb-topic /camera/color/image_raw \
  --depth-topic /camera/depth/image_raw \
  --pose-topic /camera/pose \
  --intrinsics-topic /camera/camera_info \
  --encoder-class your_package.your_encoder.YourEncoder \
  --mapper-class rayfronts.mapping.semantic_voxel_map.SemanticVoxelMap
```

## Existing Dataset Example

```bash
python3 scripts/semantic_query_cli.py \
  --source dataset \
  --dataset-class rayfronts.datasets.ros.RosnpyDataset \
  --dataset-kwargs '{"path": "/path/to/ros_dump.npz"}' \
  --object "cabinet" \
  --encoder-class your_package.your_encoder.YourEncoder
```

## Output

The CLI prints JSON with:
- `estimated_xyz`: median object position from salient voxels
- `median_distance_m`: median distance from the latest pose
- `salient_voxels`: top voxels (xyz, score, distance)
- `processed_frames`

Use `--output-json /path/result.json` to persist results.

## Visual Verification (Overlay)

Use `scripts/verify_query_overlay.py` to project the returned world coordinates
(`estimated_xyz` and salient voxels) onto MCAP RGB frames.

Door example:

```bash
python3 scripts/verify_query_overlay.py \
  --mcap-path /home/keisuke/Downloads/experiment-2/experiment-2_0.mcap \
  --query-json outputs/door.json \
  --out-dir outputs/door_overlay \
  --out-prefix door \
  --start-frame 200 \
  --max-frames 300 \
  --frame-stride 5 \
  --frame-score-mode inverse_depth \
  --selection-mode top \
  --top-frames 5 \
  --min-frame-gap 50
```

This ranks sampled frames by salience score and saves the top N overlays.
Use `--frame-stride` to trade compute for coverage.
Use `--selection-mode all` (or `--save-all-visible`) to export every visible frame.

## Notes

- MCAP mode expects compressed image/depth topics by default:
  - `/base/camera/color/image_raw/compressed`
  - `/base/camera/aligned_depth_to_color/image_raw/compressedDepth`
- For TF pose extraction, default is `target_frame=auto` and
  `pose_frame=base_footprint`.
- If TF is sparse, tune `--pose-slop` or set `--pose-fallback last`.
