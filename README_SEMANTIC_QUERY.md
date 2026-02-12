# Semantic Query CLI

`scripts/semantic_query_cli.py` supports two usage models:
- map and query in the same run
- build/save map once, then query many objects from that saved map

Estimation rule:
- select top `k` salient voxels by semantic similarity (`--top-k`, default `5`)
- compute one final object point as weighted cluster mean (`estimated_xyz`)
- also report `median_xyz` for reference

## Model A: Map And Query In One Run

Use this when you want a quick single-pass result.

Single object:

```bash
python3 scripts/semantic_query_cli.py \
  --source mcap \
  --mcap-path /home/keisuke/Downloads/experiment-2/experiment-2_0.mcap \
  --object door \
  --query-type labels \
  --top-k 5 \
  --cluster-radius 1.0 \
  --max-frames -1 \
  --frame-skip 20 \
  --pose-fallback last \
  --device cpu \
  --encoder-class rayfronts.image_encoders.radio.RadioEncoder \
  --encoder-kwargs '{"device":"cpu","model_version":"radio_v2.5-b","lang_model":"clip","return_radio_features":true}' \
  --mapper-class rayfronts.mapping.semantic_voxel_map.SemanticVoxelMap \
  --mapper-kwargs '{"vox_size":0.3,"vox_accum_period":1,"max_pts_per_frame":800,"device":"cpu"}' \
  --output-json outputs/door_online.json
```

Multiple objects while map is built once in that same run:

```bash
python3 scripts/semantic_query_cli.py \
  --source mcap \
  --mcap-path /home/keisuke/Downloads/experiment-2/experiment-2_0.mcap \
  --objects door "trash can" \
  --query-type labels \
  --top-k 5 \
  --cluster-radius 1.0 \
  --max-frames -1 \
  --frame-skip 20 \
  --pose-fallback last \
  --device cpu \
  --encoder-class rayfronts.image_encoders.radio.RadioEncoder \
  --encoder-kwargs '{"device":"cpu","model_version":"radio_v2.5-b","lang_model":"clip","return_radio_features":true}' \
  --mapper-class rayfronts.mapping.semantic_voxel_map.SemanticVoxelMap \
  --mapper-kwargs '{"vox_size":0.3,"vox_accum_period":1,"max_pts_per_frame":800,"device":"cpu"}' \
  --per-query-output-dir outputs/online_queries \
  --output-json outputs/online_queries/all_queries.json
```

Verify one query JSON with overlays:

```bash
python3 scripts/verify_query_overlay.py \
  --mcap-path /home/keisuke/Downloads/experiment-2/experiment-2_0.mcap \
  --query-json outputs/online_queries/door.json \
  --out-dir outputs/door_overlay_online \
  --out-prefix door \
  --max-frames -1 \
  --frame-stride 10 \
  --selection-mode top \
  --top-frames 5 \
  --min-frame-gap 80 \
  --frame-score-mode inverse_depth \
  --pose-fallback last
```

## Model B: Build Once, Save Map, Query Many Times

Use this when you want to run many objects without remapping.

Step 1: build and save map once:

```bash
python3 scripts/semantic_query_cli.py \
  --source mcap \
  --mcap-path /home/keisuke/Downloads/experiment-2/experiment-2_0.mcap \
  --object door \
  --query-type labels \
  --top-k 5 \
  --cluster-radius 1.0 \
  --max-frames -1 \
  --frame-skip 20 \
  --pose-fallback last \
  --device cpu \
  --encoder-class rayfronts.image_encoders.radio.RadioEncoder \
  --encoder-kwargs '{"device":"cpu","model_version":"radio_v2.5-b","lang_model":"clip","return_radio_features":true}' \
  --mapper-class rayfronts.mapping.semantic_voxel_map.SemanticVoxelMap \
  --mapper-kwargs '{"vox_size":0.3,"vox_accum_period":1,"max_pts_per_frame":800,"device":"cpu"}' \
  --save-map outputs/experiment2_map.pt
```

Step 2: query saved map for multiple objects:

```bash
python3 scripts/semantic_query_cli.py \
  --source saved_map \
  --load-map outputs/experiment2_map.pt \
  --objects door "trash can" \
  --top-k 5 \
  --cluster-radius 1.0 \
  --per-query-output-dir outputs/saved_map_queries \
  --output-json outputs/saved_map_queries/all_queries.json
```

Step 3: verify each object result:

```bash
for q in door trash_can; do
  python3 scripts/verify_query_overlay.py \
    --mcap-path /home/keisuke/Downloads/experiment-2/experiment-2_0.mcap \
    --query-json outputs/saved_map_queries/${q}.json \
    --out-dir outputs/${q}_overlay_saved_map \
    --out-prefix ${q} \
    --max-frames -1 \
    --frame-stride 10 \
    --selection-mode top \
    --top-frames 5 \
    --min-frame-gap 80 \
    --frame-score-mode inverse_depth \
    --pose-fallback last
done
```

## Output JSON

Single-query output has:
- `query`
- `estimated_xyz` (`cluster_mean_xyz` equivalent)
- `median_xyz`
- `salient_voxels`
- `processed_frames`
- `run_config`

Multi-query output (`--objects ...`) has:
- `queries`: list of per-query payloads
- `processed_frames`
- `run_config`
- optional `per_query_json` paths when `--per-query-output-dir` is used

## Notes

- MCAP defaults:
  - RGB topic: `/base/camera/color/image_raw/compressed`
  - depth topic: `/base/camera/aligned_depth_to_color/image_raw/compressedDepth`
- TF defaults:
  - `target_frame=auto`
  - `pose_frame=auto`
- If TF is sparse, increase `--pose-slop` or use `--pose-fallback last`.
- `docker/run_offline_mcap.sh` is a local wrapper script; it calls local `python3`, it does not launch a Docker container by itself.
