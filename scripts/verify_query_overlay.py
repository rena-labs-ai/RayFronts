#!/usr/bin/env python3
"""Project semantic query 3D results into MCAP RGB frames for visual checking."""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from rayfronts.datasets.mcap import McapRos2Dataset


Color = Tuple[int, int, int]


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser("Verify semantic query overlay")
  parser.add_argument("--mcap-path", required=True, type=str)
  parser.add_argument("--query-json", required=True, type=str,
                      help="Path to semantic_query_cli JSON output.")
  parser.add_argument("--out-dir", default="outputs/verify_overlay", type=str)
  parser.add_argument("--out-prefix", default="overlay", type=str)
  parser.add_argument("--max-salient", default=5, type=int)
  parser.add_argument("--max-frames", default=-1, type=int)
  parser.add_argument("--start-frame", default=0, type=int,
                      help="Ignore frames before this index.")
  parser.add_argument("--frame-stride", default=1, type=int,
                      help="Evaluate every Nth frame to reduce cost.")
  parser.add_argument("--selection-mode", default="top",
                      choices=["top", "first", "all"],
                      help="How to choose output frames.")
  parser.add_argument("--top-frames", default=3, type=int,
                      help="How many top-scoring frames to save in top mode.")
  parser.add_argument("--top-pool-multiplier", default=20, type=int,
                      help="Candidate pool size multiplier for top mode.")
  parser.add_argument("--min-frame-gap", default=0, type=int,
                      help="Minimum frame distance between selected top frames.")
  parser.add_argument("--min-visible-points", default=1, type=int,
                      help="Minimum projected points that must be in frame.")
  parser.add_argument("--depth-consistency-tol", default=0.75, type=float,
                      help="Depth agreement threshold in meters. <=0 disables.")
  parser.add_argument("--frame-score-mode", default="inverse_depth",
                      choices=["sum", "inverse_depth"],
                      help="Frame scoring: plain sum or score weighted by 1/depth.")
  parser.add_argument("--min-frame-score", default=1e-6, type=float,
                      help="Minimum frame salience score to keep. Set 0 to disable.")
  parser.add_argument("--save-all-visible", action="store_true",
                      help="Deprecated alias for --selection-mode all.")
  parser.add_argument("--circle-radius", default=8, type=int)
  parser.add_argument("--line-thickness", default=2, type=int)

  parser.add_argument("--rgb-topic", type=str,
                      default="/base/camera/color/image_raw/compressed")
  parser.add_argument("--depth-topic", type=str,
                      default="/base/camera/aligned_depth_to_color/image_raw/compressedDepth")
  parser.add_argument("--camera-info-topic", type=str,
                      default="/base/camera/color/camera_info")
  parser.add_argument("--tf-topic", type=str, default="/tf")
  parser.add_argument("--tf-static-topic", type=str, default="/tf_static")
  parser.add_argument("--target-frame", type=str, default="auto")
  parser.add_argument("--pose-frame", type=str, default="auto")
  parser.add_argument("--pose-slop", type=float, default=0.2)
  parser.add_argument("--pose-fallback", type=str,
                      choices=["drop", "last", "identity"], default="last")
  parser.add_argument("--sync-slop", type=float, default=0.05)
  parser.add_argument("--depth-scale", type=float, default=0.001)
  parser.add_argument("--src-coord-system", type=str, default="flu")
  return parser


def parse_points(query_data: Dict, max_salient: int) -> List[Dict]:
  if "estimated_xyz" not in query_data:
    raise ValueError("query JSON must contain 'estimated_xyz'.")

  points: List[Dict] = list()
  est = np.asarray(query_data["estimated_xyz"], dtype=np.float32).reshape(3)

  salient = query_data.get("salient_voxels", [])
  salient_scores = list()
  for voxel in salient[:max_salient]:
    salient_scores.append(float(voxel.get("score", 1.0)))
  est_weight = float(np.median(salient_scores)) if len(salient_scores) > 0 else 1.0
  points.append(dict(label="estimate", xyz=est, color=(0, 0, 255),
                     weight=est_weight))  # red in BGR

  cluster_xyz = query_data.get("cluster_mean_xyz")
  if cluster_xyz is not None:
    cluster_xyz = np.asarray(cluster_xyz, dtype=np.float32).reshape(3)
    points.append(dict(
      label="cluster_mean",
      xyz=cluster_xyz,
      color=(0, 165, 255),  # orange in BGR
      weight=est_weight,
    ))

  for i, voxel in enumerate(salient[:max_salient]):
    xyz = np.asarray(voxel["xyz"], dtype=np.float32).reshape(3)
    points.append(dict(
      label=f"salient_{i+1}",
      xyz=xyz,
      color=(0, 255, 255),  # yellow in BGR
      weight=float(voxel.get("score", 1.0)),
    ))
  return points


def project_world_to_uv(point_world_xyz: np.ndarray,
                        pose_4x4: np.ndarray,
                        intrinsics_3x3: np.ndarray) -> Optional[Tuple[int, int, float]]:
  point_world = np.array(
    [point_world_xyz[0], point_world_xyz[1], point_world_xyz[2], 1.0],
    dtype=np.float32)
  world_to_cam = np.linalg.inv(pose_4x4)
  point_cam = world_to_cam @ point_world
  if point_cam[2] <= 0:
    return None

  uvw = intrinsics_3x3 @ point_cam[:3]
  if uvw[2] == 0:
    return None

  u = float(uvw[0] / uvw[2])
  v = float(uvw[1] / uvw[2])
  if not np.isfinite(u) or not np.isfinite(v):
    return None
  return int(round(u)), int(round(v)), float(point_cam[2])


def evaluate_frame(img_bgr: np.ndarray,
                   depth_m: np.ndarray,
                   points: List[Dict],
                   pose_4x4: np.ndarray,
                   intrinsics_3x3: np.ndarray,
                   circle_radius: int,
                   line_thickness: int,
                   depth_consistency_tol: float,
                   frame_score_mode: str,
                   ) -> Tuple[np.ndarray, List[str], List[str], float, float]:
  out = img_bgr.copy()
  visible_labels = list()
  depth_consistent_labels = list()
  frame_score = 0.0
  visible_score = 0.0
  h, w = out.shape[:2]

  for point in points:
    label = point["label"]
    pxyz = point["xyz"]
    color = point["color"]
    weight = float(point["weight"])
    proj = project_world_to_uv(pxyz, pose_4x4, intrinsics_3x3)
    if proj is None:
      continue
    u, v, cam_depth = proj
    if not (0 <= u < w and 0 <= v < h):
      continue
    visible_labels.append(label)

    depth_consistent = True
    if depth_consistency_tol > 0:
      pix_depth = float(depth_m[v, u])
      if np.isfinite(pix_depth) and pix_depth > 0:
        depth_consistent = abs(pix_depth - cam_depth) <= depth_consistency_tol
      else:
        depth_consistent = False

    draw_color = color if (depth_consistency_tol <= 0 or depth_consistent) else (255, 0, 255)
    tag = label if depth_consistency_tol <= 0 or depth_consistent else f"{label}x"
    if frame_score_mode == "sum":
      score_term = weight
    else:
      score_term = weight / max(cam_depth, 1e-3)
    visible_score += score_term

    if depth_consistent or depth_consistency_tol <= 0:
      depth_consistent_labels.append(label)
      frame_score += score_term

    cv2.circle(out, (u, v), circle_radius, draw_color, line_thickness)
    cv2.putText(out, tag, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, draw_color, 1, cv2.LINE_AA)

  cv2.putText(
    out,
    f"score={frame_score:.3f} visScore={visible_score:.3f} vis={len(visible_labels)} cons={len(depth_consistent_labels)}",
    (10, 24),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (255, 255, 255),
    2,
    cv2.LINE_AA,
  )
  return out, visible_labels, depth_consistent_labels, frame_score, visible_score


def save_overlay_image(img: np.ndarray,
                       out_dir: Path,
                       out_prefix: str,
                       frame_idx: int,
                       score: float,
                       rank: Optional[int] = None) -> Path:
  score_tag = f"{score:.3f}".replace(".", "p")
  if rank is None:
    name = f"{out_prefix}_frame_{frame_idx:06d}_score_{score_tag}.png"
  else:
    name = f"{out_prefix}_rank_{rank:02d}_frame_{frame_idx:06d}_score_{score_tag}.png"
  out_path = out_dir / name
  cv2.imwrite(str(out_path), img)
  return out_path


def main(argv=None) -> int:
  args = build_parser().parse_args(argv)
  if args.start_frame < 0:
    raise ValueError("--start-frame must be >= 0.")
  if args.frame_stride <= 0:
    raise ValueError("--frame-stride must be > 0.")
  if args.top_frames <= 0:
    raise ValueError("--top-frames must be > 0.")
  if args.top_pool_multiplier <= 0:
    raise ValueError("--top-pool-multiplier must be > 0.")
  if args.min_frame_gap < 0:
    raise ValueError("--min-frame-gap must be >= 0.")
  if args.save_all_visible:
    args.selection_mode = "all"

  query_json = Path(args.query_json)
  if not query_json.exists():
    raise FileNotFoundError(f"Query JSON not found: {query_json}")
  query_data = json.loads(query_json.read_text(encoding="utf-8"))
  points = parse_points(query_data, max_salient=args.max_salient)

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  dataset = McapRos2Dataset(
    path=args.mcap_path,
    rgb_topic=args.rgb_topic,
    depth_topic=args.depth_topic,
    camera_info_topic=args.camera_info_topic,
    tf_topic=args.tf_topic,
    tf_static_topic=args.tf_static_topic,
    target_frame=args.target_frame,
    pose_frame=args.pose_frame,
    src_coord_system=args.src_coord_system,
    depth_scale=args.depth_scale,
    sync_slop_s=args.sync_slop,
    pose_slop_s=args.pose_slop,
    pose_fallback=args.pose_fallback,
  )

  intrinsics = dataset.intrinsics_3x3.detach().cpu().numpy().astype(np.float32)
  max_frames = args.max_frames if args.max_frames > 0 else None
  saved = 0
  summary = dict(
    query=query_data.get("query"),
    query_json=str(query_json),
    mcap_path=args.mcap_path,
    selection_mode=args.selection_mode,
    start_frame=args.start_frame,
    frame_stride=args.frame_stride,
    frame_score_mode=args.frame_score_mode,
    min_frame_gap=args.min_frame_gap,
    depth_consistency_tol=args.depth_consistency_tol,
    selected_frames=list(),
    scanned_frames=0,
    visible_frames=0,
    depth_consistent_frames=0,
    fallback_frames=0,
  )
  top_heap = list()
  top_pool_size = args.top_frames * args.top_pool_multiplier

  for i, frame in enumerate(dataset):
    if max_frames is not None and i >= max_frames:
      break
    if i < args.start_frame:
      continue
    if i % args.frame_stride != 0:
      continue
    rgb = frame["rgb_img"].detach().cpu().permute(1, 2, 0).numpy()
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    depth_m = frame["depth_img"].detach().cpu().squeeze(0).numpy().astype(np.float32)
    pose = frame["pose_4x4"].detach().cpu().numpy().astype(np.float32)

    summary["scanned_frames"] += 1

    over, visible, consistent, frame_score, visible_score = evaluate_frame(
      img_bgr=bgr,
      depth_m=depth_m,
      points=points,
      pose_4x4=pose,
      intrinsics_3x3=intrinsics,
      circle_radius=args.circle_radius,
      line_thickness=args.line_thickness,
      depth_consistency_tol=args.depth_consistency_tol,
      frame_score_mode=args.frame_score_mode,
    )

    if len(visible) < args.min_visible_points:
      continue
    summary["visible_frames"] += 1
    if len(consistent) > 0:
      summary["depth_consistent_frames"] += 1

    score_source = "depth_consistent"
    if frame_score > 0:
      effective_score = frame_score
    else:
      effective_score = visible_score
      score_source = "visible_fallback"
      summary["fallback_frames"] += 1

    if effective_score < args.min_frame_score:
      continue

    entry = dict(
      frame_idx=i,
      frame_score=round(float(frame_score), 6),
      visible_score=round(float(visible_score), 6),
      effective_score=round(float(effective_score), 6),
      score_source=score_source,
      visible_labels=visible,
      depth_consistent_labels=consistent,
    )

    if args.selection_mode == "all":
      out_path = save_overlay_image(
        img=over,
        out_dir=out_dir,
        out_prefix=args.out_prefix,
        frame_idx=i,
        score=effective_score,
      )
      print(f"saved {out_path} visible={','.join(visible)} "
            f"score={effective_score:.3f} ({score_source})")
      summary["selected_frames"].append(dict(path=str(out_path), **entry))
      saved += 1
      continue

    if args.selection_mode == "first":
      out_path = save_overlay_image(
        img=over,
        out_dir=out_dir,
        out_prefix=args.out_prefix,
        frame_idx=i,
        score=effective_score,
      )
      print(f"saved {out_path} visible={','.join(visible)} "
            f"score={effective_score:.3f} ({score_source})")
      summary["selected_frames"].append(dict(path=str(out_path), **entry))
      saved = 1
      break

    # Prefer later frames for score ties to avoid startup bias.
    key = (effective_score, i)
    item = (key, over, entry)
    if len(top_heap) < top_pool_size:
      heapq.heappush(top_heap, item)
    elif key > top_heap[0][0]:
      heapq.heapreplace(top_heap, item)

  if args.selection_mode == "top":
    ranked = sorted(top_heap, key=lambda x: x[0], reverse=True)
    selected = list()
    selected_frame_idx = list()
    for _, image, entry in ranked:
      if len(selected) >= args.top_frames:
        break
      if args.min_frame_gap > 0:
        too_close = any(abs(entry["frame_idx"] - j) < args.min_frame_gap
                        for j in selected_frame_idx)
        if too_close:
          continue
      selected.append((image, entry))
      selected_frame_idx.append(entry["frame_idx"])

    if len(selected) < args.top_frames:
      for _, image, entry in ranked:
        if len(selected) >= args.top_frames:
          break
        if entry["frame_idx"] in selected_frame_idx:
          continue
        selected.append((image, entry))
        selected_frame_idx.append(entry["frame_idx"])

    for r, (image, entry) in enumerate(selected, start=1):
      out_path = save_overlay_image(
        img=image,
        out_dir=out_dir,
        out_prefix=args.out_prefix,
        frame_idx=entry["frame_idx"],
        score=entry["effective_score"],
        rank=r,
      )
      print("saved", out_path, f"score={entry['effective_score']:.3f}",
            f"source={entry['score_source']}",
            f"visible={','.join(entry['visible_labels'])}")
      summary["selected_frames"].append(dict(rank=r, path=str(out_path), **entry))
    saved = len(selected)

  if saved == 0:
    raise RuntimeError("No visible overlay points were found in scanned frames. "
                       f"scanned={summary['scanned_frames']} visible={summary['visible_frames']} "
                       f"consistent={summary['depth_consistent_frames']}. "
                       "Try increasing --max-frames, lowering --frame-stride, "
                       "or rerun semantic_query_cli with larger --max-frames.")

  summary_path = out_dir / f"{args.out_prefix}_summary.json"
  summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(f"saved summary {summary_path}")
  print(f"done: saved {saved} overlay image(s) to {out_dir}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
