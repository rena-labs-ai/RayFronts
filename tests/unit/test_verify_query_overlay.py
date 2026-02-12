from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_query_overlay.py"


def _load_overlay_module():
  spec = importlib.util.spec_from_file_location("verify_query_overlay", SCRIPT_PATH)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def test_evaluate_frame_returns_visible_score_when_depth_inconsistent():
  mod = _load_overlay_module()
  img = np.zeros((8, 8, 3), dtype=np.uint8)
  depth_m = np.ones((8, 8), dtype=np.float32) * 1.0
  pose = np.eye(4, dtype=np.float32)
  intr = np.array([
    [4.0, 0.0, 4.0],
    [0.0, 4.0, 4.0],
    [0.0, 0.0, 1.0],
  ], dtype=np.float32)
  points = [
    dict(label="salient_1",
         xyz=np.array([0.0, 0.0, 2.0], dtype=np.float32),
         color=(0, 255, 255),
         weight=1.0)
  ]

  _, visible, consistent, frame_score, visible_score = mod.evaluate_frame(
    img_bgr=img,
    depth_m=depth_m,
    points=points,
    pose_4x4=pose,
    intrinsics_3x3=intr,
    circle_radius=2,
    line_thickness=1,
    depth_consistency_tol=0.25,
    frame_score_mode="inverse_depth",
  )

  assert visible == ["salient_1"]
  assert consistent == []
  assert frame_score == 0.0
  assert visible_score > 0.0


def test_main_falls_back_to_visible_score_when_no_consistent_points(
    tmp_path: Path, monkeypatch):
  mod = _load_overlay_module()

  class FakeDataset:
    def __init__(self, *args, **kwargs):
      self.intrinsics_3x3 = torch.tensor([
        [4.0, 0.0, 4.0],
        [0.0, 4.0, 4.0],
        [0.0, 0.0, 1.0],
      ], dtype=torch.float32)

    def __iter__(self):
      for _ in range(3):
        rgb = torch.zeros((3, 8, 8), dtype=torch.float32)
        depth = torch.ones((1, 8, 8), dtype=torch.float32) * 1.0
        pose = torch.eye(4, dtype=torch.float32)
        yield dict(rgb_img=rgb, depth_img=depth, pose_4x4=pose)

  monkeypatch.setattr(mod, "McapRos2Dataset", FakeDataset)

  query_json = tmp_path / "query.json"
  query_json.write_text(json.dumps({
    "query": "door",
    "estimated_xyz": [0.0, 0.0, 2.0],
    "salient_voxels": [
      {"xyz": [0.0, 0.0, 2.0], "score": 1.0, "distance_m": 2.0}
    ],
  }), encoding="utf-8")

  out_dir = tmp_path / "overlay_out"
  rc = mod.main([
    "--mcap-path", "/tmp/fake.mcap",
    "--query-json", str(query_json),
    "--out-dir", str(out_dir),
    "--out-prefix", "door",
    "--max-frames", "3",
    "--selection-mode", "top",
    "--top-frames", "1",
    "--frame-stride", "1",
    "--depth-consistency-tol", "0.25",
    "--frame-score-mode", "inverse_depth",
    "--min-frame-score", "1e-6",
  ])

  assert rc == 0
  summary_path = out_dir / "door_summary.json"
  assert summary_path.exists()
  summary = json.loads(summary_path.read_text(encoding="utf-8"))
  assert summary["selected_frames"]
  frame = summary["selected_frames"][0]
  assert frame["score_source"] == "visible_fallback"
  assert frame["frame_score"] == 0.0
  assert frame["visible_score"] > 0.0
  assert frame["effective_score"] > 0.0

  image_path = Path(frame["path"])
  assert image_path.exists()
  img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
  assert img is not None


def test_parse_points_includes_cluster_mean_when_available():
  mod = _load_overlay_module()
  points = mod.parse_points(
    query_data={
      "estimated_xyz": [1.0, 2.0, 3.0],
      "cluster_mean_xyz": [1.5, 2.5, 3.5],
      "salient_voxels": [
        {"xyz": [1.0, 2.0, 3.0], "score": 0.8},
      ],
    },
    max_salient=5,
  )
  labels = [p["label"] for p in points]
  assert "estimate" in labels
  assert "cluster_mean" in labels
