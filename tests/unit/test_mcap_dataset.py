from __future__ import annotations

from types import SimpleNamespace
import struct

import cv2
import numpy as np
import pytest
import torch

from rayfronts.datasets.mcap import (
  McapRos2Dataset,
  _find_png_start,
  decode_compressed_depth_to_meters,
  tf_to_matrix,
)


def test_find_png_start_prefers_magic_and_falls_back_to_header():
  png_payload = b"\x89PNG\r\n\x1a\n" + b"abc"
  assert _find_png_start(b"header" + png_payload) == 6
  assert _find_png_start(b"123456789012rest") == 12


def test_decode_compressed_depth_16uc1_with_header():
  depth_raw = np.array([[0, 1000], [2500, 5000]], dtype=np.uint16)
  ok, enc = cv2.imencode(".png", depth_raw)
  assert ok
  header = struct.pack("<iff", 0, 0.0, 0.0)
  msg = SimpleNamespace(
    format="16UC1; compressedDepth",
    data=header + enc.tobytes(),
  )
  depth_m = decode_compressed_depth_to_meters(msg, depth_scale=0.001)
  assert depth_m.shape == (2, 2)
  assert np.isnan(depth_m[0, 0])
  assert depth_m[0, 1] == pytest.approx(1.0)
  assert depth_m[1, 0] == pytest.approx(2.5)
  assert depth_m[1, 1] == pytest.approx(5.0)


def test_tf_to_matrix_builds_valid_rigid_transform():
  transform = SimpleNamespace(
    transform=SimpleNamespace(
      translation=SimpleNamespace(x=1.0, y=2.0, z=3.0),
      rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
  )
  mat = tf_to_matrix(transform)
  assert mat.shape == (4, 4)
  assert np.allclose(mat[:3, :3], np.eye(3))
  assert np.allclose(mat[:3, 3], np.array([1.0, 2.0, 3.0]))
  assert np.allclose(mat[3], np.array([0.0, 0.0, 0.0, 1.0]))


def _make_dataset_stub():
  ds = McapRos2Dataset.__new__(McapRos2Dataset)
  ds.pose_frame = "auto"
  ds.target_frame = "auto"
  ds._camera_frame = "camera_color_optical_frame"
  ds.pose_slop_ns = int(0.2 * 1e9)
  ds._src2rdf_transform = torch.eye(4, dtype=torch.float32)
  ds._tf_dynamic_samples = dict()
  ds._tf_dynamic_ns = dict()
  ds._tf_dynamic_4x4 = dict()
  ds._tf_static_4x4 = dict()
  ds._tf_neighbors = dict()
  ds._pose_path_frames = list()
  ds._resolved_target_frame = None
  ds._resolved_pose_frame = None
  return ds


def test_mcap_auto_frame_resolution_prefers_camera_frame_path():
  ds = _make_dataset_stub()
  ds._tf_dynamic_ns = {
    ("map", "body"): [0, 10],
  }
  ds._tf_dynamic_4x4 = {
    ("map", "body"): [torch.eye(4), torch.eye(4)],
  }
  ds._tf_static_4x4 = {
    ("body", "camera_color_optical_frame"): torch.eye(4),
  }
  ds._tf_neighbors = {
    "map": ["body"],
    "body": ["camera_color_optical_frame", "map"],
    "camera_color_optical_frame": ["body"],
  }

  target, pose, path = ds._resolve_frames()
  assert target == "map"
  assert pose == "camera_color_optical_frame"
  assert path == ["map", "body", "camera_color_optical_frame"]


def test_lookup_pose_composes_dynamic_and_static_chain():
  ds = _make_dataset_stub()
  t_map_body = torch.eye(4, dtype=torch.float32)
  t_map_body[0, 3] = 1.0
  t_body_cam = torch.eye(4, dtype=torch.float32)
  t_body_cam[2, 3] = 1.0

  ds._tf_dynamic_ns = {("map", "body"): [100]}
  ds._tf_dynamic_4x4 = {("map", "body"): [t_map_body]}
  ds._tf_static_4x4 = {("body", "camera_color_optical_frame"): t_body_cam}
  ds._pose_path_frames = ["map", "body", "camera_color_optical_frame"]

  pose = ds._lookup_pose(ts_ns=105)
  assert pose is not None
  assert torch.allclose(pose[:3, 3], torch.tensor([1.0, 0.0, 1.0]))


def test_lookup_pose_returns_none_when_dynamic_sample_out_of_slop():
  ds = _make_dataset_stub()
  ds.pose_slop_ns = 5
  ds._tf_dynamic_ns = {("map", "body"): [100]}
  ds._tf_dynamic_4x4 = {("map", "body"): [torch.eye(4)]}
  ds._tf_static_4x4 = {}
  ds._pose_path_frames = ["map", "body"]

  pose = ds._lookup_pose(ts_ns=200)
  assert pose is None
