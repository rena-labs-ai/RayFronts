from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_cli_raw(args):
  cmd = [sys.executable, "scripts/semantic_query_cli.py"] + args
  return subprocess.run(
    cmd,
    cwd=str(REPO_ROOT),
    text=True,
    capture_output=True,
    check=False,
  )


def _run_cli(args):
  proc = _run_cli_raw(args)
  if proc.returncode != 0:
    raise AssertionError(
      f"CLI failed with code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
  try:
    return json.loads(proc.stdout)
  except json.JSONDecodeError as exc:
    raise AssertionError(
      f"CLI did not output valid JSON.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    ) from exc


def test_cli_dataset_mode_with_stub_dataset_and_encoder():
  out = _run_cli([
    "--source", "dataset",
    "--dataset-class", "tests.stubs.semantic_stubs.ToyDataset",
    "--dataset-kwargs", '{"num_frames": 4}',
    "--object", "door",
    "--top-k", "5",
    "--max-frames", "4",
    "--encoder-class", "tests.stubs.semantic_stubs.ToyEncoder",
    "--mapper-class", "rayfronts.mapping.semantic_voxel_map.SemanticVoxelMap",
    "--mapper-kwargs",
    '{"vox_size": 0.2, "vox_accum_period": 1, "max_pts_per_frame": -1, "device": "cpu"}',
  ])
  assert out["query"] == "door"
  assert out["processed_frames"] == 4
  assert 1 <= out["top_k"] <= 5
  assert len(out["salient_voxels"]) == out["top_k"]
  assert len(out["estimated_xyz"]) == 3
  assert len(out["median_xyz"]) == 3
  assert len(out["cluster_mean_xyz"]) == 3
  assert out["cluster_size"] >= 1
  assert out["median_distance_m"] >= 0


def test_cli_device_cuda_falls_back_when_unavailable():
  out = _run_cli([
    "--source", "dataset",
    "--dataset-class", "tests.stubs.semantic_stubs.ToyDataset",
    "--dataset-kwargs", '{"num_frames": 2}',
    "--object", "door",
    "--top-k", "5",
    "--max-frames", "2",
    "--device", "cuda",
    "--encoder-class", "tests.stubs.semantic_stubs.ToyEncoder",
    "--encoder-kwargs", '{"device":"cuda"}',
    "--mapper-class", "rayfronts.mapping.semantic_voxel_map.SemanticVoxelMap",
    "--mapper-kwargs",
    '{"vox_size": 0.2, "vox_accum_period": 1, "max_pts_per_frame": -1, "device": "cuda"}',
  ])
  assert out["processed_frames"] == 2
  assert out["query"] == "door"


def test_cli_dataset_mode_with_existing_rosnpy_dataset(tmp_path: Path):
  npz_path = tmp_path / "tiny_ros.npz"
  num_frames = 3
  rgb_img = np.zeros((num_frames, 4, 4, 3), dtype=np.uint8)
  rgb_img[:, 1:3, 1:3, 0] = 255
  disparity_img = np.ones((num_frames, 4, 4), dtype=np.float32) * 50.0
  min_disp = np.ones((num_frames,), dtype=np.float32) * 1.0
  max_disp = np.ones((num_frames,), dtype=np.float32) * 100.0
  focal = np.ones((num_frames,), dtype=np.float32) * 40.0
  baseline = np.ones((num_frames,), dtype=np.float32) * 0.1
  pose_t = np.stack([np.array([0.1 * i, 0.0, 0.0], dtype=np.float32)
                     for i in range(num_frames)], axis=0)
  pose_q = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (num_frames, 1))
  intrinsics_flat = np.tile(
    np.array([[40.0, 0.0, 1.5, 0.0, 40.0, 1.5, 0.0, 0.0, 1.0]], dtype=np.float32),
    (num_frames, 1),
  )
  np.savez_compressed(
    npz_path,
    rgb_img=rgb_img,
    disparity_img=disparity_img,
    min_disparity=min_disp,
    max_disparity=max_disp,
    focal_length=focal,
    stereo_baseline=baseline,
    pose_t=pose_t,
    pose_q_wxyz=pose_q,
    intrinsics_3x3=intrinsics_flat,
  )

  out = _run_cli([
    "--source", "dataset",
    "--dataset-class", "rayfronts.datasets.ros.RosnpyDataset",
    "--dataset-kwargs", json.dumps({"path": str(npz_path)}),
    "--object", "door",
    "--top-k", "5",
    "--max-frames", "3",
    "--encoder-class", "tests.stubs.semantic_stubs.ToyEncoder",
    "--mapper-class", "rayfronts.mapping.semantic_voxel_map.SemanticVoxelMap",
    "--mapper-kwargs",
    '{"vox_size": 0.2, "vox_accum_period": 1, "max_pts_per_frame": -1, "device": "cpu"}',
  ])
  assert out["processed_frames"] == 3
  assert out["query"] == "door"
  assert out["top_k"] >= 1


def test_cli_mcap_mode_with_experiment2_recording_if_available():
  mcap_path = Path("/home/keisuke/Downloads/experiment-2/experiment-2_0.mcap")
  if not mcap_path.exists():
    pytest.skip("experiment-2 MCAP is not available on this machine.")

  out = _run_cli([
    "--source", "mcap",
    "--mcap-path", str(mcap_path),
    "--object", "door",
    "--top-k", "5",
    "--max-frames", "3",
    "--pose-fallback", "last",
    "--encoder-class", "tests.stubs.semantic_stubs.ToyEncoder",
    "--mapper-class", "rayfronts.mapping.semantic_voxel_map.SemanticVoxelMap",
    "--mapper-kwargs",
    '{"vox_size": 0.4, "vox_accum_period": 1, "max_pts_per_frame": 300, "device": "cpu"}',
  ])
  assert out["processed_frames"] == 3
  assert out["source_path"] == str(mcap_path)
  assert out["query"] == "door"
  assert len(out["salient_voxels"]) >= 1


def test_cli_reports_missing_optional_encoder_dependency():
  proc = _run_cli_raw([
    "--source", "dataset",
    "--dataset-class", "tests.stubs.semantic_stubs.ToyDataset",
    "--dataset-kwargs", '{"num_frames": 2}',
    "--object", "door",
    "--max-frames", "1",
    "--encoder-class", "tests.stubs.failing_encoders.MissingDependencyEncoder",
    "--mapper-class", "rayfronts.mapping.semantic_voxel_map.SemanticVoxelMap",
    "--mapper-kwargs",
    '{"vox_size": 0.2, "vox_accum_period": 1, "max_pts_per_frame": -1, "device": "cpu"}',
  ])
  assert proc.returncode != 0
  assert "optional dependency 'timm'" in proc.stderr.lower()
  assert "pip install timm" in proc.stderr.lower()


def test_cli_maps_module_name_to_install_package():
  proc = _run_cli_raw([
    "--source", "dataset",
    "--dataset-class", "tests.stubs.semantic_stubs.ToyDataset",
    "--dataset-kwargs", '{"num_frames": 2}',
    "--object", "door",
    "--max-frames", "1",
    "--encoder-class", "tests.stubs.failing_encoders.MissingOpenClipEncoder",
    "--mapper-class", "rayfronts.mapping.semantic_voxel_map.SemanticVoxelMap",
    "--mapper-kwargs",
    '{"vox_size": 0.2, "vox_accum_period": 1, "max_pts_per_frame": -1, "device": "cpu"}',
  ])
  assert proc.returncode != 0
  assert "optional dependency 'open_clip'" in proc.stderr.lower()
  assert "pip install open_clip_torch" in proc.stderr.lower()


def test_cli_auto_sets_radio_spatial_alignment_flag():
  out = _run_cli([
    "--source", "dataset",
    "--dataset-class", "tests.stubs.semantic_stubs.ToyDataset",
    "--dataset-kwargs", '{"num_frames": 2}',
    "--object", "door",
    "--max-frames", "1",
    "--encoder-class", "tests.stubs.radio.RadioEncoder",
    "--encoder-kwargs", '{"device":"cpu","lang_model":"clip","return_radio_features":true}',
    "--mapper-class", "rayfronts.mapping.semantic_voxel_map.SemanticVoxelMap",
    "--mapper-kwargs",
    '{"vox_size": 0.2, "vox_accum_period": 1, "max_pts_per_frame": -1, "device": "cpu"}',
  ])
  assert out["processed_frames"] == 1


def test_cli_can_query_multiple_objects_after_single_map_build():
  out = _run_cli([
    "--source", "dataset",
    "--dataset-class", "tests.stubs.semantic_stubs.ToyDataset",
    "--dataset-kwargs", '{"num_frames": 3}',
    "--objects", "door", "wall",
    "--top-k", "5",
    "--max-frames", "3",
    "--encoder-class", "tests.stubs.semantic_stubs.ToyEncoder",
    "--mapper-class", "rayfronts.mapping.semantic_voxel_map.SemanticVoxelMap",
    "--mapper-kwargs",
    '{"vox_size": 0.2, "vox_accum_period": 1, "max_pts_per_frame": -1, "device": "cpu"}',
  ])
  assert out["processed_frames"] == 3
  assert "queries" in out
  assert len(out["queries"]) == 2
  names = [q["query"] for q in out["queries"]]
  assert names == ["door", "wall"]
  assert all("cluster_mean_xyz" in q for q in out["queries"])


def test_cli_saved_map_mode_supports_multi_query_and_per_query_json(tmp_path: Path):
  map_path = tmp_path / "toy_map.pt"
  per_query_dir = tmp_path / "per_query"

  build = _run_cli_raw([
    "--source", "dataset",
    "--dataset-class", "tests.stubs.semantic_stubs.ToyDataset",
    "--dataset-kwargs", '{"num_frames": 3}',
    "--object", "door",
    "--top-k", "5",
    "--max-frames", "3",
    "--encoder-class", "tests.stubs.semantic_stubs.ToyEncoder",
    "--mapper-class", "rayfronts.mapping.semantic_voxel_map.SemanticVoxelMap",
    "--mapper-kwargs",
    '{"vox_size": 0.2, "vox_accum_period": 1, "max_pts_per_frame": -1, "device": "cpu"}',
    "--save-map", str(map_path),
  ])
  assert build.returncode == 0, build.stderr
  assert map_path.exists()
  assert Path(str(map_path) + ".meta.json").exists()

  out = _run_cli([
    "--source", "saved_map",
    "--load-map", str(map_path),
    "--objects", "door", "wall",
    "--per-query-output-dir", str(per_query_dir),
  ])
  assert "queries" in out
  assert len(out["queries"]) == 2
  assert sorted(p["query"] for p in out["queries"]) == ["door", "wall"]
  assert "per_query_json" in out
  assert len(out["per_query_json"]) == 2
  for p in out["per_query_json"]:
    pp = Path(p)
    assert pp.exists()
    payload = json.loads(pp.read_text(encoding="utf-8"))
    assert "estimated_xyz" in payload
    assert "salient_voxels" in payload
