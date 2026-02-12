from __future__ import annotations

import pytest
import torch

from rayfronts.semantic_query_engine import estimate_object_position


def test_estimate_uses_top5_salient_voxels_and_median_distance():
  query_results = dict(
    vox_xyz=torch.tensor([
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [2.0, 0.0, 0.0],
      [3.0, 0.0, 0.0],
      [4.0, 0.0, 0.0],
      [5.0, 0.0, 0.0],
    ], dtype=torch.float32),
    vox_sim=torch.tensor([[0.1, 0.9, 0.8, 0.7, 0.6, 0.5]], dtype=torch.float32),
  )
  estimate = estimate_object_position(
    query_results=query_results,
    query="door",
    top_k=5,
    reference_xyz=(0.0, 0.0, 0.0),
  )
  assert estimate.top_k == 5
  assert estimate.estimated_xyz == pytest.approx((2.7142857, 0.0, 0.0))
  assert estimate.cluster_mean_xyz == pytest.approx((2.7142857, 0.0, 0.0))
  assert estimate.median_xyz == pytest.approx((3.0, 0.0, 0.0))
  assert estimate.cluster_size == 5
  assert estimate.median_distance_m == pytest.approx(3.0)
  assert len(estimate.salient_voxels) == 5
  assert estimate.salient_voxels[0].score == pytest.approx(0.9)


def test_estimate_handles_nonfinite_and_small_voxel_sets():
  query_results = dict(
    vox_xyz=torch.tensor([
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [2.0, 0.0, 0.0],
      [3.0, 0.0, 0.0],
    ], dtype=torch.float32),
    vox_sim=torch.tensor([[float("nan"), float("inf"), 0.2, -1.0]]),
  )
  estimate = estimate_object_position(
    query_results=query_results,
    query="door",
    top_k=5,
    reference_xyz=(0.0, 0.0, 0.0),
  )
  assert estimate.top_k == 2
  assert estimate.estimated_xyz == pytest.approx((2.0, 0.0, 0.0))
  assert estimate.cluster_size == 2
  assert estimate.median_xyz == pytest.approx((2.0, 0.0, 0.0))
  assert estimate.median_distance_m == pytest.approx(2.0)


def test_estimate_cluster_mean_prefers_highest_score_component():
  query_results = dict(
    vox_xyz=torch.tensor([
      [0.0, 0.0, 0.0],
      [0.1, 0.0, 0.0],
      [3.0, 0.0, 0.0],
      [3.1, 0.0, 0.0],
      [6.0, 0.0, 0.0],
    ], dtype=torch.float32),
    vox_sim=torch.tensor([[0.9, 0.8, 0.2, 0.1, 0.85]], dtype=torch.float32),
  )
  estimate = estimate_object_position(
    query_results=query_results,
    query="door",
    top_k=5,
    cluster_radius_m=0.25,
    reference_xyz=(0.0, 0.0, 0.0),
  )
  assert estimate.cluster_size == 2
  assert estimate.estimated_xyz == pytest.approx((0.0470588, 0.0, 0.0), abs=1e-5)


def test_estimate_raises_on_invalid_query_results():
  with pytest.raises(ValueError):
    estimate_object_position(query_results=dict(), query="door")

  with pytest.raises(ValueError):
    estimate_object_position(
      query_results=dict(
        vox_xyz=torch.empty(0, 3),
        vox_sim=torch.empty(1, 0),
      ),
      query="door",
    )
