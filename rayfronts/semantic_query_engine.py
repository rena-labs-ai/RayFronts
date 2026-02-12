"""Utilities for semantic querying and object position estimation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


@dataclass
class SalientVoxel:
  xyz: Tuple[float, float, float]
  score: float
  distance_m: float


@dataclass
class ObjectPositionEstimate:
  query: str
  top_k: int
  estimated_xyz: Tuple[float, float, float]
  median_xyz: Tuple[float, float, float]
  cluster_mean_xyz: Tuple[float, float, float]
  cluster_size: int
  reference_xyz: Tuple[float, float, float]
  median_distance_m: float
  salient_voxels: List[SalientVoxel]

  def as_dict(self) -> Dict[str, Any]:
    return dict(
      query=self.query,
      top_k=self.top_k,
      estimated_xyz=list(self.estimated_xyz),
      median_xyz=list(self.median_xyz),
      cluster_mean_xyz=list(self.cluster_mean_xyz),
      cluster_size=self.cluster_size,
      reference_xyz=list(self.reference_xyz),
      median_distance_m=self.median_distance_m,
      salient_voxels=[
        dict(
          xyz=list(v.xyz),
          score=v.score,
          distance_m=v.distance_m,
        )
        for v in self.salient_voxels
      ],
    )


def import_symbol(path: str) -> Any:
  """Imports a symbol from a dotted path `pkg.module.Symbol`."""
  mod_path, _, symbol = path.rpartition(".")
  if len(mod_path) == 0 or len(symbol) == 0:
    raise ValueError(f"Invalid import path: '{path}'")
  module = import_module(mod_path)
  try:
    return getattr(module, symbol)
  except AttributeError as exc:
    raise AttributeError(f"'{path}' could not be imported.") from exc


def _get_reference_xyz(reference_pose_4x4: Optional[torch.Tensor],
                       reference_xyz: Optional[Sequence[float]]) -> torch.Tensor:
  if reference_xyz is not None:
    if len(reference_xyz) != 3:
      raise ValueError("reference_xyz must be length 3.")
    return torch.tensor(reference_xyz, dtype=torch.float32)

  if reference_pose_4x4 is not None:
    if reference_pose_4x4.shape[-2:] != (4, 4):
      raise ValueError("reference_pose_4x4 must have shape 4x4.")
    return reference_pose_4x4[:3, 3].detach().cpu().float()

  return torch.zeros(3, dtype=torch.float32)


def _cluster_weighted_mean(top_xyz: torch.Tensor,
                           top_scores: torch.Tensor,
                           cluster_radius_m: float) -> Tuple[torch.Tensor, torch.Tensor]:
  if top_xyz.shape[0] == 1 or cluster_radius_m <= 0:
    idx = torch.arange(top_xyz.shape[0], dtype=torch.long)
    return top_xyz.mean(dim=0), idx

  dist = torch.cdist(top_xyz, top_xyz)
  adjacency = dist <= cluster_radius_m

  visited = torch.zeros(top_xyz.shape[0], dtype=torch.bool)
  components = list()
  for i in range(top_xyz.shape[0]):
    if visited[i]:
      continue
    stack = [int(i)]
    visited[i] = True
    comp = list()
    while len(stack) > 0:
      j = stack.pop()
      comp.append(j)
      neigh = torch.nonzero(adjacency[j], as_tuple=False).squeeze(-1).tolist()
      for n in neigh:
        if not visited[n]:
          visited[n] = True
          stack.append(int(n))
    components.append(torch.tensor(comp, dtype=torch.long))

  def _comp_key(comp_idx: torch.Tensor):
    s = top_scores[comp_idx]
    return (float(s.sum().item()), int(comp_idx.shape[0]), float(s.max().item()))

  best = max(components, key=_comp_key)
  comp_xyz = top_xyz[best]
  comp_scores = top_scores[best]
  weights = torch.clamp(comp_scores, min=0)
  if float(weights.sum().item()) <= 0:
    weights = torch.ones_like(comp_scores)
  mean_xyz = torch.sum(comp_xyz * weights.unsqueeze(-1), dim=0) / torch.sum(weights)
  return mean_xyz, best


def estimate_object_position(query_results: Dict[str, torch.Tensor],
                             query: str,
                             query_index: int = 0,
                             top_k: int = 5,
                             cluster_radius_m: float = 1.0,
                             reference_pose_4x4: Optional[torch.Tensor] = None,
                             reference_xyz: Optional[Sequence[float]] = None
                             ) -> ObjectPositionEstimate:
  """Estimates an object position from query similarities over voxels.

  Selects the top-k most salient voxels (highest similarity), computes the
  coordinate-wise median voxel position, and computes median distance from the
  reference position.
  """
  if top_k <= 0:
    raise ValueError("top_k must be > 0.")

  if query_results is None or "vox_xyz" not in query_results or \
     "vox_sim" not in query_results:
    raise ValueError("query_results must include 'vox_xyz' and 'vox_sim'.")

  vox_xyz = query_results["vox_xyz"]
  vox_sim = query_results["vox_sim"]
  if vox_xyz is None or vox_sim is None:
    raise ValueError("Query tensors cannot be None.")
  if vox_xyz.ndim != 2 or vox_xyz.shape[-1] != 3:
    raise ValueError("vox_xyz must have shape Nx3.")

  if vox_sim.ndim == 1:
    vox_sim = vox_sim.unsqueeze(0)
  if vox_sim.ndim != 2:
    raise ValueError("vox_sim must have shape QxN or N.")
  if query_index < 0 or query_index >= vox_sim.shape[0]:
    raise IndexError("query_index is out of bounds.")
  if vox_sim.shape[1] != vox_xyz.shape[0]:
    raise ValueError("vox_sim and vox_xyz sizes are inconsistent.")

  q_scores = vox_sim[query_index].detach().cpu().float()
  q_xyz = vox_xyz.detach().cpu().float()
  finite_mask = torch.isfinite(q_scores)
  finite_mask = finite_mask & torch.isfinite(q_xyz).all(dim=-1)
  if not torch.any(finite_mask):
    raise ValueError("No finite query scores are available for estimation.")

  finite_idx = torch.nonzero(finite_mask, as_tuple=False).squeeze(-1)
  finite_scores = q_scores[finite_idx]
  k = min(top_k, finite_scores.shape[0])
  top_local_idx = torch.topk(finite_scores, k=k, largest=True).indices
  top_global_idx = finite_idx[top_local_idx]

  top_xyz = q_xyz[top_global_idx]
  top_scores = q_scores[top_global_idx]
  ref_xyz = _get_reference_xyz(reference_pose_4x4, reference_xyz)
  dists = torch.linalg.vector_norm(top_xyz - ref_xyz.unsqueeze(0), dim=-1)
  med_xyz = torch.median(top_xyz, dim=0).values
  cluster_xyz, cluster_idx = _cluster_weighted_mean(
    top_xyz=top_xyz, top_scores=top_scores, cluster_radius_m=cluster_radius_m)
  med_dist = torch.median(dists).item()

  salient_voxels = list()
  for i in range(k):
    salient_voxels.append(SalientVoxel(
      xyz=(top_xyz[i, 0].item(), top_xyz[i, 1].item(), top_xyz[i, 2].item()),
      score=top_scores[i].item(),
      distance_m=dists[i].item(),
    ))

  return ObjectPositionEstimate(
    query=query,
    top_k=k,
    estimated_xyz=(cluster_xyz[0].item(), cluster_xyz[1].item(), cluster_xyz[2].item()),
    median_xyz=(med_xyz[0].item(), med_xyz[1].item(), med_xyz[2].item()),
    cluster_mean_xyz=(cluster_xyz[0].item(), cluster_xyz[1].item(), cluster_xyz[2].item()),
    cluster_size=int(cluster_idx.shape[0]),
    reference_xyz=(ref_xyz[0].item(), ref_xyz[1].item(), ref_xyz[2].item()),
    median_distance_m=med_dist,
    salient_voxels=salient_voxels,
  )


@torch.inference_mode()
def map_rgbd_stream(mapper: Any,
                    dataset: Iterable[Dict[str, torch.Tensor]],
                    max_frames: Optional[int] = None,
                    depth_limit: float = -1) -> Tuple[int, Optional[torch.Tensor]]:
  """Runs mapping over an iterable RGBD source and returns last pose."""
  processed = 0
  last_pose = None
  mapper_device = getattr(mapper, "device", "cpu")

  for frame in dataset:
    if frame is None:
      break
    rgb_img = frame["rgb_img"].unsqueeze(0).to(mapper_device)
    depth_img = frame["depth_img"].unsqueeze(0).to(mapper_device)
    pose_4x4 = frame["pose_4x4"].unsqueeze(0).to(mapper_device)
    kwargs = dict()
    if "confidence_map" in frame:
      kwargs["conf_map"] = frame["confidence_map"].unsqueeze(0).to(mapper_device)

    if depth_limit >= 0:
      finite_mask = torch.logical_and(torch.isfinite(depth_img),
                                      depth_img > depth_limit)
      depth_img[finite_mask] = torch.inf

    mapper.process_posed_rgbd(rgb_img, depth_img, pose_4x4, **kwargs)
    processed += 1
    last_pose = frame["pose_4x4"].detach().cpu()

    if max_frames is not None and max_frames > 0 and processed >= max_frames:
      break

  return processed, last_pose


@torch.inference_mode()
def query_object_position(mapper: Any,
                          query: str,
                          top_k: int = 5,
                          cluster_radius_m: float = 1.0,
                          query_type: str = "labels",
                          softmax: bool = False,
                          compressed: bool = False,
                          reference_pose_4x4: Optional[torch.Tensor] = None
                          ) -> ObjectPositionEstimate:
  if not hasattr(mapper, "text_query"):
    raise TypeError("Mapper does not provide text_query().")
  query_results = mapper.text_query([query], query_type=query_type,
                                    softmax=softmax, compressed=compressed)
  return estimate_object_position(
    query_results=query_results,
    query=query,
    query_index=0,
    top_k=top_k,
    cluster_radius_m=cluster_radius_m,
    reference_pose_4x4=reference_pose_4x4,
  )
