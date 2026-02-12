"""Test stubs for semantic query CLI integration tests."""

from __future__ import annotations

from typing import Dict, List

import torch

from rayfronts.image_encoders.base import LangSpatialImageEncoder


class ToyDataset:
  """Deterministic RGBD dataset with a red semantic region."""

  def __init__(self, num_frames: int = 4):
    self.num_frames = num_frames
    self.intrinsics_3x3 = torch.tensor([
      [20.0, 0.0, 1.5],
      [0.0, 20.0, 1.5],
      [0.0, 0.0, 1.0],
    ], dtype=torch.float32)

  def __iter__(self):
    for i in range(self.num_frames):
      rgb = torch.zeros((3, 4, 4), dtype=torch.float32)
      rgb[0, 1:3, 1:3] = 1.0  # red patch
      rgb[1, :, :] = 0.05
      rgb[2, :, :] = 0.05

      depth = torch.ones((1, 4, 4), dtype=torch.float32) * (1.0 + 0.1 * i)
      pose = torch.eye(4, dtype=torch.float32)
      pose[0, 3] = 0.2 * i
      pose[2, 3] = 0.1

      yield dict(rgb_img=rgb, depth_img=depth, pose_4x4=pose)


class ToyEncoder(LangSpatialImageEncoder):
  """Simple language-aligned encoder: features are normalized RGB."""

  def __init__(self, device: str = "cpu"):
    super().__init__(device=device)
    self._label_to_vec: Dict[str, torch.Tensor] = {
      "door": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
      "wall": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
      "floor": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
    }

  def is_compatible_size(self, h: int, w: int) -> bool:
    return True

  def get_nearest_size(self, h, w):
    return h, w

  def encode_image_to_feat_map(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
    feat = rgb_image[:, :3, :, :] + 1e-3
    feat = torch.nn.functional.normalize(feat, dim=1)
    return feat

  def encode_labels(self, labels: List[str]) -> torch.FloatTensor:
    feats = list()
    for label in labels:
      v = self._label_to_vec.get(label.lower(), torch.ones(3, dtype=torch.float32))
      v = torch.nn.functional.normalize(v.unsqueeze(0), dim=1).squeeze(0)
      feats.append(v)
    return torch.stack(feats, dim=0).to(self.device)

  def encode_prompts(self, prompts: List[str]) -> torch.FloatTensor:
    return self.encode_labels(prompts)

  def align_spatial_features_with_language(self, features: torch.FloatTensor) -> torch.FloatTensor:
    return features
