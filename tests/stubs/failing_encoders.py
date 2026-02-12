"""Encoders used to test dependency failure handling."""

from __future__ import annotations

from typing import List

import torch

from rayfronts.image_encoders.base import LangSpatialImageEncoder


class MissingDependencyEncoder(LangSpatialImageEncoder):
  """Raises ModuleNotFoundError to emulate missing optional deps."""

  def __init__(self, device: str = "cpu"):
    super().__init__(device=device)
    exc = ModuleNotFoundError("No module named 'timm'")
    exc.name = "timm"
    raise exc

  def is_compatible_size(self, h: int, w: int) -> bool:
    return True

  def get_nearest_size(self, h, w):
    return h, w

  def encode_image_to_feat_map(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
    return rgb_image

  def encode_labels(self, labels: List[str]) -> torch.FloatTensor:
    return torch.zeros((len(labels), 1), dtype=torch.float32)

  def encode_prompts(self, prompts: List[str]) -> torch.FloatTensor:
    return self.encode_labels(prompts)

  def align_spatial_features_with_language(self, features: torch.FloatTensor) -> torch.FloatTensor:
    return features


class MissingOpenClipEncoder(LangSpatialImageEncoder):
  """Raises ModuleNotFoundError(open_clip) to test package name mapping."""

  def __init__(self, device: str = "cpu"):
    super().__init__(device=device)
    exc = ModuleNotFoundError("No module named 'open_clip'")
    exc.name = "open_clip"
    raise exc

  def is_compatible_size(self, h: int, w: int) -> bool:
    return True

  def get_nearest_size(self, h, w):
    return h, w

  def encode_image_to_feat_map(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
    return rgb_image

  def encode_labels(self, labels: List[str]) -> torch.FloatTensor:
    return torch.zeros((len(labels), 1), dtype=torch.float32)

  def encode_prompts(self, prompts: List[str]) -> torch.FloatTensor:
    return self.encode_labels(prompts)

  def align_spatial_features_with_language(self, features: torch.FloatTensor) -> torch.FloatTensor:
    return features
