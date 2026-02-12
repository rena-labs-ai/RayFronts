"""Stub module to emulate RADIO encoder behavior in tests."""

from __future__ import annotations

from typing import List

import torch

from rayfronts.image_encoders.base import LangSpatialImageEncoder


class RadioEncoder(LangSpatialImageEncoder):
  """Requires use_summ_proj_for_spatial=True for language querying."""

  def __init__(self,
               device: str = "cpu",
               lang_model: str = None,
               return_radio_features: bool = True,
               use_summ_proj_for_spatial: bool = False):
    super().__init__(device=device)
    if (lang_model is not None and
        return_radio_features and
        not use_summ_proj_for_spatial):
      raise ValueError("use_summ_proj_for_spatial must be true")

  def is_compatible_size(self, h: int, w: int) -> bool:
    return True

  def get_nearest_size(self, h, w):
    return h, w

  def encode_image_to_feat_map(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
    feat = rgb_image[:, :3, :, :] + 1e-3
    return torch.nn.functional.normalize(feat, dim=1)

  def encode_labels(self, labels: List[str]) -> torch.FloatTensor:
    return torch.nn.functional.normalize(
      torch.ones((len(labels), 3), dtype=torch.float32), dim=1)

  def encode_prompts(self, prompts: List[str]) -> torch.FloatTensor:
    return self.encode_labels(prompts)

  def align_spatial_features_with_language(self, features: torch.FloatTensor) -> torch.FloatTensor:
    return features
