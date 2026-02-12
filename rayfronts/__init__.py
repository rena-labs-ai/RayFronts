"""RayFronts package root.

Keep imports lazy to avoid importing optional heavy dependencies eagerly.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
  "datasets",
  "feat_compressors",
  "geometry3d",
  "image_encoders",
  "mapping",
  "messaging_services",
  "ros_utils",
  "utils",
  "visualizers",
]


def __getattr__(name: str) -> Any:
  if name in __all__:
    return importlib.import_module(f"rayfronts.{name}")
  raise AttributeError(f"module 'rayfronts' has no attribute '{name}'")


def __dir__():
  return sorted(__all__)
