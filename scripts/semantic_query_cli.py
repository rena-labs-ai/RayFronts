#!/usr/bin/env python3
"""Semantic query CLI for RayFronts maps built from MCAP or ROS streams."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from rayfronts.datasets.mcap import McapRos2Dataset
from rayfronts.datasets.ros import Ros2Subscriber
from rayfronts.semantic_query_engine import (
  import_symbol,
  map_rgbd_stream,
  query_object_position,
)

logger = logging.getLogger("semantic_query_cli")

PYTHON_MODULE_TO_PACKAGE = {
  "open_clip": "open_clip_torch",
}


class MissingOptionalDependencyError(RuntimeError):
  """Raised when an optional dependency is required but not installed."""


def _resolve_device(requested: Optional[str], context: str) -> Optional[str]:
  if requested is None:
    return requested
  if isinstance(requested, str) and requested.startswith("cuda"):
    if not torch.cuda.is_available():
      logger.warning("CUDA requested for %s but no CUDA device is available. "
                     "Falling back to CPU.", context)
      return "cpu"
  return requested


def _parse_json_dict(s: Optional[str], arg_name: str) -> Dict[str, Any]:
  if s is None:
    return dict()
  try:
    d = json.loads(s)
  except json.JSONDecodeError as exc:
    raise ValueError(f"{arg_name} must be valid JSON.") from exc
  if not isinstance(d, dict):
    raise ValueError(f"{arg_name} must decode to an object/dict.")
  return d


def _instantiate(cls_path: str, kwargs: Dict[str, Any]):
  cls = import_symbol(cls_path)
  try:
    return cls(**kwargs)
  except ModuleNotFoundError as exc:
    missing = exc.name if getattr(exc, "name", None) else "unknown"
    package = PYTHON_MODULE_TO_PACKAGE.get(missing, missing)
    raise MissingOptionalDependencyError(
      f"Failed to initialize '{cls_path}' because optional dependency "
      f"'{missing}' is not installed. Install it with: `python3 -m pip install {package}`"
    ) from exc


def _metadata_path_for_map(map_path: Path) -> Path:
  return Path(str(map_path) + ".meta.json")


def _load_map_metadata(map_path: Path, explicit_path: Optional[str]) -> Dict[str, Any]:
  meta_path = Path(explicit_path) if explicit_path is not None else _metadata_path_for_map(map_path)
  if not meta_path.exists():
    return dict()
  try:
    return json.loads(meta_path.read_text(encoding="utf-8"))
  except json.JSONDecodeError as exc:
    raise ValueError(f"Invalid JSON metadata file: {meta_path}") from exc


def _apply_metadata_defaults(args, metadata: Dict[str, Any]) -> None:
  if args.encoder_class is None and metadata.get("encoder_class") is not None:
    args.encoder_class = metadata["encoder_class"]

  mapper_default = "rayfronts.mapping.semantic_voxel_map.SemanticVoxelMap"
  if args.mapper_class == mapper_default and metadata.get("mapper_class") is not None:
    args.mapper_class = metadata["mapper_class"]

  if args.encoder_kwargs == "{}" and isinstance(metadata.get("encoder_kwargs"), dict):
    args.encoder_kwargs = json.dumps(metadata["encoder_kwargs"])
  if args.mapper_kwargs == "{}" and isinstance(metadata.get("mapper_kwargs"), dict):
    args.mapper_kwargs = json.dumps(metadata["mapper_kwargs"])


def _parse_intrinsics_from_metadata(metadata: Dict[str, Any]) -> torch.FloatTensor:
  intrinsics = metadata.get("intrinsics_3x3")
  if intrinsics is None:
    return torch.eye(3, dtype=torch.float32)
  intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
  if intrinsics.shape != (3, 3):
    raise ValueError("Map metadata intrinsics_3x3 must have shape 3x3.")
  return intrinsics


def _collect_queries(args) -> List[str]:
  raw = list()
  if args.query is not None:
    raw.append(args.query)
  if args.objects is not None:
    raw.extend(args.objects)

  queries = list()
  seen = set()
  for item in raw:
    for token in item.split(","):
      q = token.strip()
      if len(q) == 0:
        continue
      if q in seen:
        continue
      seen.add(q)
      queries.append(q)
  return queries


def _slugify(s: str) -> str:
  slug = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip().lower())
  slug = slug.strip("_")
  return slug if len(slug) > 0 else "query"


def _single_query_output(estimate_dict: Dict[str, Any],
                         args,
                         processed_frames: int,
                         source_path: Optional[str],
                         run_config: Dict[str, Any]) -> Dict[str, Any]:
  out = dict(estimate_dict)
  out["processed_frames"] = processed_frames
  if source_path is not None:
    out["source_path"] = source_path
  out["run_config"] = run_config
  return out


def _build_run_config(args, dataset=None, loaded_map: Optional[str] = None) -> Dict[str, Any]:
  return dict(
    source=args.source,
    query_type=args.query_type,
    top_k=args.top_k,
    cluster_radius=args.cluster_radius,
    max_frames=args.max_frames,
    frame_skip=args.frame_skip,
    depth_limit=args.depth_limit,
    target_frame=args.target_frame,
    pose_frame=args.pose_frame,
    resolved_target_frame=getattr(dataset, "_resolved_target_frame", None),
    resolved_pose_frame=getattr(dataset, "_resolved_pose_frame", None),
    mapper_class=args.mapper_class,
    mapper_kwargs=_parse_json_dict(args.mapper_kwargs, "--mapper-kwargs"),
    encoder_class=args.encoder_class,
    encoder_kwargs=_parse_json_dict(args.encoder_kwargs, "--encoder-kwargs"),
    loaded_map=loaded_map,
  )


def _build_dataset(args) -> Any:
  if args.source == "mcap":
    return McapRos2Dataset(
      path=args.mcap_path,
      rgb_topic=args.rgb_topic or "/base/camera/color/image_raw/compressed",
      depth_topic=args.depth_topic or "/base/camera/aligned_depth_to_color/image_raw/compressedDepth",
      camera_info_topic=args.camera_info_topic or "/base/camera/color/camera_info",
      tf_topic=args.tf_topic,
      tf_static_topic=args.tf_static_topic,
      target_frame=args.target_frame,
      pose_frame=args.pose_frame,
      src_coord_system=args.src_coord_system,
      depth_scale=args.depth_scale,
      sync_slop_s=args.sync_slop,
      pose_slop_s=args.pose_slop,
      pose_fallback=args.pose_fallback,
      rgb_resolution=args.rgb_resolution,
      depth_resolution=args.depth_resolution,
      frame_skip=args.frame_skip,
    )

  if args.source == "live":
    return Ros2Subscriber(
      rgb_topic=args.rgb_topic,
      pose_topic=args.pose_topic,
      rgb_resolution=args.rgb_resolution,
      depth_resolution=args.depth_resolution,
      disparity_topic=args.disparity_topic,
      depth_topic=args.depth_topic,
      confidence_topic=args.confidence_topic,
      point_cloud_topic=args.point_cloud_topic,
      intrinsics_topic=args.intrinsics_topic,
      intrinsics_file=args.intrinsics_file,
      src_coord_system=args.src_coord_system,
      frame_skip=args.frame_skip,
    )

  dataset_kwargs = _parse_json_dict(args.dataset_kwargs, "--dataset-kwargs")
  return _instantiate(args.dataset_class, dataset_kwargs)


def _build_mapper(args, dataset=None, intrinsics_3x3: Optional[torch.FloatTensor] = None) -> Any:
  mapper_kwargs = _parse_json_dict(args.mapper_kwargs, "--mapper-kwargs")
  encoder = None
  if args.encoder_class is not None:
    encoder_kwargs = _parse_json_dict(args.encoder_kwargs, "--encoder-kwargs")
    if "device" in encoder_kwargs:
      encoder_kwargs["device"] = _resolve_device(
        encoder_kwargs["device"], f"encoder '{args.encoder_class}'")
    elif args.device is not None:
      encoder_kwargs["device"] = _resolve_device(
        args.device, f"encoder '{args.encoder_class}'")

    # RADIO default may leave spatial features in a space not aligned with text.
    if args.encoder_class.endswith("radio.RadioEncoder"):
      if (encoder_kwargs.get("lang_model") is not None and
          encoder_kwargs.get("return_radio_features", True) and
          "use_summ_proj_for_spatial" not in encoder_kwargs):
        encoder_kwargs["use_summ_proj_for_spatial"] = True
        logger.info("Auto-setting RadioEncoder.use_summ_proj_for_spatial=True "
                    "for language-aligned querying.")

    encoder = _instantiate(args.encoder_class, encoder_kwargs)

  mapper_cls = import_symbol(args.mapper_class)
  sig = inspect.signature(mapper_cls.__init__).parameters
  if "intrinsics_3x3" in sig and "intrinsics_3x3" not in mapper_kwargs:
    if intrinsics_3x3 is not None:
      mapper_kwargs["intrinsics_3x3"] = intrinsics_3x3
    elif dataset is not None and hasattr(dataset, "intrinsics_3x3"):
      mapper_kwargs["intrinsics_3x3"] = dataset.intrinsics_3x3
    else:
      mapper_kwargs["intrinsics_3x3"] = torch.eye(3, dtype=torch.float32)
  if "device" in sig and "device" not in mapper_kwargs:
    mapper_kwargs["device"] = _resolve_device(args.device, "mapper")
  elif "device" in mapper_kwargs:
    mapper_kwargs["device"] = _resolve_device(mapper_kwargs["device"], "mapper")
  if "visualizer" in sig and "visualizer" not in mapper_kwargs:
    mapper_kwargs["visualizer"] = None
  if encoder is not None and "encoder" in sig and "encoder" not in mapper_kwargs:
    mapper_kwargs["encoder"] = encoder
  return mapper_cls(**mapper_kwargs)


def _parse_resolution(s: Optional[str]):
  if s is None:
    return None
  if "x" in s:
    h, w = s.lower().split("x", maxsplit=1)
    return int(h), int(w)
  v = int(s)
  return v, v


def build_parser():
  parser = argparse.ArgumentParser("Semantic Query CLI")
  parser.add_argument("--source", choices=["mcap", "live", "dataset", "saved_map"],
                      required=True, help="Input source type.")
  parser.add_argument("--object", dest="query", required=False,
                      help="Object label/prompt to query.")
  parser.add_argument("--objects", nargs="+", default=None,
                      help="Optional additional objects to query in the same mapped scene.")
  parser.add_argument("--top-k", type=int, default=5,
                      help="Number of salient voxels for estimation.")
  parser.add_argument("--cluster-radius", type=float, default=1.0,
                      help="Radius (m) used to form salient voxel clusters for final estimate.")
  parser.add_argument("--query-type", choices=["labels", "prompts"],
                      default="labels")
  parser.add_argument("--softmax", action="store_true",
                      help="Use softmax across queries before selection.")
  parser.add_argument("--compressed-query", action="store_true",
                      help="Query map in compressed feature space.")
  parser.add_argument("--max-frames", type=int, default=-1,
                      help="Frames to map before querying (-1: map full source).")
  parser.add_argument("--depth-limit", type=float, default=-1,
                      help="Clamp depth beyond this value to inf. -1 disables.")
  parser.add_argument("--device", type=str, default="cpu")
  parser.add_argument("--frame-skip", type=int, default=0)
  parser.add_argument("--rgb-resolution", type=str, default=None,
                      help="Resolution HxW or integer for RGB output.")
  parser.add_argument("--depth-resolution", type=str, default=None,
                      help="Resolution HxW or integer for depth output.")

  parser.add_argument("--mapper-class", type=str,
                      default="rayfronts.mapping.semantic_voxel_map.SemanticVoxelMap")
  parser.add_argument("--mapper-kwargs", type=str, default="{}")
  parser.add_argument("--encoder-class", type=str, default=None)
  parser.add_argument("--encoder-kwargs", type=str, default="{}")

  parser.add_argument("--output-json", type=str, default=None,
                      help="Optional path to write JSON output.")
  parser.add_argument("--per-query-output-dir", type=str, default=None,
                      help="Optional directory to write one JSON file per queried object.")
  parser.add_argument("--save-map", type=str, default=None,
                      help="Optional path to save mapped voxels for reuse.")
  parser.add_argument("--load-map", type=str, default=None,
                      help="Load an existing map and skip mapping stage.")
  parser.add_argument("--map-metadata", type=str, default=None,
                      help="Optional metadata JSON path for a loaded map.")

  # MCAP options
  parser.add_argument("--mcap-path", type=str, default=None)
  parser.add_argument("--rgb-topic", type=str, default=None)
  parser.add_argument("--depth-topic", type=str, default=None)
  parser.add_argument("--camera-info-topic", type=str, default=None)
  parser.add_argument("--tf-topic", type=str, default="/tf")
  parser.add_argument("--tf-static-topic", type=str, default="/tf_static")
  parser.add_argument("--target-frame", type=str, default="auto")
  parser.add_argument("--pose-frame", type=str, default="auto")
  parser.add_argument("--pose-slop", type=float, default=0.2)
  parser.add_argument("--pose-fallback", type=str,
                      choices=["drop", "last", "identity"], default="drop")
  parser.add_argument("--sync-slop", type=float, default=0.05)
  parser.add_argument("--depth-scale", type=float, default=0.001)
  parser.add_argument("--src-coord-system", type=str, default="flu")

  # Live ROS options
  parser.add_argument("--pose-topic", type=str, default=None)
  parser.add_argument("--disparity-topic", type=str, default=None)
  parser.add_argument("--point-cloud-topic", type=str, default=None)
  parser.add_argument("--confidence-topic", type=str, default=None)
  parser.add_argument("--intrinsics-topic", type=str, default=None)
  parser.add_argument("--intrinsics-file", type=str, default=None)

  # Generic dataset class options
  parser.add_argument("--dataset-class", type=str, default=None)
  parser.add_argument("--dataset-kwargs", type=str, default="{}")

  return parser


def validate_args(args):
  if args.cluster_radius <= 0:
    raise ValueError("--cluster-radius must be > 0.")
  queries = _collect_queries(args)
  if len(queries) == 0:
    raise ValueError("At least one query is required via --object or --objects.")
  if args.encoder_class is None:
    raise ValueError("--encoder-class is required for semantic text querying.")

  if args.source == "saved_map":
    if args.load_map is None:
      raise ValueError("--load-map is required when --source saved_map.")
    if args.save_map is not None:
      raise ValueError("--save-map cannot be used with --source saved_map.")
    return

  if args.load_map is not None:
    raise ValueError("--load-map is only supported when --source saved_map.")

  if args.source == "mcap" and not args.mcap_path:
    raise ValueError("--mcap-path is required when --source mcap.")
  if args.source == "live" and args.rgb_topic is None:
    raise ValueError("--rgb-topic is required when --source live.")
  if args.source == "live":
    if args.pose_topic is None:
      raise ValueError("--pose-topic is required when --source live.")
    has_depth_source = (
      args.depth_topic is not None or
      args.disparity_topic is not None or
      args.point_cloud_topic is not None
    )
    if not has_depth_source:
      raise ValueError(
        "One depth source is required for live mode: --depth-topic, "
        "--disparity-topic, or --point-cloud-topic.")
    if args.intrinsics_topic is None and args.intrinsics_file is None:
      raise ValueError(
        "Live mode requires either --intrinsics-topic or --intrinsics-file.")
  if args.source == "dataset" and args.dataset_class is None:
    raise ValueError("--dataset-class is required when --source dataset.")


def main(argv=None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  args.rgb_resolution = _parse_resolution(args.rgb_resolution)
  args.depth_resolution = _parse_resolution(args.depth_resolution)
  metadata = dict()

  if args.source == "saved_map":
    map_path = Path(args.load_map) if args.load_map is not None else None
    if map_path is not None and map_path.exists():
      metadata = _load_map_metadata(map_path, args.map_metadata)
      _apply_metadata_defaults(args, metadata)
  validate_args(args)

  try:
    dataset = None
    source_path = None
    if args.source == "saved_map":
      map_path = Path(args.load_map)
      if not map_path.exists():
        raise FileNotFoundError(f"Map file not found: {map_path}")
      intrinsics_3x3 = _parse_intrinsics_from_metadata(metadata)
      dataset = SimpleNamespace(intrinsics_3x3=intrinsics_3x3)
      mapper = _build_mapper(args, dataset=dataset, intrinsics_3x3=intrinsics_3x3)
      mapper.load(str(map_path))
      processed = int(metadata.get("processed_frames", 0))
      last_pose = None
      source_path = metadata.get("source_path")
    else:
      dataset = _build_dataset(args)
      mapper = _build_mapper(args, dataset)
      max_frames = None if args.max_frames is None or args.max_frames < 0 else args.max_frames
      processed, last_pose = map_rgbd_stream(
        mapper=mapper,
        dataset=dataset,
        max_frames=max_frames,
        depth_limit=args.depth_limit,
      )
      if processed <= 0:
        raise RuntimeError("No frames were mapped. Cannot run query.")
      if args.source == "mcap":
        source_path = str(Path(args.mcap_path))
      if args.save_map is not None:
        save_map_path = Path(args.save_map)
        save_map_path.parent.mkdir(parents=True, exist_ok=True)
        mapper.save(str(save_map_path))
        metadata_out = dict(
          mapper_class=args.mapper_class,
          mapper_kwargs=_parse_json_dict(args.mapper_kwargs, "--mapper-kwargs"),
          encoder_class=args.encoder_class,
          encoder_kwargs=_parse_json_dict(args.encoder_kwargs, "--encoder-kwargs"),
          source=args.source,
          source_path=source_path,
          processed_frames=processed,
          intrinsics_3x3=dataset.intrinsics_3x3.detach().cpu().tolist()
            if hasattr(dataset, "intrinsics_3x3") else None,
        )
        metadata_path = _metadata_path_for_map(save_map_path)
        metadata_path.write_text(json.dumps(metadata_out, indent=2), encoding="utf-8")
        logger.info("Saved map to %s", save_map_path)
        logger.info("Saved map metadata to %s", metadata_path)
  except MissingOptionalDependencyError as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    return 2

  queries = _collect_queries(args)
  estimates = list()
  for query in queries:
    estimate = query_object_position(
      mapper=mapper,
      query=query,
      top_k=args.top_k,
      cluster_radius_m=args.cluster_radius,
      query_type=args.query_type,
      softmax=args.softmax,
      compressed=args.compressed_query,
      reference_pose_4x4=last_pose,
    )
    estimates.append(estimate.as_dict())

  run_config = _build_run_config(
    args=args,
    dataset=dataset,
    loaded_map=args.load_map,
  )

  if len(estimates) == 1:
    output = _single_query_output(
      estimate_dict=estimates[0],
      args=args,
      processed_frames=processed,
      source_path=source_path,
      run_config=run_config,
    )
  else:
    output = dict(
      queries=estimates,
      processed_frames=processed,
      run_config=run_config,
    )
    if source_path is not None:
      output["source_path"] = source_path

  per_query_paths = list()
  if args.per_query_output_dir is not None:
    per_query_dir = Path(args.per_query_output_dir)
    per_query_dir.mkdir(parents=True, exist_ok=True)
    for estimate_dict in estimates:
      per_query = _single_query_output(
        estimate_dict=estimate_dict,
        args=args,
        processed_frames=processed,
        source_path=source_path,
        run_config=run_config,
      )
      file_name = f"{_slugify(estimate_dict['query'])}.json"
      out_path = per_query_dir / file_name
      out_path.write_text(json.dumps(per_query, indent=2), encoding="utf-8")
      per_query_paths.append(str(out_path))
    if len(estimates) > 1:
      output["per_query_json"] = per_query_paths

  print(json.dumps(output, indent=2))

  if args.output_json is not None:
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

  return 0


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  raise SystemExit(main())
