"""ROS2 MCAP dataset reader for posed RGBD streams."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import struct

import numpy as np
import torch
from scipy.spatial.transform import Rotation

try:
  import cv2
except ModuleNotFoundError:
  cv2 = None

try:
  from rosbags.highlevel import AnyReader
except ModuleNotFoundError:
  AnyReader = None

from rayfronts.datasets.base import PosedRgbdDataset
from rayfronts import geometry3d as g3d

logger = logging.getLogger(__name__)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _stamp_to_ns(stamp) -> int:
  return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _normalize_frame_id(frame_id: str) -> str:
  return frame_id.lstrip("/")


def _find_png_start(data: bytes) -> int:
  idx = data.find(PNG_MAGIC)
  if idx != -1:
    return idx
  if len(data) > 12:
    return 12
  return 0


def decode_compressed_rgb(msg) -> np.ndarray:
  """Decode a sensor_msgs/msg/CompressedImage (JPEG/PNG) to BGR uint8."""
  if cv2 is None:
    raise ModuleNotFoundError("cv2 is required to decode compressed images.")
  arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
  bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
  if bgr is None:
    raise ValueError("Failed to decode compressed RGB image.")
  return bgr


def decode_compressed_depth_to_meters(msg, depth_scale: float = 0.001) -> np.ndarray:
  """Decode compressedDepth into a metric depth image in meters."""
  if cv2 is None:
    raise ModuleNotFoundError("cv2 is required to decode compressed depth.")

  encoding = msg.format.split(";", 1)[0].strip()
  data = bytes(msg.data)
  png_start = _find_png_start(data)

  header = data[:png_start]
  inv_depth_params = None
  if len(header) >= 12:
    try:
      inv_depth_params = struct.unpack("<iff", header[:12])
    except struct.error:
      inv_depth_params = None

  arr = cv2.imdecode(np.frombuffer(data[png_start:], dtype=np.uint8),
                     cv2.IMREAD_UNCHANGED)
  if arr is None:
    raise ValueError("Failed to decode compressed depth image.")
  if arr.ndim == 3:
    arr = arr[..., 0]

  if encoding == "16UC1":
    depth_m = arr.astype(np.float32) * float(depth_scale)
  elif encoding == "32FC1":
    if inv_depth_params is None:
      depth_m = arr.astype(np.float32)
    else:
      _, depth_param_a, depth_param_b = inv_depth_params
      q = arr.astype(np.float32)
      with np.errstate(divide="ignore", invalid="ignore"):
        depth_m = depth_param_a / (q - depth_param_b)
  else:
    depth_m = arr.astype(np.float32)

  depth_m[depth_m == 0] = np.nan
  return depth_m


def tf_to_matrix(tf_msg) -> np.ndarray:
  """Converts a geometry_msgs/Transform to a 4x4 matrix."""
  t = tf_msg.transform.translation
  q = tf_msg.transform.rotation
  rot = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
  mat = np.eye(4, dtype=np.float32)
  mat[:3, :3] = rot
  mat[0, 3] = t.x
  mat[1, 3] = t.y
  mat[2, 3] = t.z
  return mat


class McapRos2Dataset(PosedRgbdDataset):
  """Loads posed RGBD frames from a ROS2 MCAP recording."""

  def __init__(self,
               path: str,
               rgb_topic: str = "/base/camera/color/image_raw/compressed",
               depth_topic: str = "/base/camera/aligned_depth_to_color/image_raw/compressedDepth",
               camera_info_topic: str = "/base/camera/color/camera_info",
               tf_topic: str = "/tf",
               tf_static_topic: str = "/tf_static",
               target_frame: str = "auto",
               pose_frame: str = "auto",
               src_coord_system: str = "flu",
               depth_scale: float = 0.001,
               sync_slop_s: float = 0.05,
               pose_slop_s: float = 0.2,
               pose_fallback: str = "drop",
               rgb_resolution: Union[Tuple[int, int], int] = None,
               depth_resolution: Union[Tuple[int, int], int] = None,
               frame_skip: int = 0,
               interp_mode: str = "bilinear"):
    super().__init__(rgb_resolution=rgb_resolution,
                     depth_resolution=depth_resolution,
                     frame_skip=frame_skip,
                     interp_mode=interp_mode)
    if AnyReader is None:
      raise ModuleNotFoundError("rosbags is required for MCAP ingestion.")

    self.path = path
    self.rgb_topic = rgb_topic
    self.depth_topic = depth_topic
    self.camera_info_topic = camera_info_topic
    self.tf_topic = tf_topic
    self.tf_static_topic = tf_static_topic
    self.target_frame = _normalize_frame_id(target_frame)
    self.pose_frame = _normalize_frame_id(pose_frame)
    self.depth_scale = depth_scale
    self.sync_slop_ns = int(sync_slop_s * 1e9)
    self.pose_slop_ns = int(pose_slop_s * 1e9)
    if pose_fallback not in ["drop", "last", "identity"]:
      raise ValueError("pose_fallback must be one of: drop, last, identity")
    self.pose_fallback = pose_fallback

    self._src2rdf_transform = g3d.mat_3x3_to_4x4(
      g3d.get_coord_system_transform(src_coord_system, "rdf"))
    self._camera_frame = None
    self._tf_dynamic_samples: Dict[Tuple[str, str], List[Tuple[int, torch.FloatTensor]]] = dict()
    self._tf_dynamic_ns: Dict[Tuple[str, str], List[int]] = dict()
    self._tf_dynamic_4x4: Dict[Tuple[str, str], List[torch.FloatTensor]] = dict()
    self._tf_static_4x4: Dict[Tuple[str, str], torch.FloatTensor] = dict()
    self._tf_neighbors: Dict[str, List[str]] = dict()
    self._pose_path_frames: List[str] = list()
    self._resolved_pose_frame: Optional[str] = None
    self._last_pose_4x4: Optional[torch.FloatTensor] = None
    self._resolved_target_frame: Optional[str] = None

    self._init_metadata()

  def _open_reader(self):
    return AnyReader([Path(self.path)])

  def _init_metadata(self):
    with self._open_reader() as reader:
      topic_to_conn = {c.topic: c for c in reader.connections}
      if self.rgb_topic not in topic_to_conn:
        raise ValueError(f"RGB topic '{self.rgb_topic}' not found in MCAP.")
      if self.depth_topic not in topic_to_conn:
        raise ValueError(f"Depth topic '{self.depth_topic}' not found in MCAP.")
      if self.camera_info_topic not in topic_to_conn:
        raise ValueError(
          f"Camera info topic '{self.camera_info_topic}' not found in MCAP.")

      cam_conn = topic_to_conn[self.camera_info_topic]
      cam_info_msg = None
      for conn, _, raw in reader.messages(connections=[cam_conn]):
        cam_info_msg = reader.deserialize(raw, conn.msgtype)
        break

      if cam_info_msg is None:
        raise ValueError("No camera info message found in MCAP.")

      self.intrinsics_3x3 = torch.tensor(cam_info_msg.k, dtype=torch.float).reshape(3, 3)
      self._camera_frame = _normalize_frame_id(cam_info_msg.header.frame_id)
      self.original_h = int(cam_info_msg.height)
      self.original_w = int(cam_info_msg.width)
      self.rgb_h = self.original_h if self.rgb_h <= 0 else self.rgb_h
      self.rgb_w = self.original_w if self.rgb_w <= 0 else self.rgb_w
      self.depth_h = self.original_h if self.depth_h <= 0 else self.depth_h
      self.depth_w = self.original_w if self.depth_w <= 0 else self.depth_w

      if self.depth_h != self.original_h or self.depth_w != self.original_w:
        h_ratio = self.depth_h / self.original_h
        w_ratio = self.depth_w / self.original_w
        self.intrinsics_3x3[0, :] = self.intrinsics_3x3[0, :] * w_ratio
        self.intrinsics_3x3[1, :] = self.intrinsics_3x3[1, :] * h_ratio

      tf_conns = list()
      if self.tf_topic in topic_to_conn:
        tf_conns.append(topic_to_conn[self.tf_topic])
      if self.tf_static_topic in topic_to_conn:
        tf_conns.append(topic_to_conn[self.tf_static_topic])

      for conn, _, raw in reader.messages(connections=tf_conns):
        is_static_conn = conn.topic == self.tf_static_topic
        msg = reader.deserialize(raw, conn.msgtype)
        for transform in msg.transforms:
          parent = _normalize_frame_id(transform.header.frame_id)
          child = _normalize_frame_id(transform.child_frame_id)
          ts_ns = _stamp_to_ns(transform.header.stamp)
          mat = torch.tensor(tf_to_matrix(transform), dtype=torch.float32)
          edge = (parent, child)
          if is_static_conn:
            self._tf_static_4x4[edge] = mat
          else:
            self._tf_dynamic_samples.setdefault(edge, list()).append((ts_ns, mat))

      for edge, samples in self._tf_dynamic_samples.items():
        samples = sorted(samples, key=lambda x: x[0])
        self._tf_dynamic_ns[edge] = [x[0] for x in samples]
        self._tf_dynamic_4x4[edge] = [x[1] for x in samples]

      neighbors = defaultdict(set)
      for parent, child in set(self._tf_dynamic_ns.keys()) | set(self._tf_static_4x4.keys()):
        neighbors[parent].add(child)
        neighbors[child].add(parent)
      self._tf_neighbors = {k: sorted(v) for k, v in neighbors.items()}

      (self._resolved_target_frame,
       self._resolved_pose_frame,
       self._pose_path_frames) = self._resolve_frames()

      dynamic_edge_samples = 0
      for i in range(len(self._pose_path_frames) - 1):
        a = self._pose_path_frames[i]
        b = self._pose_path_frames[i + 1]
        if (a, b) in self._tf_dynamic_ns:
          dynamic_edge_samples += len(self._tf_dynamic_ns[(a, b)])
        elif (b, a) in self._tf_dynamic_ns:
          dynamic_edge_samples += len(self._tf_dynamic_ns[(b, a)])

      logger.info("McapRos2Dataset initialized. pose frame: %s->%s, path=%s, dynamic_samples=%d",
                  self._resolved_target_frame, self._resolved_pose_frame,
                  " -> ".join(self._pose_path_frames), dynamic_edge_samples)

  def _ordered_unique(self, values: List[str]) -> List[str]:
    out = list()
    seen = set()
    for value in values:
      if value is None:
        continue
      value = _normalize_frame_id(value)
      if len(value) == 0:
        continue
      if value in seen:
        continue
      seen.add(value)
      out.append(value)
    return out

  def _find_tf_path(self, src: str, dst: str) -> Optional[List[str]]:
    src = _normalize_frame_id(src)
    dst = _normalize_frame_id(dst)
    if src == dst:
      return [src]
    if src not in self._tf_neighbors or dst not in self._tf_neighbors:
      return None

    q = deque([src])
    prev = {src: None}
    while len(q) > 0:
      node = q.popleft()
      if node == dst:
        break
      for nxt in self._tf_neighbors.get(node, []):
        if nxt in prev:
          continue
        prev[nxt] = node
        q.append(nxt)

    if dst not in prev:
      return None

    path = list()
    node = dst
    while node is not None:
      path.append(node)
      node = prev[node]
    path.reverse()
    return path

  def _pose_candidates(self) -> List[str]:
    if self.pose_frame != "auto":
      return [self.pose_frame]

    child_counts = defaultdict(int)
    for (_, child), ts in self._tf_dynamic_ns.items():
      child_counts[child] += len(ts)
    dynamic_children = [k for k, _ in sorted(child_counts.items(),
                                              key=lambda kv: kv[1],
                                              reverse=True)]

    return self._ordered_unique([
      self._camera_frame,
      "camera_color_optical_frame",
      "camera_depth_optical_frame",
      "camera_link",
      "body",
      "base_body_link",
      "base_camera_link",
      "base_link",
      "base_footprint",
      *dynamic_children,
      *self._tf_neighbors.keys(),
    ])

  def _target_candidates(self) -> List[str]:
    if self.target_frame != "auto":
      return [self.target_frame]

    parent_counts = defaultdict(int)
    for (parent, _), ts in self._tf_dynamic_ns.items():
      parent_counts[parent] += len(ts)
    dynamic_parents = [k for k, _ in sorted(parent_counts.items(),
                                            key=lambda kv: kv[1],
                                            reverse=True)]

    return self._ordered_unique([
      "map",
      "odom",
      "world",
      "camera_init",
      *dynamic_parents,
      *self._tf_neighbors.keys(),
    ])

  def _path_dynamic_stats(self, path: List[str]) -> Tuple[int, int]:
    edge_count = 0
    sample_count = 0
    for i in range(len(path) - 1):
      a = path[i]
      b = path[i + 1]
      if (a, b) in self._tf_dynamic_ns:
        edge_count += 1
        sample_count += len(self._tf_dynamic_ns[(a, b)])
      elif (b, a) in self._tf_dynamic_ns:
        edge_count += 1
        sample_count += len(self._tf_dynamic_ns[(b, a)])
    return edge_count, sample_count

  def _pick_best_frame_combo(self, pose_candidates: List[str],
                             target_candidates: List[str]) -> Optional[Tuple[str, str, List[str]]]:
    best = None
    best_score = None
    for p_idx, pose_frame in enumerate(pose_candidates):
      for t_idx, target_frame in enumerate(target_candidates):
        path = self._find_tf_path(target_frame, pose_frame)
        if path is None:
          continue
        dyn_edges, dyn_samples = self._path_dynamic_stats(path)
        score = (
          1 if dyn_edges > 0 else 0,
          -p_idx,
          -t_idx,
          dyn_samples,
          -len(path),
        )
        if best_score is None or score > best_score:
          best_score = score
          best = (target_frame, pose_frame, path)
    return best

  def _resolve_frames(self) -> Tuple[str, str, List[str]]:
    pose_candidates = self._pose_candidates()
    target_candidates = self._target_candidates()

    if self.pose_frame != "auto" and self.target_frame != "auto":
      path = self._find_tf_path(self.target_frame, self.pose_frame)
      if path is None:
        raise ValueError(f"No TF path found from '{self.target_frame}' to '{self.pose_frame}'.")
      return self.target_frame, self.pose_frame, path

    if self.pose_frame == "auto" and self.target_frame != "auto":
      best = self._pick_best_frame_combo(
        pose_candidates=pose_candidates,
        target_candidates=[self.target_frame],
      )
      if best is not None:
        return best
      raise ValueError(f"No TF path found from '{self.target_frame}' to any pose frame candidate.")

    if self.pose_frame != "auto" and self.target_frame == "auto":
      best = self._pick_best_frame_combo(
        pose_candidates=[self.pose_frame],
        target_candidates=target_candidates,
      )
      if best is not None:
        return best
      raise ValueError(f"No TF path found to pose frame '{self.pose_frame}'.")

    best = self._pick_best_frame_combo(
      pose_candidates=pose_candidates,
      target_candidates=target_candidates,
    )
    if best is not None:
      return best

    raise ValueError("Could not auto-resolve pose/target frames from TF graph.")

  def _lookup_direct_tf(self, parent: str, child: str,
                        ts_ns: int) -> Optional[torch.FloatTensor]:
    edge = (_normalize_frame_id(parent), _normalize_frame_id(child))

    ns = self._tf_dynamic_ns.get(edge, None)
    mats = self._tf_dynamic_4x4.get(edge, None)
    if ns is not None and mats is not None and len(ns) > 0:
      idx = bisect_left(ns, ts_ns)
      candidates = list()
      if idx < len(ns):
        candidates.append(idx)
      if idx > 0:
        candidates.append(idx - 1)

      if len(candidates) > 0:
        best_idx = min(candidates, key=lambda i: abs(ns[i] - ts_ns))
        if abs(ns[best_idx] - ts_ns) <= self.pose_slop_ns:
          return mats[best_idx]

    if edge in self._tf_static_4x4:
      return self._tf_static_4x4[edge]
    return None

  def _lookup_relative_tf(self, src: str, dst: str,
                          ts_ns: int) -> Optional[torch.FloatTensor]:
    mat = self._lookup_direct_tf(src, dst, ts_ns)
    if mat is not None:
      return mat
    inv = self._lookup_direct_tf(dst, src, ts_ns)
    if inv is not None:
      return torch.linalg.inv(inv)
    return None

  def _lookup_pose(self, ts_ns: int) -> Optional[torch.FloatTensor]:
    if len(self._pose_path_frames) == 0:
      return None

    src_pose_4x4 = torch.eye(4, dtype=torch.float32)
    for i in range(len(self._pose_path_frames) - 1):
      a = self._pose_path_frames[i]
      b = self._pose_path_frames[i + 1]
      rel = self._lookup_relative_tf(a, b, ts_ns)
      if rel is None:
        return None
      src_pose_4x4 = src_pose_4x4 @ rel

    return g3d.transform_pose_4x4(src_pose_4x4, self._src2rdf_transform)

  def _format_frame(self, rgb_bgr: np.ndarray, depth_m: np.ndarray,
                    ts_ns: int) -> Optional[Dict[str, torch.Tensor]]:
    pose_4x4 = self._lookup_pose(ts_ns)
    if pose_4x4 is None:
      if self.pose_fallback == "drop":
        return None
      if self.pose_fallback == "last":
        pose_4x4 = self._last_pose_4x4
        if pose_4x4 is None:
          return None
      else:
        pose_4x4 = torch.eye(4, dtype=torch.float32)

    self._last_pose_4x4 = pose_4x4

    rgb_img = torch.tensor(rgb_bgr[..., (2, 1, 0)].copy(), dtype=torch.float32)
    rgb_img = rgb_img.permute(2, 0, 1) / 255
    depth_img = torch.tensor(depth_m, dtype=torch.float32).unsqueeze(0)

    if (self.rgb_h != rgb_img.shape[-2] or self.rgb_w != rgb_img.shape[-1]):
      rgb_img = torch.nn.functional.interpolate(
        rgb_img.unsqueeze(0), size=(self.rgb_h, self.rgb_w),
        mode=self.interp_mode,
        antialias=self.interp_mode in ["bilinear", "bicubic"]).squeeze(0)

    if (self.depth_h != depth_img.shape[-2] or self.depth_w != depth_img.shape[-1]):
      depth_img = torch.nn.functional.interpolate(
        depth_img.unsqueeze(0), size=(self.depth_h, self.depth_w),
        mode="nearest-exact").squeeze(0)

    return dict(rgb_img=rgb_img, depth_img=depth_img, pose_4x4=pose_4x4,
                ts=torch.tensor([ts_ns / 1e9], dtype=torch.float32))

  def __iter__(self):
    f = 0
    with self._open_reader() as reader:
      topic_to_conn = {c.topic: c for c in reader.connections}
      rgb_conn = topic_to_conn[self.rgb_topic]
      depth_conn = topic_to_conn[self.depth_topic]
      rgb_msg = None
      depth_msg = None

      for conn, _, raw in reader.messages(connections=[rgb_conn, depth_conn]):
        msg = reader.deserialize(raw, conn.msgtype)
        ts_ns = _stamp_to_ns(msg.header.stamp)
        if conn.topic == self.rgb_topic:
          rgb_msg = (ts_ns, msg)
        else:
          depth_msg = (ts_ns, msg)

        while rgb_msg is not None and depth_msg is not None:
          rgb_ts, rgb_data = rgb_msg
          depth_ts, depth_data = depth_msg
          dt_ns = abs(rgb_ts - depth_ts)
          if dt_ns <= self.sync_slop_ns:
            rgb_bgr = decode_compressed_rgb(rgb_data)
            depth_m = decode_compressed_depth_to_meters(
              depth_data, depth_scale=self.depth_scale)
            frame = self._format_frame(rgb_bgr, depth_m, ts_ns=rgb_ts)
            rgb_msg = None
            depth_msg = None
            if frame is None:
              break
            if self.frame_skip > 0 and f % (self.frame_skip + 1) != 0:
              f += 1
              break
            f += 1
            yield frame
            break

          if rgb_ts < depth_ts:
            rgb_msg = None
          else:
            depth_msg = None
