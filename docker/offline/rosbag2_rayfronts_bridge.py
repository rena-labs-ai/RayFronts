#!/usr/bin/env python3
"""
Bridge a rosbag2 recording to RayFronts' `Ros2Subscriber` input format.

This node:
  - Subscribes to `sensor_msgs/msg/CompressedImage` for RGB and compressedDepth
  - Decodes + republishes:
      * RGB as `sensor_msgs/msg/Image` (`bgr8`)
      * Depth as `sensor_msgs/msg/Image` (`32FC1`, meters)
  - Publishes a `geometry_msgs/msg/PoseStamped` by querying TF at the synced
    image timestamp.

Defaults are set for the bag in `/home/keisuke/Downloads/experiment-2`.
"""

from __future__ import annotations

import argparse
import io
import struct
from dataclasses import dataclass

import numpy as np
from PIL import Image as PilImage

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time

import message_filters
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class Topics:
  rgb_in: str
  depth_in: str
  rgb_out: str
  depth_out: str
  pose_out: str
  camera_info: str


@dataclass(frozen=True)
class StaticTfSpec:
  parent: str
  child: str
  xyz_m: tuple[float, float, float]
  quat_xyzw: tuple[float, float, float, float]


def _decode_jpeg_or_png_bgr(compressed: bytes) -> np.ndarray:
  pil = PilImage.open(io.BytesIO(compressed)).convert("RGB")
  rgb = np.asarray(pil)
  if rgb.ndim != 3 or rgb.shape[2] != 3:
    raise ValueError(f"Expected RGB image with 3 channels, got shape={rgb.shape}")
  bgr = rgb[..., ::-1].copy()
  return bgr


def _find_png_start(data: bytes) -> int:
  idx = data.find(PNG_MAGIC)
  if idx != -1:
    return idx
  # Fallback: compressed_depth_image_transport prepends a 12-byte header.
  return 12 if len(data) > 12 else 0


def _decode_compressed_depth_to_meters(
  msg: CompressedImage,
  depth_scale: float,
) -> np.ndarray:
  # Typical format: "16UC1; compressedDepth"
  encoding = msg.format.split(";", 1)[0].strip()
  data = bytes(msg.data)

  png_start = _find_png_start(data)

  # If we have the 12-byte header, parse it (useful for 32FC1 inverse-depth mode).
  header = data[:png_start]
  inv_depth_params = None
  if len(header) >= 12:
    try:
      fmt_i32, depth_param_a, depth_param_b = struct.unpack("<iff", header[:12])
      inv_depth_params = (fmt_i32, depth_param_a, depth_param_b)
    except struct.error:
      inv_depth_params = None

  pil = PilImage.open(io.BytesIO(data[png_start:]))
  arr = np.asarray(pil)
  if arr.ndim == 3:
    arr = arr[..., 0]

  if encoding == "16UC1":
    depth_m = arr.astype(np.float32) * float(depth_scale)
  elif encoding == "32FC1":
    # Some producers store 32FC1 via inverse-depth quantization in the header.
    # If params are missing, pass through as float32.
    if inv_depth_params is None:
      depth_m = arr.astype(np.float32)
    else:
      fmt_i32, depth_param_a, depth_param_b = inv_depth_params
      # fmt_i32 == 0 corresponds to INV_DEPTH in compressed_depth_image_transport.
      q = arr.astype(np.float32)
      with np.errstate(divide="ignore", invalid="ignore"):
        depth_m = depth_param_a / (q - depth_param_b)
  else:
    depth_m = arr.astype(np.float32)

  depth_m[depth_m == 0] = np.nan
  return depth_m


def _bgr_to_ros_image(bgr: np.ndarray, header) -> Image:
  if bgr.dtype != np.uint8:
    raise ValueError(f"Expected uint8 BGR, got dtype={bgr.dtype}")
  if bgr.ndim != 3 or bgr.shape[2] != 3:
    raise ValueError(f"Expected HxWx3 BGR, got shape={bgr.shape}")

  msg = Image()
  msg.header = header
  msg.height = int(bgr.shape[0])
  msg.width = int(bgr.shape[1])
  msg.encoding = "bgr8"
  msg.is_bigendian = False
  msg.step = int(msg.width * 3)
  msg.data = bgr.tobytes()
  return msg


def _depth_to_ros_image(depth_m: np.ndarray, header) -> Image:
  if depth_m.ndim != 2:
    raise ValueError(f"Expected HxW depth, got shape={depth_m.shape}")
  depth_m = depth_m.astype(np.float32, copy=False)

  msg = Image()
  msg.header = header
  msg.height = int(depth_m.shape[0])
  msg.width = int(depth_m.shape[1])
  msg.encoding = "32FC1"
  msg.is_bigendian = False
  msg.step = int(msg.width * 4)
  msg.data = depth_m.tobytes()
  return msg


def _make_static_transform(spec: StaticTfSpec) -> TransformStamped:
  t = TransformStamped()
  # For static transforms, a zero stamp is fine (tf2 treats them as timeless).
  t.header.stamp.sec = 0
  t.header.stamp.nanosec = 0
  t.header.frame_id = spec.parent
  t.child_frame_id = spec.child
  t.transform.translation.x = float(spec.xyz_m[0])
  t.transform.translation.y = float(spec.xyz_m[1])
  t.transform.translation.z = float(spec.xyz_m[2])
  t.transform.rotation.x = float(spec.quat_xyzw[0])
  t.transform.rotation.y = float(spec.quat_xyzw[1])
  t.transform.rotation.z = float(spec.quat_xyzw[2])
  t.transform.rotation.w = float(spec.quat_xyzw[3])
  return t


class Rosbag2RayFrontsBridge(Node):
  def __init__(
    self,
    topics: Topics,
    target_frame: str,
    depth_scale: float,
    sync_slop_s: float,
    sync_queue: int,
    tf_timeout_s: float,
    pose_fallback: str,
    stats_period: int,
    static_tfs: list[StaticTfSpec],
    dropped_tf_edges: set[tuple[str, str]],
  ):
    super().__init__("rayfronts_rosbag2_bridge")

    self._topics = topics
    self._target_frame_arg = target_frame
    self._target_frame = None if target_frame in ("", "auto") else target_frame
    self._depth_scale = float(depth_scale)
    self._tf_timeout = Duration(seconds=float(tf_timeout_s))
    self._pose_fallback = str(pose_fallback)
    self._stats_period = int(stats_period)
    if self._pose_fallback not in ("drop", "last", "identity"):
      raise ValueError("--pose-fallback must be one of: drop, last, identity")

    self._tf_buffer = Buffer()
    self._tf_callback_group = ReentrantCallbackGroup()
    self._image_callback_group = ReentrantCallbackGroup()

    # Some recordings/simulations can contain TF loops. tf2 refuses to serve
    # transforms when the tree is cyclic; dropping a single edge can break the
    # cycle. (For `/home/keisuke/Downloads/experiment-2`, dropping
    # `camera_init -> map` fixes the loop.)
    self._dropped_tf_edges = dropped_tf_edges

    inserted = 0
    for spec in static_tfs:
      try:
        self._tf_buffer.set_transform_static(_make_static_transform(spec), "synthetic")
        inserted += 1
      except Exception as e:
        self.get_logger().warning(
          f"Failed to insert static TF {spec.parent} -> {spec.child}: {e}")
    if inserted:
      self.get_logger().info(f"Inserted {inserted} synthetic static TF link(s).")

    # NOTE: For bag replay, the effective QoS for `/tf` and `/tf_static` can
    # vary. Using BEST_EFFORT + VOLATILE maximizes compatibility with rosbag2
    # playback (RELIABLE subscribers will not match BEST_EFFORT publishers).
    tf_qos = QoSProfile(
      depth=100,
      reliability=ReliabilityPolicy.BEST_EFFORT,
      durability=DurabilityPolicy.VOLATILE,
    )
    tf_static_qos = QoSProfile(
      depth=100,
      reliability=ReliabilityPolicy.BEST_EFFORT,
      durability=DurabilityPolicy.VOLATILE,
    )
    self._tf_sub = self.create_subscription(
      TFMessage,
      "/tf",
      self._on_tf,
      qos_profile=tf_qos,
      callback_group=self._tf_callback_group,
    )
    self._tf_static_sub = self.create_subscription(
      TFMessage,
      "/tf_static",
      self._on_tf_static,
      qos_profile=tf_static_qos,
      callback_group=self._tf_callback_group,
    )

    self._rgb_pub = self.create_publisher(Image, topics.rgb_out, 10)
    self._depth_pub = self.create_publisher(Image, topics.depth_out, 10)
    self._pose_pub = self.create_publisher(PoseStamped, topics.pose_out, 10)

    qos = QoSProfile(
      depth=10,
      reliability=ReliabilityPolicy.BEST_EFFORT,
      durability=DurabilityPolicy.VOLATILE,
    )
    self._rgb_sub = message_filters.Subscriber(
      self,
      CompressedImage,
      topics.rgb_in,
      qos_profile=qos,
      callback_group=self._image_callback_group,
    )
    self._depth_sub = message_filters.Subscriber(
      self,
      CompressedImage,
      topics.depth_in,
      qos_profile=qos,
      callback_group=self._image_callback_group,
    )

    self._sync = message_filters.ApproximateTimeSynchronizer(
      [self._rgb_sub, self._depth_sub],
      queue_size=int(sync_queue),
      slop=float(sync_slop_s),
      allow_headerless=False,
    )
    self._sync.registerCallback(self._on_pair)

    self.get_logger().info(
      "Bridge ready. "
      f"RGB: {topics.rgb_in} -> {topics.rgb_out} | "
      f"Depth: {topics.depth_in} -> {topics.depth_out} | "
      f"Pose: TF -> {topics.pose_out}"
    )
    self._tf_fail_count = 0
    self._auto_target_fail_count = 0
    self._decode_fail_count = 0
    self._published_count = 0
    self._pose_fallback_used = 0
    self._pairs_seen = 0
    self._last_pose = None
    self._tf_msgs_seen = 0
    self._tf_static_msgs_seen = 0
    self._tf_transforms_seen = 0
    self._tf_static_transforms_seen = 0
    self._tf_transforms_inserted = 0
    self._tf_transforms_dropped = 0
    self._tf_static_transforms_inserted = 0
    self._tf_static_transforms_dropped = 0

  def _should_drop_tf_edge(self, parent_frame: str, child_frame: str) -> bool:
    # Normalize leading slashes.
    parent_frame = parent_frame.lstrip("/")
    child_frame = child_frame.lstrip("/")
    return (parent_frame, child_frame) in self._dropped_tf_edges

  @staticmethod
  def _norm_frame(frame_id: str) -> str:
    return (frame_id or "").lstrip("/")

  def _on_tf(self, msg: TFMessage):
    self._tf_msgs_seen += 1
    for t in msg.transforms:
      self._tf_transforms_seen += 1
      parent = self._norm_frame(t.header.frame_id)
      child = self._norm_frame(t.child_frame_id)
      if self._should_drop_tf_edge(parent, child):
        continue
      try:
        # tf2 frame ids should not start with '/'.
        t.header.frame_id = parent
        t.child_frame_id = child
        self._tf_buffer.set_transform(t, "rosbag2")
        self._tf_transforms_inserted += 1
      except Exception:
        # Ignore malformed transforms.
        self._tf_transforms_dropped += 1
        continue

  def _on_tf_static(self, msg: TFMessage):
    self._tf_static_msgs_seen += 1
    for t in msg.transforms:
      self._tf_static_transforms_seen += 1
      parent = self._norm_frame(t.header.frame_id)
      child = self._norm_frame(t.child_frame_id)
      if self._should_drop_tf_edge(parent, child):
        continue
      try:
        t.header.frame_id = parent
        t.child_frame_id = child
        self._tf_buffer.set_transform_static(t, "rosbag2")
        self._tf_static_transforms_inserted += 1
      except Exception:
        self._tf_static_transforms_dropped += 1
        continue

  def _auto_pick_target_frame(self, source_frame: str, stamp: Time) -> str | None:
    # Common TF roots first.
    candidates = ["map", "odom", "world"]

    # Try to extract TF roots from the buffer YAML (no YAML dependency).
    try:
      frames_yaml = self._tf_buffer.all_frames_as_yaml()
    except Exception:
      frames_yaml = ""

    parents = {}
    current = None
    for line in frames_yaml.splitlines():
      if not line.strip():
        continue
      if not line.startswith(" "):
        current = line[:-1] if line.endswith(":") else None
        continue
      if current and line.strip().startswith("parent:"):
        parent = line.split("parent:", 1)[1].strip().strip("'").strip('"')
        parents[current] = parent

    roots = set()
    for frame, parent in parents.items():
      if parent in ("", "NO_PARENT", "no parent", "None", "null"):
        roots.add(frame)
    if roots:
      candidates.extend(sorted(roots))

    for cand in candidates:
      try:
        if self._tf_buffer.can_transform(
          self._norm_frame(cand), self._norm_frame(source_frame), stamp, timeout=self._tf_timeout):
          return cand
      except Exception:
        continue
    return None

  def _lookup_pose(self, target_frame: str, source_frame: str, stamp_time: Time, stamp_msg):
    tf = self._tf_buffer.lookup_transform(
      self._norm_frame(target_frame), self._norm_frame(source_frame), stamp_time, timeout=self._tf_timeout)

    pose = PoseStamped()
    pose.header.stamp = stamp_msg
    pose.header.frame_id = target_frame
    pose.pose.position.x = tf.transform.translation.x
    pose.pose.position.y = tf.transform.translation.y
    pose.pose.position.z = tf.transform.translation.z
    pose.pose.orientation = tf.transform.rotation
    return pose

  def _make_identity_pose(self, frame_id: str, stamp_msg):
    pose = PoseStamped()
    pose.header.stamp = stamp_msg
    pose.header.frame_id = frame_id
    pose.pose.position.x = 0.0
    pose.pose.position.y = 0.0
    pose.pose.position.z = 0.0
    pose.pose.orientation.x = 0.0
    pose.pose.orientation.y = 0.0
    pose.pose.orientation.z = 0.0
    pose.pose.orientation.w = 1.0
    return pose

  def _on_pair(self, rgb_msg: CompressedImage, depth_msg: CompressedImage):
    self._pairs_seen += 1
    if self._stats_period > 0 and self._pairs_seen % self._stats_period == 1:
      self.get_logger().info(
        "Stats: "
        f"seen={self._pairs_seen} "
        f"published={self._published_count} "
        f"decode_fail={self._decode_fail_count} "
        f"tf_fail={self._tf_fail_count} "
        f"fallback_used={self._pose_fallback_used} "
        f"tf_msgs={self._tf_msgs_seen} tf_static_msgs={self._tf_static_msgs_seen} "
        f"tf_ins={self._tf_transforms_inserted}/{self._tf_transforms_seen} "
        f"tf_static_ins={self._tf_static_transforms_inserted}/{self._tf_static_transforms_seen}."
      )
    header = rgb_msg.header
    stamp_time = Time.from_msg(header.stamp)

    # Decode RGB + depth.
    try:
      bgr = _decode_jpeg_or_png_bgr(bytes(rgb_msg.data))
      depth_m = _decode_compressed_depth_to_meters(depth_msg, self._depth_scale)
    except Exception as e:
      self._decode_fail_count += 1
      self.get_logger().warning(f"Decode failed: {e}")
      return

    # Derive pose from TF.
    source_frame = self._norm_frame(header.frame_id)
    if not source_frame:
      self.get_logger().warning("RGB message has empty frame_id; cannot query TF.")
      return

    target_frame = self._target_frame
    if target_frame is None:
      target_frame = self._auto_pick_target_frame(source_frame, stamp_time)
      if target_frame is not None:
        self._target_frame = target_frame
        self.get_logger().info(f"Auto-selected TF world frame: {target_frame}")

    if target_frame is None:
      self._auto_target_fail_count += 1
      # Throttle: log every ~50 failures.
      if self._auto_target_fail_count % 50 == 1:
        self.get_logger().warning(
          "Could not auto-select TF world frame yet; try setting --target-frame.")
      return

    try:
      pose = self._lookup_pose(target_frame, source_frame, stamp_time, header.stamp)
    except Exception as e:
      self._tf_fail_count += 1
      if self._tf_fail_count == 1:
        try:
          frames_yaml = self._tf_buffer.all_frames_as_yaml()
          has_source = f"{source_frame}:" in frames_yaml
          self.get_logger().warning(
            "First TF failure: "
            f"target={target_frame} source={source_frame} "
            f"source_in_buffer={has_source} "
            f"tf_msgs={self._tf_msgs_seen} tf_static_msgs={self._tf_static_msgs_seen}"
          )
        except Exception:
          pass
      if self._pose_fallback == "last" and self._last_pose is not None:
        pose = PoseStamped()
        pose.header.stamp = header.stamp
        pose.header.frame_id = self._last_pose.header.frame_id
        pose.pose = self._last_pose.pose
        self._pose_fallback_used += 1
      elif self._pose_fallback == "identity":
        pose = self._make_identity_pose(target_frame, header.stamp)
        self._pose_fallback_used += 1
      else:
        # Throttle: log every ~100 failures.
        if self._tf_fail_count % 100 == 1:
          self.get_logger().warning(
            f"TF lookup failed ({target_frame} -> {source_frame}) "
            f"at t={stamp_time.nanoseconds}ns: {type(e).__name__}: {e}"
          )
        # TF might lag behind; drop this pair.
        return

    # Publish with a consistent timestamp (use the RGB header for everything).
    try:
      self._rgb_pub.publish(_bgr_to_ros_image(bgr, header))
      self._depth_pub.publish(_depth_to_ros_image(depth_m, header))
      self._pose_pub.publish(pose)
      self._last_pose = pose
      self._published_count += 1
      if self._stats_period > 0 and self._published_count % self._stats_period == 1:
        self.get_logger().info(
          "Published RGBD+pose pairs: "
          f"published={self._published_count} "
          f"seen={self._pairs_seen} "
          f"decode_fail={self._decode_fail_count} "
          f"tf_fail={self._tf_fail_count} "
          f"fallback_used={self._pose_fallback_used}."
        )
    except Exception as e:
      self.get_logger().warning(f"Publish failed: {e}")


def _parse_args(argv: list[str]):
  parser = argparse.ArgumentParser(
    description="Republish a rosbag2 RGB+compressedDepth stream into RayFronts topics.")
  parser.add_argument("--rgb-in", default="/base/camera/color/image_raw/compressed")
  parser.add_argument(
    "--depth-in",
    default="/base/camera/aligned_depth_to_color/image_raw/compressedDepth",
  )
  parser.add_argument("--rgb-out", default="/rayfronts/rgb")
  parser.add_argument("--depth-out", default="/rayfronts/depth")
  parser.add_argument("--pose-out", default="/rayfronts/pose")
  parser.add_argument("--camera-info", default="/base/camera/color/camera_info")
  parser.add_argument("--target-frame", default="auto",
                      help="TF frame to use as 'world' (default: auto; tries map/odom/world)")
  parser.add_argument("--depth-scale", type=float, default=0.001,
                      help="Scale for 16UC1 depth to meters (default: 0.001)")
  parser.add_argument("--sync-slop", type=float, default=0.1,
                      help="Approx sync slop for RGB/depth (seconds)")
  parser.add_argument("--sync-queue", type=int, default=30,
                      help="Approx sync queue size")
  parser.add_argument("--tf-timeout", type=float, default=0.2,
                      help="TF lookup timeout (seconds)")
  parser.add_argument(
    "--pose-fallback",
    default="drop",
    choices=["drop", "last", "identity"],
    help=("What to do when TF lookup fails for a frame: "
          "'drop' (default), 'last' (reuse last pose), or 'identity'."),
  )
  parser.add_argument(
    "--stats-period",
    type=int,
    default=0,
    help="Log publish stats every N published pairs (0 disables).",
  )
  parser.add_argument(
    "--static-tf",
    action="append",
    nargs=9,
    metavar=("PARENT", "CHILD", "X", "Y", "Z", "QX", "QY", "QZ", "QW"),
    help=("Insert a synthetic static TF (PARENT->CHILD). "
          "Translation in meters + quaternion (x y z w). Can be repeated."),
  )
  parser.add_argument(
    "--static-tf-identity",
    action="append",
    nargs=2,
    metavar=("PARENT", "CHILD"),
    help="Insert an identity synthetic static TF (PARENT->CHILD). Can be repeated.",
  )
  parser.add_argument(
    "--drop-tf-edge",
    action="append",
    nargs=2,
    metavar=("PARENT", "CHILD"),
    help=("Drop a TF edge (PARENT->CHILD) from /tf(/_static) before inserting "
          "into the tf2 buffer. Can be repeated."),
  )

  args, ros_args = parser.parse_known_args(argv)
  topics = Topics(
    rgb_in=args.rgb_in,
    depth_in=args.depth_in,
    rgb_out=args.rgb_out,
    depth_out=args.depth_out,
    pose_out=args.pose_out,
    camera_info=args.camera_info,
  )

  static_tfs: list[StaticTfSpec] = []
  for parent, child in (args.static_tf_identity or []):
    static_tfs.append(
      StaticTfSpec(parent=parent, child=child, xyz_m=(0.0, 0.0, 0.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0))
    )
  for spec in (args.static_tf or []):
    parent, child, x, y, z, qx, qy, qz, qw = spec
    static_tfs.append(
      StaticTfSpec(
        parent=parent,
        child=child,
        xyz_m=(float(x), float(y), float(z)),
        quat_xyzw=(float(qx), float(qy), float(qz), float(qw)),
      )
    )

  dropped_tf_edges: set[tuple[str, str]] = set()
  for parent, child in (args.drop_tf_edge or []):
    dropped_tf_edges.add((parent.lstrip("/"), child.lstrip("/")))

  return args, ros_args, topics, static_tfs, dropped_tf_edges


def main(argv: list[str] | None = None) -> int:
  args, ros_args, topics, static_tfs, dropped_tf_edges = _parse_args(
    list(argv) if argv is not None else None)

  rclpy.init(args=ros_args)
  node = Rosbag2RayFrontsBridge(
    topics=topics,
    target_frame=args.target_frame,
    depth_scale=args.depth_scale,
    sync_slop_s=args.sync_slop,
    sync_queue=args.sync_queue,
    tf_timeout_s=args.tf_timeout,
    pose_fallback=args.pose_fallback,
    stats_period=args.stats_period,
    static_tfs=static_tfs,
    dropped_tf_edges=dropped_tf_edges,
  )

  # A multi-threaded executor prevents TF starvation when image callbacks
  # wait for transforms with a timeout.
  executor = MultiThreadedExecutor(num_threads=4)
  executor.add_node(node)

  try:
    executor.spin()
  except KeyboardInterrupt:
    pass
  finally:
    try:
      executor.remove_node(node)
    except Exception:
      pass
    executor.shutdown()
    node.destroy_node()
    try:
      rclpy.shutdown()
    except Exception:
      # On SIGTERM, rclpy may already have shut down the context.
      pass

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
