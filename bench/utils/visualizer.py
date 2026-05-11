"""
Visualizer module for OmniNavBench.

This module provides tools for recording videos and generating top-down visualizations
of robot trajectories. It is designed to fail fast if dependencies (like Isaac Sim)
are missing, avoiding silent failures.
"""

import threading
import queue
import json
import math
import os
import time
import numpy as np
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, Any, Dict, List

from OmniNavExt.envset.recording import build_recording_payload

# =============================================================================
# Async Video Writer
# =============================================================================

@dataclass
class FramePacket:
    """Frame data packet for async video writing."""
    idx: int
    rgb: np.ndarray
    depth: Optional[np.ndarray]
    metadata: Optional[Dict[str, Any]] = None
    image_path: Optional[Path] = None


@dataclass
class MultiCameraFramePacket:
    """Grouped frame packet for synchronized multi-camera video writing."""
    idx: int
    frames: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]
    metadata: Optional[Dict[str, Any]] = None


class AsyncVideoWriter:
    """Background thread video writer for RGB and Depth streams."""

    def __init__(
        self,
        rgb_video_path: Optional[Union[str, Path]] = None,
        depth_video_path: Optional[Union[str, Path]] = None,
        fps: int = 10,
        max_queue: int = 256,
        recording_json_path: Optional[Union[str, Path]] = None,
        recording_instruction: Optional[str] = None,
    ):
        self._rgb_video_path = Path(rgb_video_path) if rgb_video_path else None
        self._depth_video_path = Path(depth_video_path) if depth_video_path else None
        self._fps = fps
        self._q: "queue.Queue[Optional[FramePacket]]" = queue.Queue(maxsize=int(max_queue))
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._rgb_writer = None
        self._depth_writer = None
        self._recording_json_path = Path(recording_json_path) if recording_json_path else None
        self._recording_instruction = str(recording_instruction or "")
        self._recording_path: List[Dict[str, Any]] = []
        self._written_rgb_frames = 0
        self._worker_error: Optional[BaseException] = None

        # Start the thread immediately
        self._th.start()

    def push(
        self,
        idx: int,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        image_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Push a frame to the write queue."""
        pkt = FramePacket(
            idx=idx,
            rgb=rgb,
            depth=depth,
            metadata=metadata,
            image_path=Path(image_path) if image_path else None,
        )
        # Bench/replay outputs are authoritative data; block instead of silently dropping frames.
        self._q.put(pkt, block=True)

    def close(self) -> None:
        """Signal the writer thread to stop and wait for it to finish."""
        self._q.put(None)
        if self._th.is_alive():
            self._th.join()
        if self._worker_error is not None:
            raise RuntimeError("AsyncVideoWriter worker failed") from self._worker_error

    def _loop(self) -> None:
        """Background loop to consume frames and write to disk."""
        try:
            while True:
                pkt = self._q.get()
                if pkt is None:
                    break

                rgb = pkt.rgb
                depth = pkt.depth
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                # Initialize writers on first frame
                if self._rgb_writer is None and self._rgb_video_path is not None:
                    self._rgb_video_path.parent.mkdir(parents=True, exist_ok=True)
                    h, w = rgb.shape[:2]
                    self._rgb_writer = cv2.VideoWriter(
                        str(self._rgb_video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self._fps,
                        (w, h),
                    )

                if self._depth_writer is None and self._depth_video_path and depth is not None:
                    self._depth_video_path.parent.mkdir(parents=True, exist_ok=True)
                    h, w = depth.shape[:2]
                    self._depth_writer = cv2.VideoWriter(
                        str(self._depth_video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self._fps,
                        (w, h),
                        isColor=False,
                    )

                # Write frames
                # RGB: Convert RGB to BGR for OpenCV
                if self._rgb_writer:
                    self._rgb_writer.write(bgr)
                    self._record_trajectory_entry(pkt.metadata)

                # Depth: Normalize and write
                if self._depth_writer and depth is not None:
                    # Normalize depth for visualization (0-10m -> 0-255)
                    # This is a simple visualization normalization
                    d_vis = np.clip(depth, 0, 10.0) / 10.0 * 255.0
                    d_vis = d_vis.astype(np.uint8)
                    self._depth_writer.write(d_vis)

                if pkt.image_path is not None:
                    pkt.image_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(pkt.image_path), bgr)
        except Exception as exc:
            self._worker_error = exc
        finally:
            # Ensure writers are released to write file headers/footers
            if self._rgb_writer:
                self._rgb_writer.release()
            if self._depth_writer:
                self._depth_writer.release()
            self._write_recording_json()

    def _record_trajectory_entry(self, metadata: Optional[Dict[str, Any]]) -> None:
        if metadata is None:
            self._written_rgb_frames += 1
            return

        pose = metadata.get("pose") or {}
        if self._recording_json_path is not None:
            recording_entry = {
                "frame": int(metadata.get("frame", self._written_rgb_frames)),
                "time_s": float(metadata["sim_time_s"]),
                "xyz": [
                    float(pose["x"]),
                    float(pose["y"]),
                    float(pose["z"]),
                ],
                "yaw_deg": float(math.degrees(float(pose["yaw"]))),
            }
            sim_step = metadata.get("sim_step")
            if sim_step is not None:
                recording_entry["sim_step"] = int(sim_step)
            self._recording_path.append(recording_entry)
        self._written_rgb_frames += 1

    def _build_video_manifest(self, base_path: Path) -> Dict[str, Dict[str, Optional[str]]]:
        return {
            "front": {
                "rgb": (
                    str(os.path.relpath(self._rgb_video_path, start=base_path))
                    if self._rgb_video_path is not None
                    else None
                ),
                "depth": (
                    str(os.path.relpath(self._depth_video_path, start=base_path))
                    if self._depth_video_path is not None
                    else None
                ),
            }
        }

    def _write_recording_json(self) -> None:
        if self._recording_json_path is None:
            return

        distance_total_xy = 0.0
        prev_xy: Optional[Tuple[float, float]] = None
        path_entries: List[Dict[str, Any]] = []
        for entry in self._recording_path:
            xyz = entry["xyz"]
            cur_xy = (float(xyz[0]), float(xyz[1]))
            if prev_xy is None:
                distance_xy = 0.0
            else:
                dx = cur_xy[0] - prev_xy[0]
                dy = cur_xy[1] - prev_xy[1]
                distance_xy = float((dx * dx + dy * dy) ** 0.5)
                distance_total_xy += distance_xy
            wp = dict(entry)
            wp["distance_xy"] = float(distance_xy)
            wp["distance_total_xy"] = float(distance_total_xy)
            path_entries.append(wp)
            prev_xy = cur_xy

        metadata: Dict[str, Any] = {
            "source": "bench_visualizer",
            "sample_count": len(path_entries),
            "distance_total_xy": float(distance_total_xy),
            "camera_names": ["front"],
            "videos": self._build_video_manifest(self._recording_json_path.parent),
        }
        if path_entries:
            metadata["robot_initial_pose"] = {
                "xyz": list(path_entries[0]["xyz"]),
                "yaw_deg": float(path_entries[0]["yaw_deg"]),
            }
            metadata["robot_final_pose"] = {
                "xyz": list(path_entries[-1]["xyz"]),
                "yaw_deg": float(path_entries[-1]["yaw_deg"]),
            }

        payload = build_recording_payload(
            instruction=self._recording_instruction,
            gt_path=path_entries,
            metadata=metadata,
        )
        self._recording_json_path.parent.mkdir(parents=True, exist_ok=True)
        self._recording_json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class MultiCameraAsyncVideoWriter:
    """Background writer for synchronized multi-camera RGB/depth replay output."""

    def __init__(
        self,
        camera_outputs: Dict[str, Dict[str, Optional[Union[str, Path]]]],
        fps: int = 10,
        max_queue: int = 256,
        recording_json_path: Optional[Union[str, Path]] = None,
        recording_instruction: Optional[str] = None,
    ):
        if not camera_outputs:
            raise ValueError("camera_outputs cannot be empty")

        self._camera_names = list(camera_outputs.keys())
        self._camera_outputs: Dict[str, Dict[str, Optional[Path]]] = {}
        for name, paths in camera_outputs.items():
            rgb_path = paths.get("rgb")
            if rgb_path is None:
                raise ValueError(f"camera_outputs[{name!r}] missing rgb path")
            depth_path = paths.get("depth")
            self._camera_outputs[name] = {
                "rgb": Path(rgb_path),
                "depth": Path(depth_path) if depth_path is not None else None,
            }

        self._fps = fps
        self._q: "queue.Queue[Optional[MultiCameraFramePacket]]" = queue.Queue(maxsize=int(max_queue))
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._rgb_writers: Dict[str, Any] = {}
        self._depth_writers: Dict[str, Any] = {}
        self._recording_json_path = Path(recording_json_path) if recording_json_path else None
        self._recording_instruction = str(recording_instruction or "")
        self._recording_path: List[Dict[str, Any]] = []
        self._written_rgb_frames = 0
        self._worker_error: Optional[BaseException] = None

        self._th.start()

    def push(
        self,
        idx: int,
        frames: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        pkt = MultiCameraFramePacket(idx=idx, frames=frames, metadata=metadata)
        # Replay output is authoritative data; block instead of silently dropping frames.
        self._q.put(pkt, block=True)

    def close(self) -> None:
        self._q.put(None)
        if self._th.is_alive():
            self._th.join()
        if self._worker_error is not None:
            raise RuntimeError("MultiCameraAsyncVideoWriter worker failed") from self._worker_error

    def _loop(self) -> None:
        try:
            while True:
                pkt = self._q.get()
                if pkt is None:
                    break

                for camera_name in self._camera_names:
                    if camera_name not in pkt.frames:
                        raise RuntimeError(f"Missing frame for camera '{camera_name}' in grouped packet")
                    rgb, depth = pkt.frames[camera_name]
                    self._ensure_camera_writers(camera_name=camera_name, rgb=rgb, depth=depth)

                    rgb_writer = self._rgb_writers[camera_name]
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    rgb_writer.write(bgr)

                    depth_writer = self._depth_writers.get(camera_name)
                    if depth_writer is not None and depth is not None:
                        depth_writer.write(self._normalize_depth_frame(depth))

                self._record_trajectory_entry(pkt.metadata)
        except Exception as exc:
            self._worker_error = exc
        finally:
            for writer in self._rgb_writers.values():
                writer.release()
            for writer in self._depth_writers.values():
                writer.release()
            self._write_recording_json()

    def _ensure_camera_writers(
        self,
        *,
        camera_name: str,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
    ) -> None:
        if camera_name not in self._rgb_writers:
            rgb_path = self._camera_outputs[camera_name]["rgb"]
            rgb_path.parent.mkdir(parents=True, exist_ok=True)
            h, w = rgb.shape[:2]
            writer = cv2.VideoWriter(
                str(rgb_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self._fps,
                (w, h),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open RGB video writer for camera '{camera_name}': {rgb_path}")
            self._rgb_writers[camera_name] = writer

        depth_path = self._camera_outputs[camera_name]["depth"]
        if (
            camera_name not in self._depth_writers
            and depth_path is not None
            and depth is not None
        ):
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            h, w = depth.shape[:2]
            writer = cv2.VideoWriter(
                str(depth_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self._fps,
                (w, h),
                isColor=False,
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open depth video writer for camera '{camera_name}': {depth_path}")
            self._depth_writers[camera_name] = writer

    @staticmethod
    def _normalize_depth_frame(depth: np.ndarray) -> np.ndarray:
        d_vis = np.clip(depth, 0, 10.0) / 10.0 * 255.0
        return d_vis.astype(np.uint8)

    def _record_trajectory_entry(self, metadata: Optional[Dict[str, Any]]) -> None:
        if metadata is None:
            self._written_rgb_frames += 1
            return

        pose = metadata.get("pose") or {}
        if self._recording_json_path is not None:
            recording_entry = {
                "frame": int(metadata.get("frame", self._written_rgb_frames)),
                "time_s": float(metadata["sim_time_s"]),
                "xyz": [
                    float(pose["x"]),
                    float(pose["y"]),
                    float(pose["z"]),
                ],
                "yaw_deg": float(math.degrees(float(pose["yaw"]))),
            }
            sim_step = metadata.get("sim_step")
            if sim_step is not None:
                recording_entry["sim_step"] = int(sim_step)
            self._recording_path.append(recording_entry)
        self._written_rgb_frames += 1

    def _build_video_manifest(self, base_path: Path) -> Dict[str, Dict[str, Optional[str]]]:
        return {
            name: {
                "rgb": str(os.path.relpath(self._camera_outputs[name]["rgb"], start=base_path)),
                "depth": (
                    str(os.path.relpath(self._camera_outputs[name]["depth"], start=base_path))
                    if self._camera_outputs[name]["depth"] is not None
                    else None
                ),
            }
            for name in self._camera_names
        }

    def _write_recording_json(self) -> None:
        if self._recording_json_path is None:
            return

        distance_total_xy = 0.0
        prev_xy: Optional[Tuple[float, float]] = None
        path_entries: List[Dict[str, Any]] = []
        for entry in self._recording_path:
            xyz = entry["xyz"]
            cur_xy = (float(xyz[0]), float(xyz[1]))
            if prev_xy is None:
                distance_xy = 0.0
            else:
                dx = cur_xy[0] - prev_xy[0]
                dy = cur_xy[1] - prev_xy[1]
                distance_xy = float((dx * dx + dy * dy) ** 0.5)
                distance_total_xy += distance_xy
            wp = dict(entry)
            wp["distance_xy"] = float(distance_xy)
            wp["distance_total_xy"] = float(distance_total_xy)
            path_entries.append(wp)
            prev_xy = cur_xy

        metadata: Dict[str, Any] = {
            "source": "replay_video",
            "sample_count": len(path_entries),
            "distance_total_xy": float(distance_total_xy),
            "camera_names": list(self._camera_names),
            "videos": self._build_video_manifest(self._recording_json_path.parent),
        }
        if path_entries:
            metadata["robot_initial_pose"] = {
                "xyz": list(path_entries[0]["xyz"]),
                "yaw_deg": float(path_entries[0]["yaw_deg"]),
            }
            metadata["robot_final_pose"] = {
                "xyz": list(path_entries[-1]["xyz"]),
                "yaw_deg": float(path_entries[-1]["yaw_deg"]),
            }

        payload = build_recording_payload(
            instruction=self._recording_instruction,
            gt_path=path_entries,
            metadata=metadata,
        )
        self._recording_json_path.parent.mkdir(parents=True, exist_ok=True)
        self._recording_json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


# =============================================================================
# Visualizer
# =============================================================================

class Visualizer:
    """
    Handles real-time visualization of robot trajectory on a top-down map.

    Combines the first-person RGB view with a live-updated trajectory map
    and records the result to video.
    """

    def __init__(
        self,
        env: Any,
        output_rgb_path: Optional[Union[str, Path]] = None,
        output_depth_path: Optional[Union[str, Path]] = None,
        fps: int = 10,
        image_output_dir: Optional[Union[str, Path]] = None,
        image_interval_s: float = 1.0,
        map_resolution: float = 0.05,
        recording_json_path: Optional[Union[str, Path]] = None,
        recording_instruction: Optional[str] = None,
    ):
        """
        Args:
            env: The simulation environment instance.
            output_rgb_path: Path to save the combined video.
            output_depth_path: Path to save depth video.
            fps: Video frame rate for RGB/depth mp4 outputs.
            image_output_dir: Directory to save periodic RGB frames.
            image_interval_s: Sim-time interval between saved RGB frames.
            map_resolution: Meters per pixel for the top-down map.
        """
        if output_rgb_path is None and image_output_dir is None:
            raise ValueError("Visualizer requires at least one output target")
        if image_interval_s <= 0.0:
            raise ValueError("image_interval_s must be > 0")

        self.env = env
        self.map_res = map_resolution
        self._image_output_dir = Path(image_output_dir) if image_output_dir else None
        self._image_interval_s = float(image_interval_s)
        self._next_image_time_s = 0.0

        # Initialize the async writer
        self.writer = AsyncVideoWriter(
            output_rgb_path,
            output_depth_path,
            fps=fps,
            recording_json_path=recording_json_path,
            recording_instruction=recording_instruction,
        )

        # State for mapping
        self.map_canvas: Optional[np.ndarray] = None
        self.map_origin: Tuple[float, float] = (0.0, 0.0) # min_x, max_y
        self.last_pixel_pos: Optional[Tuple[int, int]] = None

        # Lazy initialization flag
        self._map_initialized = False

    def _image_path_for_step(self, step_idx: int, sim_time_s: float) -> Optional[Path]:
        if self._image_output_dir is None:
            return None

        if sim_time_s + 1e-9 < self._next_image_time_s:
            return None

        while self._next_image_time_s <= sim_time_s + 1e-9:
            self._next_image_time_s += self._image_interval_s

        return self._image_output_dir / f"frame_{step_idx:06d}.jpg"

    def _init_map(self) -> None:
        """
        Generates the static top-down map from the simulation environment.
        Raises RuntimeError if map generation fails.
        """
        # 1. Check dependencies
        try:
            import omni.usd
            from isaacsim.asset.gen.omap.bindings import _omap
            import omni.physx
            from pxr import Usd, UsdGeom
        except ImportError as e:
            raise RuntimeError(
                "Visualizer requires Isaac Sim python environment with `omni` and `isaacsim` modules."
            ) from e

        # 2. Compute bounds of the stage (NavMesh volumes or full stage)
        stage = omni.usd.get_context().get_stage()
        if not stage:
            raise RuntimeError("No USD stage found. Cannot generate map.")

        # Compute AABB of the world using bbox cache
        # Using a simplified approach: Scan the NavMesh if possible, else Stage.
        # cache = omni.usd.get_context().get_bbox_cache()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
        root_prim = stage.GetPseudoRoot()
        bound = cache.ComputeWorldBound(root_prim)
        box = bound.GetBox()
        min_xyz = box.GetMin()
        max_xyz = box.GetMax()

        # 3. Configure Generator
        physx = omni.physx.acquire_physx_interface()
        stage_id = omni.usd.get_context().get_stage_id()
        generator = _omap.Generator(physx, stage_id)

        # Colors: 0=Unknown (Black), 255=Free (White), 128=Occupied (Gray)
        # We invert this for better visuals: Free=White (255), Occupied=Black (0)
        # But generator uses specific int IDs.
        occ_val, free_val, unk_val = 0, 255, 128
        generator.update_settings(self.map_res, occ_val, free_val, unk_val)

        # Set bounds (add some padding)
        pad = 1.0
        min_x, min_y = min_xyz[0] - pad, min_xyz[1] - pad
        max_x, max_y = max_xyz[0] + pad, max_xyz[1] + pad

        # Z-slice: We assume the robot is on a floor.
        # We scan slightly above the min_z to avoid floor Z-fighting, and up to ceiling.
        scan_z_min = min_xyz[2] + 0.2
        scan_z_max = min(min_xyz[2] + 2.5, max_xyz[2])

        generator.set_transform(
            (0.0, 0.0, 0.0),
            (min_x, min_y, scan_z_min),
            (max_x, max_y, scan_z_max)
        )

        # 4. Generate
        generator.generate2d()
        buf = generator.get_buffer()
        if not buf:
            raise RuntimeError("Failed to generate occupancy map: Buffer empty.")

        dims = generator.get_dimensions()
        w, h = dims[0], dims[1]

        # Convert buffer to numpy
        if len(buf) > 0 and isinstance(buf[0], str):
             data_bytes = "".join(buf).encode('latin1')
             grid = np.frombuffer(data_bytes, dtype=np.uint8)
        else:
             grid = np.asarray(buf, dtype=np.uint8)

        grid = grid.reshape((h, w))

        # Convert to BGR canvas
        self.map_canvas = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)

        # Store transform info
        # Origin for top-left of image corresponds to (min_x, max_y) in world.
        # Pixel calculation:
        # col = (x - min_x) / res
        # row = (max_y - y) / res
        self.map_origin = (min_x, max_y)
        self._map_initialized = True

    def _world_to_pixel(self, pos: np.ndarray) -> Tuple[int, int]:
        """Maps world (x,y) to pixel (col, row)."""
        x, y = pos[0], pos[1]
        min_x, max_y = self.map_origin

        col = int((x - min_x) / self.map_res)
        row = int((max_y - y) / self.map_res)

        # Clamp to be safe
        if self.map_canvas is None:
            return 0, 0

        h, w = self.map_canvas.shape[:2]
        col = int(np.clip(col, 0, w - 1))
        row = int(np.clip(row, 0, h - 1))

        return col, row

    def record_step(self, step_idx: int, obs: Any) -> None:
        """
        Process a simulation step: update map and write video frame.

        Args:
            step_idx: Current simulation step index.
            obs: Observation object containing .rgb (H,W,3) and .position (3,).
        """
        # --- MAP LOGIC DISABLED (Commented out) ---
        """
        if not self._map_initialized:
            self._init_map()

        # 1. Update Map
        if self.map_canvas is None:
            raise RuntimeError("Map canvas is None after initialization.")

        curr_pixel = self._world_to_pixel(obs.position)

        if self.last_pixel_pos is not None:
            # Draw line from previous pos to current pos
            cv2.line(
                self.map_canvas,
                self.last_pixel_pos,
                curr_pixel,
                (0, 0, 255), # Red trajectory
                2
            )
        else:
            # Mark start point
            cv2.circle(self.map_canvas, curr_pixel, 4, (0, 255, 0), -1) # Green start

        self.last_pixel_pos = curr_pixel

        # 2. Compose Image
        # Resize map to match RGB height
        rgb = obs.rgb
        h_rgb, w_rgb = rgb.shape[:2]
        h_map, w_map = self.map_canvas.shape[:2]

        # Scale map to have same height as RGB
        # Use nearest neighbor for map to keep it sharp-ish and fast
        scale = h_rgb / float(h_map)
        w_new = int(w_map * scale)
        map_resized = cv2.resize(self.map_canvas, (w_new, h_rgb), interpolation=cv2.INTER_NEAREST)

        # Concatenate horizontally
        combined = np.hstack((rgb, map_resized))

        # 3. Push to writer
        self.writer.push(step_idx, combined, obs.depth)
        """

        # --- FALLBACK: DIRECT RGB RECORDING ---
        image_path = self._image_path_for_step(step_idx, float(obs.time_s))
        now = time.time()
        orientation = tuple(getattr(obs, "orientation", (1.0, 0.0, 0.0, 0.0)))
        yaw = self._quat_to_yaw(orientation)
        metadata = {
            "timestamp": float(now),
            "timestamp_ms": int(now * 1000.0),
            "sim_time_s": float(obs.time_s),
            "sim_step": int(step_idx),
            "pose": {
                "x": float(obs.position[0]),
                "y": float(obs.position[1]),
                "z": float(obs.position[2]),
                "yaw": float(yaw),
            },
        }
        self.writer.push(step_idx, obs.rgb, obs.depth, metadata=metadata, image_path=image_path)

    def close(self) -> None:
        """Clean up resources."""
        self.writer.close()

    @staticmethod
    def _quat_to_yaw(quat: Tuple[float, float, float, float]) -> float:
        """Extract yaw (z-rotation) from quaternion in (w, x, y, z) order."""
        w, x, y, z = [float(v) for v in quat]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(math.atan2(siny_cosp, cosy_cosp))
