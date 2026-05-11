"""Offline Waypoint Comparison Test Framework.

This module provides tools for comparing waypoints from episode JSON files
with server-predicted waypoints WITHOUT using the simulator.

The Uni-NaVid model maintains a video history (rgb_list), so this framework
supports loading video files and sending frames sequentially to the server.

Usage:
    from bench.policy.uninavid_waypoint.offline_waypoint_tester import OfflineWaypointTester

    tester = OfflineWaypointTester(
        server_url="http://localhost:8001",
        video_path="/path/to/recorded/video.mp4"  # or an image directory
    )

    # Test single episode
    results = tester.test_episode("model_test_episode_aliengo.json")

    # Save report
    tester.save_report([results], "offline_test_results.json")
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Union

import numpy as np
import requests
from PIL import Image

from OmniNavExt.envset.recording import resolve_recording_dirs, resolve_recording_payload


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RobotWaypoint:
    """A single robot waypoint."""
    frame: int
    xyz: Tuple[float, float, float]  # World-frame [x, y, z].
    yaw_deg: float                    # Heading, in degrees.
    time_s: float
    sim_step: Optional[int] = None
    distance_xy: float = 0.0
    distance_total_xy: float = 0.0
    command: Optional[Dict[str, float]] = None  # {v, w, lateral}


@dataclass
class EpisodeData:
    """Parsed episode data."""
    scenario_id: str
    instruction: str
    robot_type: str
    initial_position: Tuple[float, float, float]
    initial_yaw_deg: float
    gt_path: List[RobotWaypoint]
    vh_gt_path: Optional[Dict[str, List[Dict]]] = None


@dataclass
class ComparisonResult:
    """Per-step comparison result."""
    frame: int
    time_s: float
    # Ground truth
    gt_position: Tuple[float, float, float]
    gt_yaw_deg: float
    # Prediction (robot-local frame).
    pred_waypoints_local: List[List[float]]  # [[fwd, left, yaw], ...]
    # Prediction (world frame).
    pred_waypoints_world: List[Tuple[float, float, float]]
    # New metrics: deviation across the 5 predicted waypoints.
    waypoint_deviations: List[float] = field(default_factory=list)  # Distance from each pred point to the GT path.
    mean_waypoint_deviation_m: float = 0.0
    max_waypoint_deviation_m: float = 0.0
    trajectory_ndtw: float = 0.0            # nDTW of predicted trajectory vs. GT segment.
    direction_error_deg: float = 0.0        # Angle between predicted direction and GT path direction.
    arrive_probs: List[float] = field(default_factory=list)
    # Legacy metrics (kept for compatibility).
    path_deviation_m: float = 0.0     # Deviation of the first waypoint only (legacy).
    position_error_m: float = 0.0
    yaw_error_deg: float = 0.0


@dataclass
class EpisodeMetrics:
    """Episode-level aggregated metrics."""
    scenario_id: str
    num_steps: int
    # New metrics: aggregated waypoint deviation.
    mean_waypoint_deviation_m: float = 0.0
    max_waypoint_deviation_m: float = 0.0
    std_waypoint_deviation_m: float = 0.0
    # New metric: trajectory nDTW.
    mean_trajectory_ndtw: float = 0.0
    # New metric: direction error.
    mean_direction_error_deg: float = 0.0
    max_direction_error_deg: float = 0.0
    # Legacy metrics (kept for compatibility).
    mean_path_deviation_m: float = 0.0
    max_path_deviation_m: float = 0.0
    std_path_deviation_m: float = 0.0
    mean_position_error_m: float = 0.0
    max_position_error_m: float = 0.0
    std_position_error_m: float = 0.0
    mean_yaw_error_deg: float = 0.0
    max_yaw_error_deg: float = 0.0
    step_results: List[ComparisonResult] = field(default_factory=list)


# =============================================================================
# Data Loader
# =============================================================================

class WaypointDataLoader:
    """Loads and parses episode JSON files."""

    def load_episode(self, json_path: str) -> EpisodeData:
        """Load a single episode JSON file.

        Expected structure::

            {
              "scenarios": [{
                "id": "matterport_01",
                "robots": {
                  "entries": [{
                    "type": "aliengo",
                    "initial_pose": {"position": [x,y,z], "orientation_deg": yaw},
                    "rb_gt_waypoints": [...]  # legacy input, read-only compatibility
                  }]
                },
                "task": {"instruction": "..."}
              }]
            }
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        scenario: Dict[str, Any]
        robot_entry: Dict[str, Any]
        instruction: str
        scenario_id: str
        vh_gt_waypoints: Optional[Dict[str, List[Dict]]]

        if isinstance(data.get('scenarios'), list) and data['scenarios']:
            scenario = data['scenarios'][0]
            robot_entry = scenario['robots']['entries'][0]
            task = scenario.get('task', {})
            instruction = task.get('instruction', '')
            scenario_id = scenario.get('id', 'unknown')
            vh_gt_waypoints = scenario.get('virtual_humans', {}).get('vh_gt_waypoints')
        else:
            scenario = data
            robot_entry = {
                'type': data.get('robot_type', 'unknown'),
                'initial_pose': data.get('initial_pose', {}),
            }
            instruction = str(data.get('instruction', ''))
            scenario_id = str(data.get('scenario_id', Path(json_path).stem))
            vh_gt_waypoints = data.get('vh_gt_waypoints')

        recording = resolve_recording_payload(scenario, envset_path=Path(json_path))
        recording_waypoints = (recording or {}).get("gt_path") if isinstance(recording, dict) else None
        gt_path = []
        for wp in (recording_waypoints or robot_entry.get('rb_gt_waypoints', [])):
            gt_path.append(RobotWaypoint(
                frame=wp['frame'],
                xyz=tuple(wp['xyz']),
                yaw_deg=wp['yaw_deg'],
                time_s=wp['time_s'],
                sim_step=wp.get('sim_step'),
                distance_xy=wp.get('distance_xy', 0),
                distance_total_xy=wp.get('distance_total_xy', 0),
                command=wp.get('command')
            ))

        initial_pose = robot_entry.get('initial_pose', {})
        initial_position = tuple(initial_pose.get('position', [0, 0, 0]))
        initial_yaw_deg = initial_pose.get('orientation_deg', 0)

        return EpisodeData(
            scenario_id=scenario_id,
            instruction=instruction,
            robot_type=robot_entry.get('type', 'unknown'),
            initial_position=initial_position,
            initial_yaw_deg=initial_yaw_deg,
            gt_path=gt_path,
            vh_gt_path=vh_gt_waypoints,
        )

    def get_waypoint_at_frame(self, episode: EpisodeData, frame: int) -> Optional[RobotWaypoint]:
        """Return the waypoint for ``frame``, falling back to the closest one."""
        for wp in episode.gt_path:
            if wp.frame == frame:
                return wp
        if episode.gt_path:
            closest = min(episode.gt_path,
                          key=lambda w: abs(w.frame - frame))
            return closest
        return None


# =============================================================================
# Video/Image Loader
# =============================================================================

class VideoLoader:
    """Loads frames from a video file or a directory of images."""

    def __init__(self, source: str):
        """Args:
            source: video file path (.mp4, .avi) or image directory.
        """
        self.source = source
        self.is_video = source.endswith(('.mp4', '.avi', '.mov', '.mkv'))
        self._frames: Optional[List[np.ndarray]] = None
        self._frame_count = 0

    def load(self) -> int:
        """Load the video/images and return the total frame count."""
        if self.is_video:
            return self._load_video()
        else:
            return self._load_images()

    def _load_video(self) -> int:
        """Load all frames from a video file."""
        try:
            import cv2
        except ImportError:
            raise ImportError("cv2 is required for video loading. Install with: pip install opencv-python")

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.source}")

        self._frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._frames.append(frame_rgb)

        cap.release()
        self._frame_count = len(self._frames)
        print(f"[VideoLoader] Loaded {self._frame_count} frames from video: {self.source}")
        return self._frame_count

    def _load_images(self) -> int:
        """Load all frames from a directory of images."""
        if not os.path.isdir(self.source):
            raise ValueError(f"Not a directory: {self.source}")

        image_files = []
        for ext in ['.png', '.jpg', '.jpeg']:
            image_files.extend([f for f in os.listdir(self.source) if f.lower().endswith(ext)])

        # Sort by filename, assuming filenames embed a frame index.
        def extract_frame_num(filename):
            import re
            nums = re.findall(r'\d+', filename)
            return int(nums[0]) if nums else 0

        image_files.sort(key=extract_frame_num)

        self._frames = []
        for filename in image_files:
            path = os.path.join(self.source, filename)
            img = Image.open(path)
            self._frames.append(np.array(img.convert('RGB')))

        self._frame_count = len(self._frames)
        print(f"[VideoLoader] Loaded {self._frame_count} frames from directory: {self.source}")
        return self._frame_count

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Return the frame at ``frame_idx`` (or None if out of range)."""
        if self._frames is None:
            self.load()
        if 0 <= frame_idx < len(self._frames):
            return self._frames[frame_idx]
        return None

    def get_frames_range(self, start: int, end: int) -> List[np.ndarray]:
        """Return frames in the half-open range ``[start, end)``."""
        if self._frames is None:
            self.load()
        return self._frames[start:end]

    @property
    def frame_count(self) -> int:
        if self._frames is None:
            self.load()
        return self._frame_count


# =============================================================================
# Coordinate Converter
# =============================================================================

class CoordinateConverter:
    """Coordinate-frame conversion utilities."""

    @staticmethod
    def local_to_world(
        local_waypoints: List[List[float]],  # [[fwd, left, yaw], ...]
        robot_position: np.ndarray,           # [x, y, z]
        robot_yaw_rad: float
    ) -> List[Tuple[float, float, float]]:
        """Convert robot-local waypoints to the world frame.

        Local frame: x=forward, y=left (robot view).
        World frame: standard XY plane.

        Conversion (mirrors uninavid_waypoint_http_policy.py:377-380)::

            world_x = robot_x + fwd * cos(yaw) - left * sin(yaw)
            world_y = robot_y + fwd * sin(yaw) + left * cos(yaw)
        """
        cos_yaw = np.cos(robot_yaw_rad)
        sin_yaw = np.sin(robot_yaw_rad)

        world_points = []
        accumulated_fwd = 0.0
        accumulated_left = 0.0

        for wp in local_waypoints:
            if not isinstance(wp, (list, tuple)) or len(wp) < 2:
                continue

            delta_fwd = float(wp[0])
            delta_left = float(wp[1])

            accumulated_fwd += delta_fwd
            accumulated_left += delta_left

            world_x = robot_position[0] + accumulated_fwd * cos_yaw - accumulated_left * sin_yaw
            world_y = robot_position[1] + accumulated_fwd * sin_yaw + accumulated_left * cos_yaw
            world_z = robot_position[2]

            world_points.append((float(world_x), float(world_y), float(world_z)))

        return world_points


# =============================================================================
# Metrics
# =============================================================================

class WaypointMetrics:
    """Compute comparison metrics.

    Key metrics:
    - ``path_deviation``: shortest distance from a predicted point to the GT path
      (point-to-segment distance).
    - ``direction_error``: angle between predicted heading and GT path direction.
    """

    @staticmethod
    def point_to_segment_distance(
        point: Tuple[float, float],
        seg_start: Tuple[float, float],
        seg_end: Tuple[float, float]
    ) -> float:
        """Shortest 2D distance from a point to a line segment."""
        px, py = point[0], point[1]
        x1, y1 = seg_start[0], seg_start[1]
        x2, y2 = seg_end[0], seg_end[1]

        dx = x2 - x1
        dy = y2 - y1

        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq < 1e-10:
            # Degenerate segment (start == end).
            return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)

        # Projection parameter t, clamped to [0, 1] so we stay on the segment.
        t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))

        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        return float(np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2))

    @staticmethod
    def point_to_path_distance(
        point: Tuple[float, float, float],
        path: List[Tuple[float, float, float]]
    ) -> float:
        """Shortest 2D distance from a point to a polyline path (z is ignored)."""
        if len(path) < 2:
            if len(path) == 1:
                return np.sqrt((point[0] - path[0][0]) ** 2 + (point[1] - path[0][1]) ** 2)
            return float('inf')

        min_dist = float('inf')
        for i in range(len(path) - 1):
            dist = WaypointMetrics.point_to_segment_distance(
                (point[0], point[1]),
                (path[i][0], path[i][1]),
                (path[i + 1][0], path[i + 1][1])
            )
            min_dist = min(min_dist, dist)

        return float(min_dist)

    @staticmethod
    def compute_path_direction(
        path: List[Tuple[float, float, float]],
        current_idx: int,
        lookahead: int = 3
    ) -> Optional[float]:
        """Direction of ``path`` at ``current_idx`` (radians), averaged over ``lookahead`` points.

        Returns None if a direction cannot be computed (end of path or zero displacement).
        """
        if current_idx >= len(path) - 1:
            return None

        end_idx = min(current_idx + lookahead, len(path) - 1)
        if end_idx <= current_idx:
            return None

        dx = path[end_idx][0] - path[current_idx][0]
        dy = path[end_idx][1] - path[current_idx][1]

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None

        return float(np.arctan2(dy, dx))

    @staticmethod
    def compute_prediction_direction(
        pred_waypoints_local: List[List[float]],
        robot_yaw_rad: float
    ) -> Optional[float]:
        """World-frame heading (radians) of the predicted waypoints.

        Aggregates the first 3 local waypoints, then rotates into the world frame.
        Returns None if no usable displacement is present.
        """
        if not pred_waypoints_local:
            return None

        total_fwd = 0.0
        total_left = 0.0
        for wp in pred_waypoints_local[:3]:
            if len(wp) >= 2:
                total_fwd += wp[0]
                total_left += wp[1]

        if abs(total_fwd) < 1e-6 and abs(total_left) < 1e-6:
            return None

        local_dir = np.arctan2(total_left, total_fwd)
        world_dir = robot_yaw_rad + local_dir

        return float(world_dir)

    @staticmethod
    def direction_error(pred_dir_rad: float, gt_dir_rad: float) -> float:
        """Absolute angular error in degrees, wrapped to [0, 180]."""
        diff = np.rad2deg(pred_dir_rad - gt_dir_rad)
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return abs(diff)

    @staticmethod
    def position_error(pred: Tuple[float, float, float],
                       gt: Tuple[float, float, float]) -> float:
        """Position error: 2D Euclidean distance in the XY plane."""
        return float(np.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2))

    @staticmethod
    def yaw_error(pred_yaw_deg: float, gt_yaw_deg: float) -> float:
        """Absolute yaw error (degrees), normalized to [-180, 180]."""
        diff = pred_yaw_deg - gt_yaw_deg
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return abs(diff)

    @staticmethod
    def compute_all_waypoint_deviations(
        pred_waypoints_world: List[Tuple[float, float, float]],
        gt_path: List[Tuple[float, float, float]]
    ) -> List[float]:
        """Distance from each predicted waypoint (world frame) to the GT path."""
        deviations = []
        for pred_point in pred_waypoints_world:
            dist = WaypointMetrics.point_to_path_distance(pred_point, gt_path)
            deviations.append(dist)
        return deviations

    @staticmethod
    def compute_trajectory_ndtw(
        pred_path: List[Tuple[float, float, float]],
        gt_path: List[Tuple[float, float, float]],
        success_threshold: float = 1.0
    ) -> float:
        """nDTW between predicted and GT trajectories.

        ``nDTW = exp(-DTW / (len(ref_path) * success_threshold))``; higher is better.
        """
        if not pred_path or not gt_path or success_threshold <= 0:
            return 0.0

        n, m = len(pred_path), len(gt_path)
        dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        dtw[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # 2D Euclidean cost.
                cost = np.sqrt(
                    (pred_path[i - 1][0] - gt_path[j - 1][0]) ** 2 +
                    (pred_path[i - 1][1] - gt_path[j - 1][1]) ** 2
                )
                dtw[i, j] = cost + min(
                    dtw[i - 1, j],      # deletion
                    dtw[i, j - 1],      # insertion
                    dtw[i - 1, j - 1],  # match
                )

        dtw_cost = float(dtw[n, m])
        if not np.isfinite(dtw_cost):
            return 0.0

        norm = len(gt_path) * success_threshold
        if norm <= 0:
            return 0.0

        return float(np.exp(-dtw_cost / norm))

    @staticmethod
    def extract_gt_segment(
        gt_path: List[Tuple[float, float, float]],
        current_idx: int,
        num_points: int = 5
    ) -> List[Tuple[float, float, float]]:
        """Extract the GT segment of length ``num_points`` starting at ``current_idx``."""
        end_idx = min(current_idx + num_points, len(gt_path))
        return gt_path[current_idx:end_idx]

    @staticmethod
    def compute_episode_metrics(results: List[ComparisonResult], scenario_id: str = "") -> EpisodeMetrics:
        """Aggregate per-step results into episode-level metrics."""
        if not results:
            return EpisodeMetrics(
                scenario_id=scenario_id,
                num_steps=0,
                mean_waypoint_deviation_m=0.0,
                max_waypoint_deviation_m=0.0,
                std_waypoint_deviation_m=0.0,
                mean_trajectory_ndtw=0.0,
                mean_direction_error_deg=0.0,
                max_direction_error_deg=0.0,
                mean_path_deviation_m=0.0,
                max_path_deviation_m=0.0,
                std_path_deviation_m=0.0,
                mean_position_error_m=0.0,
                max_position_error_m=0.0,
                std_position_error_m=0.0,
                mean_yaw_error_deg=0.0,
                max_yaw_error_deg=0.0,
                step_results=[]
            )

        # New metric: per-waypoint deviation aggregated across steps.
        all_waypoint_devs = []
        for r in results:
            all_waypoint_devs.extend(r.waypoint_deviations)

        mean_wp_dev = float(np.mean(all_waypoint_devs)) if all_waypoint_devs else 0.0
        max_wp_dev = float(np.max(all_waypoint_devs)) if all_waypoint_devs else 0.0
        std_wp_dev = float(np.std(all_waypoint_devs)) if all_waypoint_devs else 0.0

        # New metric: trajectory nDTW.
        trajectory_ndtws = [r.trajectory_ndtw for r in results]
        mean_ndtw = float(np.mean(trajectory_ndtws)) if trajectory_ndtws else 0.0

        # Direction error.
        direction_errors = [r.direction_error_deg for r in results]

        # Legacy metrics.
        path_deviations = [r.path_deviation_m for r in results]
        pos_errors = [r.position_error_m for r in results]
        yaw_errors = [r.yaw_error_deg for r in results]

        return EpisodeMetrics(
            scenario_id=scenario_id,
            num_steps=len(results),
            mean_waypoint_deviation_m=mean_wp_dev,
            max_waypoint_deviation_m=max_wp_dev,
            std_waypoint_deviation_m=std_wp_dev,
            mean_trajectory_ndtw=mean_ndtw,
            mean_direction_error_deg=float(np.mean(direction_errors)),
            max_direction_error_deg=float(np.max(direction_errors)),
            mean_path_deviation_m=float(np.mean(path_deviations)),
            max_path_deviation_m=float(np.max(path_deviations)),
            std_path_deviation_m=float(np.std(path_deviations)),
            mean_position_error_m=float(np.mean(pos_errors)),
            max_position_error_m=float(np.max(pos_errors)),
            std_position_error_m=float(np.std(pos_errors)),
            mean_yaw_error_deg=float(np.mean(yaw_errors)),
            max_yaw_error_deg=float(np.max(yaw_errors)),
            step_results=results
        )


# =============================================================================
# Main Tester
# =============================================================================

class OfflineWaypointTester:
    """Offline waypoint comparison test framework.

    Supports two input modes:
      1. Video files (.mp4, .avi, ...).
      2. Image directories (filenames embed the frame index).

    Note: the Uni-NaVid model maintains a video history (rgb_list); each call to
    ``act`` accumulates one more frame. Frames must therefore be sent in order to
    mirror real online inference.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8001",
        video_source: Optional[str] = None,
        timeout: float = 60.0,
        predict_every_n_frames: int = 10,
        output_dir: str = "offline_test_results"
    ):
        """Args:
            server_url: waypoint prediction server URL.
            video_source: video file path or image directory.
            timeout: request timeout in seconds.
            predict_every_n_frames: run prediction every N frames.
            output_dir: directory to write report outputs.
        """
        self.server_url = server_url.rstrip('/')
        self.video_loader = VideoLoader(video_source) if video_source else None
        self.timeout = timeout
        self.predict_every_n_frames = predict_every_n_frames
        self.output_dir = output_dir

        self.data_loader = WaypointDataLoader()
        self.converter = CoordinateConverter()
        self.metrics = WaypointMetrics()

        self.session = requests.Session()

    def check_server_health(self) -> bool:
        """Return True if the server's /health endpoint responds OK."""
        try:
            resp = self.session.get(f"{self.server_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def reset_server(self, instruction: str) -> bool:
        """Reset the server's episode state and clear its video history."""
        try:
            resp = self.session.post(
                f"{self.server_url}/reset",
                json={"instruction": instruction, "task_type": "vln"},
                timeout=self.timeout
            )
            return resp.status_code == 200
        except Exception:
            return False

    def send_frame(self, image: np.ndarray, instruction: str) -> Dict[str, Any]:
        """Send a single frame to the server and return the prediction.

        The server appends this frame to its internal ``rgb_list`` history.

        Args:
            image: RGB image array (H, W, 3).
            instruction: navigation instruction.

        Returns:
            ``{"waypoints": [[fwd,left,yaw],...], "arrive_probs": [...], "step": N}``.
        """
        # Encode the image as base64 PNG.
        pil_img = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        resp = self.session.post(
            f"{self.server_url}/act",
            json={
                "instruction": instruction,
                "image": img_base64,
                "image_shape": list(image.shape)
            },
            timeout=self.timeout
        )
        return resp.json()

    def test_episode(
        self,
        episode_path: str,
        video_source: Optional[str] = None
    ) -> EpisodeMetrics:
        """Run the test on a single episode and return its aggregated metrics.

        Args:
            episode_path: episode JSON file path.
            video_source: optional override for the video source set at init.
        """
        episode = self.data_loader.load_episode(episode_path)
        print(f"[OfflineTester] Testing episode: {episode.scenario_id}")
        print(f"[OfflineTester] Instruction: {episode.instruction[:80]}...")
        print(f"[OfflineTester] Total GT waypoints: {len(episode.gt_path)}")

        loader = VideoLoader(video_source) if video_source else self.video_loader
        if loader is None:
            raise ValueError("No video source specified")

        total_frames = loader.load()
        print(f"[OfflineTester] Total video frames: {total_frames}")

        if not self.reset_server(episode.instruction):
            print("[OfflineTester] Warning: Failed to reset server")

        results = []
        waypoints = episode.gt_path

        # GT polyline used for path-deviation metrics.
        gt_path = [(wp.xyz[0], wp.xyz[1], wp.xyz[2]) for wp in waypoints]

        # frame -> waypoint and frame -> index lookups.
        frame_to_wp = {wp.frame: wp for wp in waypoints}
        frame_to_idx = {wp.frame: i for i, wp in enumerate(waypoints)}

        for frame_idx in range(total_frames):
            image = loader.get_frame(frame_idx)
            if image is None:
                continue

            if frame_idx % self.predict_every_n_frames != 0:
                continue

            current_wp = frame_to_wp.get(frame_idx)
            current_wp_idx = frame_to_idx.get(frame_idx)
            if current_wp is None:
                # Fall back to the closest GT waypoint and recover its index.
                current_wp = self.data_loader.get_waypoint_at_frame(episode, frame_idx)
                for i, wp in enumerate(waypoints):
                    if wp.frame == current_wp.frame:
                        current_wp_idx = i
                        break

            if current_wp is None:
                continue

            try:
                pred = self.send_frame(image, episode.instruction)

                robot_pos = np.array(current_wp.xyz)
                robot_yaw_rad = np.deg2rad(current_wp.yaw_deg)

                pred_waypoints = pred.get('waypoints', [])
                pred_world = self.converter.local_to_world(
                    pred_waypoints,
                    robot_pos,
                    robot_yaw_rad
                )

                # ============================================================
                # New metrics: aggregated waypoint deviation + nDTW.
                # ============================================================
                waypoint_deviations = []
                mean_wp_deviation = 0.0
                max_wp_deviation = 0.0
                trajectory_ndtw = 0.0
                direction_error = 0.0
                path_deviation = 0.0  # Legacy metric: first waypoint only.

                if pred_world:
                    # 1. Distance from each predicted waypoint to the GT path.
                    waypoint_deviations = self.metrics.compute_all_waypoint_deviations(
                        pred_world, gt_path
                    )
                    if waypoint_deviations:
                        mean_wp_deviation = float(np.mean(waypoint_deviations))
                        max_wp_deviation = float(np.max(waypoint_deviations))
                        path_deviation = waypoint_deviations[0]  # Legacy metric.

                    # 2. Trajectory nDTW against a GT segment of matching length.
                    gt_segment = self.metrics.extract_gt_segment(
                        gt_path,
                        current_wp_idx if current_wp_idx is not None else 0,
                        num_points=len(pred_world)
                    )
                    if gt_segment:
                        trajectory_ndtw = self.metrics.compute_trajectory_ndtw(
                            pred_world, gt_segment, success_threshold=1.0
                        )

                    # 3. Direction error: predicted heading vs. GT path direction.
                    pred_dir = self.metrics.compute_prediction_direction(
                        pred_waypoints, robot_yaw_rad
                    )
                    gt_dir = self.metrics.compute_path_direction(
                        gt_path, current_wp_idx if current_wp_idx is not None else 0
                    )

                    if pred_dir is not None and gt_dir is not None:
                        direction_error = self.metrics.direction_error(pred_dir, gt_dir)

                # ============================================================
                # Legacy metrics (kept for compatibility).
                # ============================================================
                pos_error = 0.0
                yaw_error = 0.0

                # Find the next GT waypoint after the current frame for comparison.
                next_wp = None
                for wp in waypoints:
                    if wp.frame > frame_idx:
                        next_wp = wp
                        break

                if pred_world and next_wp:
                    pos_error = self.metrics.position_error(pred_world[0], next_wp.xyz)
                    if len(pred_waypoints) > 0 and len(pred_waypoints[0]) >= 3:
                        pred_yaw_rad = pred_waypoints[0][2]
                        pred_yaw_deg = current_wp.yaw_deg + np.rad2deg(pred_yaw_rad)
                        yaw_error = self.metrics.yaw_error(pred_yaw_deg, next_wp.yaw_deg)

                result = ComparisonResult(
                    frame=frame_idx,
                    time_s=current_wp.time_s,
                    gt_position=current_wp.xyz,
                    gt_yaw_deg=current_wp.yaw_deg,
                    pred_waypoints_local=pred_waypoints,
                    pred_waypoints_world=pred_world,
                    waypoint_deviations=waypoint_deviations,
                    mean_waypoint_deviation_m=mean_wp_deviation,
                    max_waypoint_deviation_m=max_wp_deviation,
                    trajectory_ndtw=trajectory_ndtw,
                    direction_error_deg=direction_error,
                    arrive_probs=pred.get('arrive_probs', []),
                    path_deviation_m=path_deviation,
                    position_error_m=pos_error,
                    yaw_error_deg=yaw_error,
                )
                results.append(result)

                print(f"[OfflineTester] Frame {frame_idx}: "
                      f"mean_wp_dev={mean_wp_deviation:.3f}m, "
                      f"nDTW={trajectory_ndtw:.3f}, "
                      f"dir_err={direction_error:.1f}deg, "
                      f"step={pred.get('step', 0)}")

            except Exception as e:
                print(f"[OfflineTester] Error at frame {frame_idx}: {e}")
                continue

        return self.metrics.compute_episode_metrics(results, episode.scenario_id)

    def save_report(self, results: List[EpisodeMetrics], output_path: str):
        """Save the aggregated test report to ``output_path``."""
        if results:
            mean_wp_deviation = float(np.mean([r.mean_waypoint_deviation_m for r in results]))
            mean_trajectory_ndtw = float(np.mean([r.mean_trajectory_ndtw for r in results]))
            mean_direction_error = float(np.mean([r.mean_direction_error_deg for r in results]))
            # Legacy metrics.
            mean_path_deviation = float(np.mean([r.mean_path_deviation_m for r in results]))
            mean_pos_error = float(np.mean([r.mean_position_error_m for r in results]))
            mean_yaw_error = float(np.mean([r.mean_yaw_error_deg for r in results]))
        else:
            mean_wp_deviation = 0.0
            mean_trajectory_ndtw = 0.0
            mean_direction_error = 0.0
            mean_path_deviation = 0.0
            mean_pos_error = 0.0
            mean_yaw_error = 0.0

        report = {
            "timestamp": datetime.now().isoformat(),
            "server_url": self.server_url,
            "num_episodes": len(results),
            "summary": {
                # Primary (new) metrics.
                "mean_waypoint_deviation_m": mean_wp_deviation,
                "mean_trajectory_ndtw": mean_trajectory_ndtw,
                "mean_direction_error_deg": mean_direction_error,
                # Legacy metrics.
                "mean_path_deviation_m": mean_path_deviation,
                "mean_position_error_m": mean_pos_error,
                "mean_yaw_error_deg": mean_yaw_error,
            },
            "episodes": []
        }

        for r in results:
            episode_dict = {
                "scenario_id": r.scenario_id,
                "num_steps": r.num_steps,
                # New metrics.
                "mean_waypoint_deviation_m": r.mean_waypoint_deviation_m,
                "max_waypoint_deviation_m": r.max_waypoint_deviation_m,
                "std_waypoint_deviation_m": r.std_waypoint_deviation_m,
                "mean_trajectory_ndtw": r.mean_trajectory_ndtw,
                "mean_direction_error_deg": r.mean_direction_error_deg,
                "max_direction_error_deg": r.max_direction_error_deg,
                # Legacy metrics.
                "mean_path_deviation_m": r.mean_path_deviation_m,
                "max_path_deviation_m": r.max_path_deviation_m,
                "std_path_deviation_m": r.std_path_deviation_m,
                "mean_position_error_m": r.mean_position_error_m,
                "max_position_error_m": r.max_position_error_m,
                "std_position_error_m": r.std_position_error_m,
                "mean_yaw_error_deg": r.mean_yaw_error_deg,
                "max_yaw_error_deg": r.max_yaw_error_deg,
                "step_results": [asdict(s) for s in r.step_results]
            }
            report["episodes"].append(episode_dict)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"[OfflineTester] Report saved to: {output_path}")


# =============================================================================
# Data Directory Discovery
# =============================================================================

def discover_episode_data(
    data_dir: str,
    video_dir: Optional[str] = None
) -> List[Tuple[str, str]]:
    """Discover episode JSONs and matching video files under ``data_dir``.

    Three layouts are supported:

    1. Current bench-runner output (``video_dir`` is None)::

        {data_dir}/
        ├── {scene_id}/
        │   └── final_episode_{N}.json
        │   ├── video/
        │   │   └── final_episode_{N}/
        │   │       └── front/
        │   │           └── rgb.mp4
        │   └── path/
        │       └── final_episode_{N}/
        │           └── path.json

    2. Legacy bench-runner output (``video_dir`` is None)::

        {data_dir}/
        ├── {scene_id}/
        │   └── final_episode_{N}.json
        └── videos/
            └── {scene_id}/
                └── final_episode_{N}/
                    └── rgb.mp4

    3. Separate data and video directories (``video_dir`` provided)::

        {data_dir}/                          # Episode JSON directory
        └── {scene_id}/
            └── final_episode_{N}.json

        {video_dir}/                         # Video directory
        └── {scene_id}/
            └── final_episode_{N}/
                └── rgb.mp4

    Args:
        data_dir: directory containing episode JSON files.
        video_dir: directory containing video files. If omitted, the new bench
            layout is tried first, then the legacy ``data_dir/videos`` fallback.

    Returns:
        List of ``(episode_json_path, video_path)`` tuples.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    if video_dir:
        video_root = Path(video_dir)
        if not video_root.exists():
            raise ValueError(f"Video directory does not exist: {video_dir}")
    else:
        video_root = data_path / "videos"

    episodes = []

    for json_file in data_path.rglob("final_episode_*.json"):
        # Skip files that live under a ``videos`` directory.
        if "videos" in json_file.parts:
            continue

        scene_id = json_file.parent.name
        episode_name = json_file.stem  # e.g., "final_episode_1"

        # Search for a matching video across the supported layouts.
        new_video_root, _ = resolve_recording_dirs(json_file)
        candidates = [
            new_video_root / "front" / "rgb.mp4",
            video_root / scene_id / "video" / episode_name / "front" / "rgb.mp4",
            video_root / scene_id / episode_name / "front" / "rgb.mp4",
            video_root / scene_id / episode_name / "rgb.mp4",
        ]

        matched = False
        for video_path in candidates:
            if video_path.exists():
                episodes.append((str(json_file), str(video_path)))
                matched = True
                break

        if matched:
            continue

        # Fall back to image directories if no video file is present.
        image_candidates = [
            new_video_root / "front" / "rgb_frames",
            video_root / scene_id / "video" / episode_name / "front" / "rgb_frames",
            video_root / scene_id / episode_name / "front" / "rgb_frames",
            video_root / scene_id / episode_name,
        ]
        for image_dir in image_candidates:
            if image_dir.exists() and image_dir.is_dir():
                has_images = any(
                    f.suffix.lower() in ['.png', '.jpg', '.jpeg']
                    for f in image_dir.iterdir()
                )
                if has_images:
                    episodes.append((str(json_file), str(image_dir)))
                    matched = True
                    break

        if not matched:
            print(
                f"Warning: No video found for {json_file}, expected at "
                f"{candidates[0]} or {candidates[1]}"
            )

    return sorted(episodes)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Offline Waypoint Comparison Tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use a data directory (recommended; episodes and videos are auto-discovered).
  python -m bench.policy.uninavid_waypoint.offline_waypoint_tester \\
      --data-dir results/visual-uninavid

  # Separate data and video directories (OmniNavBench HuggingFace layout).
  python -m bench.policy.uninavid_waypoint.offline_waypoint_tester \\
      --data-dir "$OMNINAV_BENCH_DATASET_ROOT/annotations/train/concise/dog" \\
      --video-dir "$OMNINAV_BENCH_DATASET_ROOT/videos/train/concise/dog"

  # Manually specify episode and video.
  python -m bench.policy.uninavid_waypoint.offline_waypoint_tester \\
      --episode results/visual-uninavid/scene_01/final_episode_1.json \\
      --video results/visual-uninavid/scene_01/video/final_episode_1/front/rgb.mp4
        """
    )

    # Mutually exclusive data sources.
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--data-dir", type=str,
        help="Episode JSON directory (auto-discovers episodes)"
    )
    source_group.add_argument(
        "--episode", type=str,
        help="Episode JSON file path (requires --video)"
    )

    parser.add_argument(
        "--video-dir", type=str,
        help="Video directory (use with --data-dir when videos are in a separate location)"
    )
    parser.add_argument(
        "--video", type=str,
        help="Video file (.mp4) or image directory (required with --episode)"
    )
    parser.add_argument(
        "--server-url", type=str, default="http://localhost:8001",
        help="Waypoint server URL (default: http://localhost:8001)"
    )
    parser.add_argument(
        "--output", type=str, default="offline_test_results.json",
        help="Output report path (default: offline_test_results.json)"
    )
    parser.add_argument(
        "--predict-every", type=int, default=10,
        help="Predict every N frames (default: 10)"
    )

    args = parser.parse_args()

    if args.episode and not args.video:
        parser.error("--video is required when using --episode")
    if args.video_dir and not args.data_dir:
        parser.error("--video-dir can only be used with --data-dir")

    # Build the list of (episode, video) pairs to test.
    if args.data_dir:
        print(f"[OfflineTester] Discovering episodes from: {args.data_dir}")
        if args.video_dir:
            print(f"[OfflineTester] Video directory: {args.video_dir}")
        episode_pairs = discover_episode_data(args.data_dir, args.video_dir)
        if not episode_pairs:
            print(f"Error: No episodes found in {args.data_dir}")
            return
        print(f"[OfflineTester] Found {len(episode_pairs)} episode(s)")
    else:
        episode_pairs = [(args.episode, args.video)]

    # Each episode supplies its own video source, so leave it unset on the tester.
    tester = OfflineWaypointTester(
        server_url=args.server_url,
        video_source=None,
        predict_every_n_frames=args.predict_every
    )

    if not tester.check_server_health():
        print(f"Error: Server at {args.server_url} is not available")
        return

    all_results = []
    for episode_path, video_path in episode_pairs:
        print(f"\n[OfflineTester] Testing: {episode_path}")
        print(f"[OfflineTester] Video: {video_path}")
        try:
            results = tester.test_episode(episode_path, video_source=video_path)
            all_results.append(results)
        except Exception as e:
            print(f"[OfflineTester] Error testing {episode_path}: {e}")
            continue

    if not all_results:
        print("Error: No episodes were successfully tested")
        return

    tester.save_report(all_results, args.output)

    print("\n" + "=" * 50)
    print("=== Summary ===")
    print("=" * 50)
    print(f"Total episodes: {len(all_results)}")
    total_steps = sum(r.num_steps for r in all_results)
    print(f"Total steps: {total_steps}")

    if all_results:
        # Primary metrics: aggregated waypoint deviation + nDTW.
        mean_wp_dev = np.mean([r.mean_waypoint_deviation_m for r in all_results])
        max_wp_dev = max(r.max_waypoint_deviation_m for r in all_results)
        mean_ndtw = np.mean([r.mean_trajectory_ndtw for r in all_results])
        mean_dir_error = np.mean([r.mean_direction_error_deg for r in all_results])
        max_dir_error = max(r.max_direction_error_deg for r in all_results)

        print(f"\n--- Main Metrics (All 5 Waypoints) ---")
        print(f"Mean waypoint deviation: {mean_wp_dev:.3f}m")
        print(f"Max waypoint deviation: {max_wp_dev:.3f}m")
        print(f"Mean trajectory nDTW: {mean_ndtw:.3f}")
        print(f"Mean direction error: {mean_dir_error:.1f}deg")
        print(f"Max direction error: {max_dir_error:.1f}deg")

        # Legacy metrics (kept for compatibility).
        mean_path_dev = np.mean([r.mean_path_deviation_m for r in all_results])
        mean_pos_error = np.mean([r.mean_position_error_m for r in all_results])
        mean_yaw_error = np.mean([r.mean_yaw_error_deg for r in all_results])

        print(f"\n--- Legacy Metrics (First Waypoint Only) ---")
        print(f"Mean path deviation: {mean_path_dev:.3f}m")
        print(f"Mean position error: {mean_pos_error:.3f}m")
        print(f"Mean yaw error: {mean_yaw_error:.1f}deg")


if __name__ == "__main__":
    main()
