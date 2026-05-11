"""Uni-NaVid Waypoint Points HTTP client policy for OmniNavBench.

This policy communicates with a Uni-NaVid Waypoint HTTP server and executes
local waypoint predictions with go_toward_point controller.

================================================================================
How to Run
================================================================================

1. Start the server (terminal 1, ``uni-navid`` env):

   conda activate uni-navid
   cd /path/to/Uni-NaVid_waypoints
   python -m bench.policy.uninavid_waypoint.uninavid_waypoint_server_points \
       --uninavid_path /path/to/Uni-NaVid_waypoints \
       --model_path /path/to/Uni-NaVid_waypoints/model_zoo/omninav_waypoint_lora \
       --model_base /path/to/Uni-NaVid_waypoints/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2 \
       --port 8002 \
       --debug

2. Run the benchmark (terminal 2, ``isaaclab`` env):

   conda activate isaaclab
   cd $OMNINAV_REPO_ROOT

   # Single JSON file
   python runBench.py \
       --config configs/aliengoh1_test.yaml \
       --scene-root $OMNINAV_SCENE_ROOT \
       --envset $OMNINAV_REPO_ROOT/model_test_episode_aliengo.json \
       --output results/uninavid_waypoint_points_test/ \
       --policy uninavid_waypoint_points \
       --uninavid-waypoint-points-server-url http://localhost:8002 \
       --headless

   # Or pass a directory (all JSON files inside are iterated)
   python runBench.py \
       --config configs/aliengoh1_test.yaml \
       --scene-root $OMNINAV_SCENE_ROOT \
       --envset /path/to/dataset/dog \
       --output results/uninavid_waypoint_points_test/ \
       --policy uninavid_waypoint_points \
       --uninavid-waypoint-points-server-url http://localhost:8002 \
       --headless

3. Or use it directly from code:

   from bench.policy.uninavid_waypoint import UniNaVidWaypointPointsHTTPPolicy
   policy = UniNaVidWaypointPointsHTTPPolicy(server_url="http://localhost:8002", debug=True)
   policy.reset(instruction="Go to the kitchen")
   action = policy.act(observation)

================================================================================

Coordinate System:
    - Uni-NaVid local frame: x=forward (positive), y=left (positive)
    - OmniNavBench world frame: standard XY plane with yaw rotation

Action Output:
    - Returns Action with extra={"controller": "go_toward_point", "theta_rad": ..., "r_m": ...}
    - EpisodeRunner uses go_toward_point controller to execute local waypoints
"""

from __future__ import annotations

import base64
import io
import os
import time
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
from scipy.spatial.transform import Rotation as R

from bench.policy.base import BasePolicy, Observation, Action


class UniNaVidWaypointPointsHTTPPolicy(BasePolicy):
    """Uni-NaVid Waypoint Points policy that communicates via HTTP with a remote server.

    This policy:
    1. Sends RGB observations to the waypoint prediction server
    2. Receives 5 waypoints in robot-centric local frame (x=forward, y=left)
    3. Uses the first waypoint as a local go_toward_point command
    4. Returns go_toward_point Action for local waypoint following
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8001",
        timeout: float = 60.0,
        max_retries: int = 3,
        wall_timeout_s: Optional[float] = 300.0,
        arrive_threshold: float = 0.5,   # Stop if arrive_prob > threshold
        waypoint_reach_threshold: float = 0.3,  # Distance threshold to consider waypoint reached (meters)
        session_id: Optional[str] = None,
        debug: bool = False,
        debug_dir: str = "debug_waypoint_policy",
        debug_interval: int = 10,
    ):
        """Initialize Uni-NaVid Waypoint HTTP policy.

        Args:
            server_url: Base URL of the waypoint server
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            wall_timeout_s: Wall-clock timeout for a single episode
            arrive_threshold: Threshold for arrive probability to stop
            waypoint_reach_threshold: Distance threshold to consider waypoint reached (meters)
            session_id: Optional session identifier
            debug: Enable debug mode (save images and logs)
            debug_dir: Directory for debug outputs
            debug_interval: Save debug info every N steps
        """
        super().__init__()
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.wall_timeout_s = None if wall_timeout_s is None else float(wall_timeout_s)
        self.arrive_threshold = arrive_threshold
        self.waypoint_reach_threshold = waypoint_reach_threshold
        self.session_id = session_id or f"session_{id(self)}"
        self._policy_step_count = 0
        self._episode_start_monotonic: Optional[float] = None

        # Debug settings
        self._debug = debug
        self._debug_dir = debug_dir
        self._debug_interval = debug_interval
        if self._debug:
            os.makedirs(self._debug_dir, exist_ok=True)
            print(f"[WaypointPolicy] Debug enabled, saving to: {self._debug_dir}")

        # Track last predicted waypoints for yaw verification
        self._last_waypoints = None  # List of [x, y, yaw]
        self._last_robot_yaw = None  # Robot yaw when waypoints were predicted

        # Current waypoint tracking - wait until reached before requesting new one
        self._current_target_world: Optional[Tuple[float, float, float]] = None  # (x, y, z) in world frame
        self._current_action: Optional[Action] = None  # Cached action to return
        self._current_arrive_probs: Optional[List[float]] = None

        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Test connection
        self._check_health()
        print(f"[WaypointPolicy] Connected to server at {self.server_url}")

    def _check_health(self):
        """Check if the server is healthy."""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=10.0)
            response.raise_for_status()
            result = response.json()
            if not result.get("model_loaded", False):
                raise RuntimeError("Server reports model not loaded")
            print(f"[WaypointPolicy] Server health: {result}")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to server at {self.server_url}: {e}")

    def reset(self, instruction: str = ""):
        """Reset policy state for new episode."""
        super().reset(instruction)
        self._policy_step_count = 0
        self._last_waypoints = None
        self._last_robot_yaw = None
        self._episode_start_monotonic = time.monotonic()

        # Reset waypoint tracking
        self._current_target_world = None
        self._current_action = None
        self._current_arrive_probs = None

        try:
            response = self.session.post(
                f"{self.server_url}/reset",
                json={
                    "instruction": instruction,
                    "task_type": "vln",
                    "wall_timeout_s": self.wall_timeout_s,
                },
                timeout=5.0
            )
            response.raise_for_status()
            result = response.json()
            print(f"[WaypointPolicy] Reset: {result.get('status', 'ok')}")
        except Exception as e:
            print(f"[WaypointPolicy] Reset request failed: {e}")
            raise e

    def act(self, observation: Observation) -> Action:
        """Generate waypoint-based action by sending observation to HTTP server.

        Uses go_toward_point controller for local waypoint following.
        Waits until current waypoint is reached before requesting a new one.
        """
        # Update history for context-aware policies
        self.update_history(observation)

        self._policy_step_count += 1

        if self.wall_timeout_s is not None and self._episode_start_monotonic is not None:
            elapsed_s = time.monotonic() - self._episode_start_monotonic
            if elapsed_s >= self.wall_timeout_s:
                print(
                    f"[WaypointPolicy] Step {self._policy_step_count}: "
                    f"wall timeout reached ({elapsed_s:.2f}s >= {self.wall_timeout_s:.2f}s), stopping"
                )
                return Action(action_type="stop", stop=True)

        # Check if we have a current waypoint target and if we've reached it
        if self._current_target_world is not None:
            robot_pos = observation.position
            dx = robot_pos[0] - self._current_target_world[0]
            dy = robot_pos[1] - self._current_target_world[1]
            dist_to_target = np.sqrt(dx * dx + dy * dy)

            if dist_to_target > self.waypoint_reach_threshold:
                # Still moving towards current waypoint, return cached action
                print(
                    f"[WaypointPolicy] Step {self._policy_step_count}: "
                    f"moving to waypoint, dist={dist_to_target:.2f}m (threshold={self.waypoint_reach_threshold:.2f}m)"
                )
                return self._current_action
            else:
                # Reached current waypoint, clear it and request new one
                print(
                    f"[WaypointPolicy] Step {self._policy_step_count}: "
                    f"reached waypoint (dist={dist_to_target:.2f}m), requesting new waypoint"
                )
                self._current_target_world = None
                self._current_action = None

        # Get RGB image - try observation.rgb first, then fall back to cameras list
        rgb = observation.rgb
        if rgb is None:
            # Try to get from cameras list (like OmniNav)
            cameras_list = observation.extra.get("cameras", [])
            for camera_data in cameras_list:
                if isinstance(camera_data, dict) and camera_data.get("name") == "camera":
                    rgb = camera_data.get("rgb")
                    break

        if rgb is None:
            print(f"[WaypointPolicy] Step {self._policy_step_count}: No RGB image available, stopping")
            return Action(action_type="stop", stop=True)

        # Prepare image
        rgb_array = self._prepare_image(rgb)
        image_base64 = self._encode_image(rgb_array)

        # Prepare payload
        payload = {
            "instruction": observation.instruction,
            "image": image_base64,
            "image_shape": list(rgb_array.shape),
        }

        try:
            response = self.session.post(
                f"{self.server_url}/act",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            print(f"[WaypointPolicy] Request failed: {e}")
            return Action(action_type="stop", stop=True)

        if result.get("stop") or result.get("timed_out"):
            reason = result.get("reason", "server requested stop")
            print(f"[WaypointPolicy] Step {self._policy_step_count}: server stop: {reason}")
            return Action(action_type="stop", stop=True)

        # Parse response and convert to action
        return self._parse_waypoint_response(result, observation, rgb_array)

    def _parse_waypoint_response(
        self,
        result: Dict[str, Any],
        observation: Observation,
        rgb_array: np.ndarray
    ) -> Action:
        """Parse server response and create move_along_path action.

        Server returns waypoints in robot-centric local frame:
            - waypoints: List of [x, y, yaw] where x=forward, y=left
            - arrive_probs: List of arrival probabilities for each waypoint

        We use the first two waypoints for move_along_path controller.
        """
        try:
            # Get waypoints from server response
            waypoints = result.get("waypoints", [])
            arrive_probs = result.get("arrive_probs", [])

            if not waypoints:
                print(f"[WaypointPolicy] Step {self._policy_step_count}: no waypoints received, stopping")
                return Action(action_type="stop", stop=True)

            # Check arrive probability - if high enough, stop
            if arrive_probs and arrive_probs[0] > self.arrive_threshold:
                print(f"[WaypointPolicy] Step {self._policy_step_count}: arrive_prob={arrive_probs[0]:.3f} > {self.arrive_threshold}, stopping")
                return Action(action_type="stop", stop=True)

            # Store for debugging
            self._last_waypoints = waypoints
            robot_ori = observation.orientation
            qw, qx, qy, qz = robot_ori
            self._last_robot_yaw = float(R.from_quat([qx, qy, qz, qw]).as_euler("xyz")[2])

            # Convert waypoints to world coordinates
            world_points, local_points = self._waypoints_to_world_path(waypoints, observation)

            if not world_points:
                print(f"[WaypointPolicy] Step {self._policy_step_count}: no valid world points, stopping")
                return Action(action_type="stop", stop=True)

            # Use first two waypoints for move_along_path (or just one if only one available)
            num_points = min(2, len(world_points))
            path_points = world_points[:num_points]

            # Debug output
            if self._debug and self._policy_step_count % self._debug_interval == 0:
                self._save_debug_image(
                    rgb_array, local_points, world_points, arrive_probs, observation
                )

            print(
                f"[WaypointPolicy] Step {self._policy_step_count}: "
                f"NEW path with {num_points} points: {[(round(p[0], 2), round(p[1], 2)) for p in path_points]}"
            )

            # Save last target in world coordinates for reach detection
            self._current_target_world = path_points[-1]  # Last point in path

            # Create and cache the action with move_along_path controller
            self._current_action = Action(
                action_type="waypoint",
                extra={
                    "controller": "move_along_path",
                    "path_points": path_points,
                    "waypoints": waypoints,
                    "world_path": world_points,
                    "arrive_probs": arrive_probs,
                }
            )
            self._current_arrive_probs = arrive_probs

            return self._current_action

        except Exception as e:
            print(f"[WaypointPolicy] Error parsing response: {e}")
            import traceback
            traceback.print_exc()
            return Action(action_type="stop", stop=True)

    def _waypoints_to_world_path(
        self,
        waypoints: List[List[float]],
        observation: Observation,
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float]]]:
        """Convert local waypoints to world path points.

        Uni-NaVid waypoint format:
            - Positions are in incremental form but all in the SAME coordinate frame:
              * All positions are in robot's current orientation frame
              * delta[0]: absolute position relative to robot
              * delta[i] (i>0): increment relative to previous waypoint
              * To get absolute position: cumsum(deltas)
            - Yaws are NOT incremental:
              * Each yaw is relative to robot's CURRENT orientation
            - Coordinate system:
              * Robot frame: x = forward, y = left
              * World frame: x = left/right, y = forward/back
              * Robot's forward (x) maps to world's y axis

        Args:
            waypoints: List of [fwd, left, yaw] in robot's coordinate frame (positions incremental, yaws absolute)
            observation: Current observation with robot pose

        Returns:
            Tuple of (world_path_points, local_points)
        """
        if not waypoints:
            return [], []

        robot_pos = observation.position
        robot_ori = observation.orientation

        # Extract robot's current yaw from quaternion (w, x, y, z)
        qw, qx, qy, qz = robot_ori
        robot_yaw = float(R.from_quat([qx, qy, qz, qw]).as_euler("xyz")[2])

        path_points: List[Tuple[float, float, float]] = []
        local_points: List[Tuple[float, float]] = []

        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)

        # Accumulate positions in robot's coordinate frame
        accumulated_fwd = 0.0
        accumulated_left = 0.0

        for idx, waypoint in enumerate(waypoints):
            if not isinstance(waypoint, (list, tuple)) or len(waypoint) < 3:
                continue

            try:
                # Position delta in robot's coordinate frame
                delta_fwd = float(waypoint[0])   # forward (robot x)
                delta_left = float(waypoint[1])  # left (robot y)

                local_points.append((delta_fwd, delta_left))

                # Accumulate in robot frame
                accumulated_fwd += delta_fwd
                accumulated_left += delta_left

                # Transform from robot frame to world frame
                # Standard 2D rotation: R(yaw) * local_vec + robot_pos
                # Robot frame: fwd = forward, left = left
                # world_x = robot_x + fwd*cos(yaw) - left*sin(yaw)
                # world_y = robot_y + fwd*sin(yaw) + left*cos(yaw)
                world_x = robot_pos[0] + accumulated_fwd * cos_yaw - accumulated_left * sin_yaw
                world_y = robot_pos[1] + accumulated_fwd * sin_yaw + accumulated_left * cos_yaw
                world_z = robot_pos[2]

                path_points.append((float(world_x), float(world_y), float(world_z)))

            except (TypeError, ValueError) as e:
                print(f"[WaypointPolicy] Skip invalid waypoint[{idx}]: {waypoint}")
                continue

        return path_points, local_points

    def _save_debug_image_simple(
        self,
        rgb_array: np.ndarray,
        local_x: float,
        local_y: float,
        theta_rad: float,
        r_m: float,
        arrive_probs: List[float],
        observation: Observation
    ):
        """Save debug image with single waypoint visualization for go_toward_point."""
        if not self._debug:
            return

        try:
            import cv2

            img = rgb_array.copy()
            h, w = img.shape[:2]

            # Robot position (bottom center)
            origin_x, origin_y = w // 2, int(h * 0.9)
            scale = 50  # pixels per meter

            # Draw robot position
            cv2.circle(img, (origin_x, origin_y), 6, (0, 0, 255), -1)

            # Draw target waypoint: x=forward (up), y=left (left)
            img_x = int(origin_x - local_y * scale)
            img_y = int(origin_y - local_x * scale)
            img_x = max(0, min(w-1, img_x))
            img_y = max(0, min(h-1, img_y))

            cv2.line(img, (origin_x, origin_y), (img_x, img_y), (0, 255, 0), 2)
            cv2.circle(img, (img_x, img_y), 5, (0, 255, 0), -1)

            # Add info text
            info = f"Step:{self._policy_step_count} theta={np.rad2deg(theta_rad):.1f}deg r={r_m:.2f}m"
            cv2.putText(img, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Save image
            save_path = os.path.join(self._debug_dir, f"policy_step_{self._policy_step_count:04d}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"[WaypointPolicy] Debug image save failed: {e}")

    def _save_debug_image(
        self,
        rgb_array: np.ndarray,
        local_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float, float]],
        arrive_probs: List[float],
        observation: Observation
    ):
        """Save debug image with waypoints visualization."""
        if not self._debug:
            return

        try:
            import cv2

            img = rgb_array.copy()
            h, w = img.shape[:2]

            # Robot position (bottom center)
            origin_x, origin_y = w // 2, int(h * 0.9)
            scale = 50  # pixels per meter

            # Draw robot position
            cv2.circle(img, (origin_x, origin_y), 6, (0, 0, 255), -1)

            # Colors for waypoints
            colors = [(0, 255, 0), (0, 255, 255), (0, 165, 255), (255, 0, 255), (255, 0, 0)]

            prev_pt = (origin_x, origin_y)
            for i, (lx, ly) in enumerate(local_points):
                # Convert local to image: x=forward (up), y=left (left)
                img_x = int(origin_x - ly * scale)
                img_y = int(origin_y - lx * scale)
                img_x = max(0, min(w-1, img_x))
                img_y = max(0, min(h-1, img_y))

                color = colors[i % len(colors)]
                cv2.line(img, prev_pt, (img_x, img_y), color, 2)
                cv2.circle(img, (img_x, img_y), 5, color, -1)

                label = f"{i+1}"
                if i < len(arrive_probs):
                    label += f"({arrive_probs[i]:.2f})"
                cv2.putText(img, label, (img_x+5, img_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                prev_pt = (img_x, img_y)

            # Add info text
            info = f"Step:{self._policy_step_count} Pos:({observation.position[0]:.1f},{observation.position[1]:.1f})"
            cv2.putText(img, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Save image
            save_path = os.path.join(self._debug_dir, f"policy_step_{self._policy_step_count:04d}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"[WaypointPolicy] Debug image save failed: {e}")

    def _prepare_image(self, rgb: np.ndarray) -> np.ndarray:
        """Prepare image for transmission."""
        if not isinstance(rgb, np.ndarray):
            rgb = np.array(rgb)
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
        if len(rgb.shape) == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        elif len(rgb.shape) == 3 and rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        return rgb

    def _encode_image(self, rgb_array: np.ndarray) -> str:
        """Encode image as base64 string."""
        img = Image.fromarray(rgb_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def close(self):
        """Close the HTTP session."""
        if hasattr(self, 'session'):
            self.session.close()
