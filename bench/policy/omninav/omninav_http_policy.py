"""OmniNav HTTP client policy for OmniNavBench.

OmniNav is a waypoint-based navigation policy that predicts navigation waypoints
based on panoramic vision (left, front, right cameras) and navigation instructions.
"""

from __future__ import annotations

import base64
import json
import numpy as np
import requests
import cv2
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Any, Optional, List
from collections import deque
from scipy.spatial.transform import Rotation as R

from bench.policy.base import BasePolicy, Observation, Action

# OmniNav model parameters (copied from upstream).
INPUT_IMG_SIZE = (640, 569)  # Native OmniNav input size.
HISTORY_RESIZE_RATIO = 1 / 4
MAX_HISTORY_FRAMES = 20
PREDICT_SCALE = 1.0


class OmniNavHTTPPolicy(BasePolicy):
    """OmniNav policy that communicates via HTTP with a remote server.

    OmniNav predicts navigation waypoints based on panoramic RGB images and instructions.
    This policy handles the client-side communication and waypoint-to-action conversion.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8005",
        timeout: float = 30.0,
        max_retries: int = 3,
        save_debug_images: bool = False,
        session_id: Optional[str] = None,
    ):
        super().__init__()
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.save_debug_images = save_debug_images
        self.session_id = session_id or f"session_{id(self)}"
        self._policy_step_count = 0

        # Path following state
        self._current_path = []  # List of waypoints to follow
        self._current_path_index = 0  # Current waypoint index in path

        # Debug image saving
        if self.save_debug_images:
            self.debug_dir = "debug_images_omninav"
            os.makedirs(self.debug_dir, exist_ok=True)
            print(f"[OmniNavHTTPPolicy] Debug images will be saved to: {os.path.abspath(self.debug_dir)}")

        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Test connection
        self._check_health()
        print(f"[OmniNavHTTPPolicy] Connected to server at {self.server_url} (session: {self.session_id})")

    def _check_health(self):
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5.0)
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Cannot connect to OmniNav server at {self.server_url}: {e}")

    def reset(self, instruction: str = ""):
        """Reset policy state for new episode."""
        super().reset(instruction)

        # Reset path following state
        self._current_path = []
        self._current_path_index = 0
        try:
            response = self.session.post(
                f"{self.server_url}/reset",
                json={"session_id": self.session_id, "instruction": instruction},
                timeout=self.timeout
            )
            response.raise_for_status()
        except Exception as e:
            print(f"[OmniNavHTTPPolicy] ⚠️ Reset request failed: {e}")

    def act(self, observation: Observation) -> Action:
        """Generate waypoint-based action by sending observation to HTTP server."""
        if observation.rgb is None:
            return Action(action_type="stop", stop=True)

        self._policy_step_count += 1

        # With the move_along_path controller, the policy no longer tracks per-step
        # path state: the controller consumes all waypoints in order, and EpisodeRunner
        # calls act() again only once they are exhausted.



        # Prepare images - OmniNav needs left, front, right cameras
        # In OmniNavBench, cameras are accessed through observation.extra
        cameras_list = observation.extra.get("cameras", [])

        # Find cameras by name from the cameras list
        def find_camera_rgb(camera_name: str):
            for camera_data in cameras_list:
                if isinstance(camera_data, dict) and camera_data.get("name") == camera_name:
                    return camera_data.get("rgb")
            return None

        left_rgb = find_camera_rgb("left")
        front_rgb = find_camera_rgb("front")
        right_rgb = find_camera_rgb("right")

        # Handle single camera fallback - use the same image for all three views
        if left_rgb is None or front_rgb is None or right_rgb is None:
            camera_names = [cam.get("name") for cam in cameras_list if isinstance(cam, dict)]
            print(f"[OmniNavHTTPPolicy] ⚠️ Missing required cameras. Required: ['left', 'front', 'right'], Available: {camera_names}")

            # Try to use single 'camera' as fallback for all three views
            if len(camera_names) == 1 and camera_names[0] == 'camera':
                fallback_rgb = find_camera_rgb("camera")
                if fallback_rgb is not None:
                    print(f"[OmniNavHTTPPolicy] Using single 'camera' as fallback for all three views (left, front, right)")
                    left_rgb = front_rgb = right_rgb = fallback_rgb
                else:
                    print(f"[OmniNavHTTPPolicy] Fallback camera not found")
                    return Action(stop=True)
            else:
                print(f"[OmniNavHTTPPolicy] Cannot use available cameras as fallback")
                return Action(stop=True)

        # Prepare images for model input
        left_rgb_processed = self._prepare_image(left_rgb)
        front_rgb_processed = self._prepare_image(front_rgb)
        right_rgb_processed = self._prepare_image(right_rgb)

        # Convert pose to habitat format
        current_pose = self._convert_pose_to_habitat(observation.position, observation.orientation)

        # Save debug images
        if self.save_debug_images and self._policy_step_count % 1 == 0:
            self._save_debug_images(left_rgb_processed, front_rgb_processed, right_rgb_processed, observation.step)

        # Prepare payload for server
        payload = {
            "session_id": self.session_id,
            "instruction": observation.instruction,
            "left_image": self._encode_image(left_rgb_processed),
            "front_image": self._encode_image(front_rgb_processed),
            "right_image": self._encode_image(right_rgb_processed),
            "pose": current_pose,
            "step": observation.step,
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
            print(f"[OmniNavHTTPPolicy] ⚠️ Request failed: {e}")
            return Action(stop=True)

        # Parse server response and convert to appropriate action
        return self._parse_response(result, observation)

    def _prepare_image(self, rgb: np.ndarray) -> np.ndarray:
        """Prepare single RGB image for OmniNav model input."""
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

        # Resize to OmniNav input size
        if rgb.shape[0] != INPUT_IMG_SIZE[1] or rgb.shape[1] != INPUT_IMG_SIZE[0]:
            rgb = cv2.resize(rgb, INPUT_IMG_SIZE, interpolation=cv2.INTER_CUBIC)

        return rgb

    def _encode_image(self, rgb_array: np.ndarray) -> str:
        """Encode RGB array to base64 string."""
        from PIL import Image
        import io
        img = Image.fromarray(rgb_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _convert_pose_to_habitat(self, position: tuple, orientation: tuple) -> dict:
        """Convert OmniNav pose format to Habitat format."""
        return {
            'position': list(position),
            'rotation': list(orientation)  # Already in quaternion format
        }

    def _quaternion_to_rotation_matrix(self, quaternion: tuple) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
        w, x, y, z = quaternion

        # Normalize quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm

        # Convert to rotation matrix
        rotation_matrix = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])

        return rotation_matrix

    def _save_debug_images(self, left_rgb, front_rgb, right_rgb, step):
        """Save debug images for visualization."""
        try:
            # Save individual camera images
            for name, img in [("left", left_rgb), ("front", front_rgb), ("right", right_rgb)]:
                save_path = os.path.join(self.debug_dir, f"step_{step}_{name}.png")
                cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # Create and save combined panoramic view
            combined = np.concatenate([left_rgb, front_rgb, right_rgb], axis=1)
            combined_path = os.path.join(self.debug_dir, f"step_{step}_panorama.png")
            cv2.imwrite(combined_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

            print(f"[OmniNavHTTPPolicy] Saved debug images at step {step}")
        except Exception as e:
            print(f"[OmniNavHTTPPolicy] Failed to save debug images: {e}")

    def _parse_response(self, result: Dict[str, Any], observation: Observation) -> Action:
        """Parse server response and return waypoints for WAYPOINT mode."""
        return self._parse_waypoint_response(result, observation)

    def _parse_waypoint_response(self, result: Dict[str, Any], observation: Observation) -> Action:
        """Parse server response and create move_along_path action from first 3 waypoints."""
        try:
            waypoints = result.get("waypoints", [])
            arrive_pred = result.get("arrive_pred", 0.0)
            recover_angles = result.get("recover_angles")
            if recover_angles is None:
                recover_angles = result.get("recover_angle", [])
            if isinstance(recover_angles, (int, float)):
                recover_angles = [recover_angles]
            if recover_angles:
                formatted = ", ".join(f"{float(a):.3f}" for a in recover_angles)
                print(f"[OmniNavHTTPPolicy] Recover angles (rad): [{formatted}]")

            # If arrive prediction > 0.5, consider task complete
            if arrive_pred > 0.5:
                print(f"[OmniNavHTTPPolicy] Arrive prediction > 0.5, stopping")
                return Action(action_type="stop", stop=True)

            # If no waypoints, stop
            if not waypoints or len(waypoints) == 0:
                print(f"[OmniNavHTTPPolicy] No waypoints received, holding position")
                return Action(linear_velocity=0.0, angular_velocity=0.0)

            path_points = self._waypoints_to_world_path(waypoints, observation)
            if not path_points:
                print("[OmniNavHTTPPolicy] No valid path points, holding position")
                return Action(linear_velocity=0.0, angular_velocity=0.0)

            path_points = path_points[:3]
            print(f"[OmniNavHTTPPolicy] Using move_along_path with {len(path_points)} points")

            return Action(
                extra={
                    "controller": "move_along_path",
                    "path_points": path_points,
                    "threshold_m": 0.1,
                }
            )

        except Exception as e:
            print(f"[OmniNavHTTPPolicy] Error parsing waypoint response: {e}")
            return Action(linear_velocity=0.0, angular_velocity=0.0)

    def _waypoints_to_world_path(
        self,
        waypoints: List[Any],
        observation: Observation,
    ) -> List[tuple[float, float, float]]:
        """Convert local (x, z) waypoints to world (x, y, z) path points."""
        if not waypoints:
            return []

        robot_pos = observation.position
        robot_ori = observation.orientation
        # OmniNav local frame: x=right, z=forward. Map to OmniNav XY plane via yaw.
        qw, qx, qy, qz = robot_ori
        yaw = float(R.from_quat([qx, qy, qz, qw]).as_euler("xyz")[2])

        path_points: List[tuple[float, float, float]] = []
        local_points: List[tuple[float, float]] = []
        for idx, waypoint in enumerate(waypoints):
            if not isinstance(waypoint, (list, tuple)) or len(waypoint) < 2:
                print(f"[OmniNavHTTPPolicy] Skip invalid waypoint[{idx}]: {waypoint}")
                continue
            try:
                local_xy = (float(waypoint[0]), float(waypoint[1]))
                local_points.append(local_xy)
                local_waypoint = np.array(
                    [
                        local_xy[0],
                        local_xy[1],
                        0.0,
                    ],
                    dtype=np.float32,
                )
            except (TypeError, ValueError):
                print(f"[OmniNavHTTPPolicy] Skip non-numeric waypoint[{idx}]: {waypoint}")
                continue

            dx_local = local_waypoint[1]
            dy_local = -local_waypoint[0]
            world_x = robot_pos[0] + (np.cos(yaw) * dx_local - np.sin(yaw) * dy_local)
            world_y = robot_pos[1] + (np.sin(yaw) * dx_local + np.cos(yaw) * dy_local)
            world_z = robot_pos[2]
            path_points.append((float(world_x), float(world_y), float(world_z)))

        if path_points:
            local_formatted = ", ".join(
                f"({pt[0]:.3f}, {pt[1]:.3f})" for pt in local_points
            )
            world_formatted = ", ".join(
                f"({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})" for pt in path_points
            )
            print(f"[OmniNavHTTPPolicy] Path points (local): {local_formatted}")
            print(f"[OmniNavHTTPPolicy] Path points (world): {world_formatted}")

        return path_points

    def _parse_action_response(self, result: Dict[str, Any], observation: Observation) -> Action:
        """Parse server response in ACTION mode - convert model waypoints to discrete STEP_ACTION."""
        try:
            # In action mode, the server still returns waypoints, we need to convert them to discrete actions
            waypoints = result.get("action", [])  # Server returns waypoints as "action" in action mode
            arrive_pred = result.get("arrive_pred", 0.0)
            recover_angles = result.get("recover_angle", [])

            print(f"[OmniNavHTTPPolicy] Action mode waypoints: waypoints={waypoints}, arrive_pred={arrive_pred}")

            # If arrive prediction indicates task completion, stop
            if arrive_pred > 0.5:
                print(f"[OmniNavHTTPPolicy] Arrive prediction > 0.5, stopping")
                return Action(action_type="stop", stop=True)

            # If no waypoints, stop
            if not waypoints or len(waypoints) == 0:
                print(f"[OmniNavHTTPPolicy] Empty waypoints, stopping")
                return Action(action_type="stop", stop=True)

            # Use the first waypoint and convert it to a discrete action
            # Based on the waypoint coordinates, determine the action
            first_waypoint = waypoints[0]
            if isinstance(first_waypoint, list) and len(first_waypoint) >= 2:
                waypoint_x, waypoint_z = first_waypoint[0], first_waypoint[1]
            elif hasattr(first_waypoint, '__iter__') and len(first_waypoint) >= 2:
                waypoint_x, waypoint_z = first_waypoint[0], first_waypoint[1]
            else:
                print(f"[OmniNavHTTPPolicy] Invalid waypoint format: {first_waypoint}, using stop")
                return Action(stop=True)

            print(f"[OmniNavHTTPPolicy] First waypoint: x={waypoint_x:.4f}, z={waypoint_z:.4f}")

            # Convert waypoint to discrete action based on direction and magnitude
            distance = (waypoint_x ** 2 + waypoint_z ** 2) ** 0.5

            # If waypoint is very close, consider it as stop
            if distance < 0.01:  # Very small movement threshold
                print(f"[OmniNavHTTPPolicy] Waypoint too close (dist={distance:.4f}), using stop")
                return Action(action_type="stop", stop=True)

            # Calculate angle of the waypoint relative to forward direction
            angle = np.arctan2(waypoint_x, waypoint_z)  # atan2(x, z) gives angle from z-axis (forward)
            angle_deg = np.degrees(angle)

            print(f"[OmniNavHTTPPolicy] Waypoint angle: {angle_deg:.1f}°, distance: {distance:.4f}")

            # Determine discrete action based on angle and distance
            # Forward: -45° to 45°
            # Left: 45° to 135°
            # Right: -135° to -45°
            # Stop: very small distance or outside angle ranges

            if -45 <= angle_deg <= 45:
                action_name = "forward"
            elif 45 < angle_deg <= 135:
                action_name = "left"
            elif -135 <= angle_deg < -45:
                action_name = "right"
            else:
                # For angles outside the main directions, still try to go forward
                action_name = "forward"

            print(f"[OmniNavHTTPPolicy] Converted to discrete action: {action_name}")

            if action_name == "stop":
                return Action(action_type="stop", stop=True)
            elif action_name == "forward":
                return Action(action_type="forward")
            elif action_name == "left":
                return Action(action_type="left")
            elif action_name == "right":
                return Action(action_type="right")
            else:
                print(f"[OmniNavHTTPPolicy] Unknown action: {action_name}, using stop")
                return Action(action_type="stop", stop=True)

        except Exception as e:
            print(f"[OmniNavHTTPPolicy] Error parsing action response: {e}")
            return Action(action_type="stop", stop=True)

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'session'):
            self.session.close()
