"""MTU3D HTTP client policy for OmniNavBench (Isaac/OmniNav).

This policy talks to `bench.policy.mtu3d_server` running in the `mtu3d` conda env.

Execution mode: `ExecutionMode.WAYPOINT` (controller-goal actions).
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

from bench.policy.base import Action, BasePolicy, Observation
from bench.policy.mtu3d.mtu3d_topdown_frontier import (
    FogOfWarFrontier,
    generate_oracle_topdown_map,
    resolve_navmesh_bake_bounds,
)
from bench.policy.mtu3d.robot_config import NUM_CAMERAS


def _quat_to_yaw(quat_wxyz: Tuple[float, float, float, float]) -> float:
    w, x, y, z = quat_wxyz
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _yaw_to_quat_wxyz(yaw_rad: float) -> Tuple[float, float, float, float]:
    half = yaw_rad * 0.5
    return (float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))


def _encode_png_b64(rgb_uint8: np.ndarray) -> str:
    img = Image.fromarray(rgb_uint8, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _encode_npy_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy arrays inside ``obj`` to lists so it is JSON-serializable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    else:
        return obj


@dataclass
class _Frame:
    rgb: np.ndarray  # uint8 (H, W, 3)
    depth_m: np.ndarray  # float32 (H, W), meters
    sensor_pos: Tuple[float, float, float]
    sensor_quat_wxyz: Tuple[float, float, float, float]


@dataclass
class _MultiFrame:
    """Per-step frame snapshot. Original MTU3D uses one camera (`camera_0`); the
    list-shaped fields are kept for forward compatibility with multi-camera setups."""
    rgb_list: List[np.ndarray]  # List of uint8 (H, W, 3) arrays
    depth_list: List[np.ndarray]  # List of float32 (H, W) arrays
    sensor_pos_list: List[Tuple[float, float, float]]  # List of camera positions
    sensor_quat_wxyz_list: List[Tuple[float, float, float, float]]  # List of camera orientations
    camera_names: List[str]  # Camera names for reference
    camera_params_list: List[Optional[Dict[str, Any]]]  # Camera parameters for each camera


class MTU3DHTTPPolicy(BasePolicy):
    """MTU3D policy that requests target 3D positions from an MTU3D server."""

    def __init__(
        self,
        server_url: str = "http://localhost:8010",
        timeout_s: float = 120.0,
        waypoint_threshold_m: float = 0.1,
        rotate_threshold_rad: float = 0.02,
        goto_max_frames: int = 6,
        spin_steps: int = 12,
        topdown_resolution_m: float = 0.05,
        visible_radius_m: float = 3.0,
        frontier_max_candidates: int = 64,
        frontier_min_separation_m: float = 0.5,
    ) -> None:
        super().__init__()
        self.server_url = server_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.waypoint_threshold_m = float(waypoint_threshold_m)
        self.rotate_threshold_rad = float(rotate_threshold_rad)
        # MTU3D keeps up to 6 "goto" frames (subsampled) + 12 spin frames per decision.
        self.goto_max_frames = int(goto_max_frames)
        self.spin_steps = int(spin_steps)
        self.topdown_resolution_m = float(topdown_resolution_m)
        self.visible_radius_m = float(visible_radius_m)
        self.frontier_max_candidates = int(frontier_max_candidates)
        self.frontier_min_separation_m = float(frontier_min_separation_m)

        if self.waypoint_threshold_m <= 0:
            raise ValueError("waypoint_threshold_m must be > 0")
        if self.rotate_threshold_rad <= 0:
            raise ValueError("rotate_threshold_rad must be > 0")
        if self.goto_max_frames <= 0:
            raise ValueError("goto_max_frames must be > 0")
        if self.spin_steps <= 0:
            raise ValueError("spin_steps must be > 0")
        if self.topdown_resolution_m <= 0:
            raise ValueError("topdown_resolution_m must be > 0")
        if self.visible_radius_m <= 0:
            raise ValueError("visible_radius_m must be > 0")
        if self.frontier_max_candidates <= 0:
            raise ValueError("frontier_max_candidates must be > 0")
        if self.frontier_min_separation_m <= 0:
            raise ValueError("frontier_min_separation_m must be > 0")

        self._session = requests.Session()
        self._check_health()

        # Keep two buffers to mirror MTU3D:
        # - goto_frames: frames collected while moving to the last target
        # - spin_frames: frames collected during the 360° scan before decision
        self._goto_frames: List[_MultiFrame] = []
        self._spin_frames: List[_MultiFrame] = []
        self._decision_num: int = 0
        self._phase: str = "spin"  # "spin" -> "decide" -> "goto"
        self._spin_remaining: int = self.spin_steps
        self._spin_target_yaw: Optional[float] = None
        self._spinning_active: bool = False
        self._frontier: Optional[FogOfWarFrontier] = None
        self._visited_frontier: set[Tuple[float, float, float]] = set()
        # Avoid per-frame camera spam: check+log only once per episode.
        self._camera_check_logged: bool = False
        # After we have confirmed camera_0 is producing valid RGB-D, enforce it to avoid silent failures.
        self._camera_check_passed: bool = False

    def _check_health(self) -> None:
        resp = self._session.get(f"{self.server_url}/health", timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"MTU3D server unhealthy: {data}")

    def reset(self, instruction: str = "") -> None:
        super().reset(instruction)
        self._goto_frames.clear()
        self._spin_frames.clear()
        self._decision_num = 0
        self._phase = "spin"
        self._spin_remaining = self.spin_steps
        self._spin_target_yaw = None
        self._spinning_active = False
        self._frontier = None
        self._visited_frontier.clear()
        self._camera_check_logged = False
        self._camera_check_passed = False

        resp = self._session.post(f"{self.server_url}/reset", json={}, timeout=self.timeout_s)
        resp.raise_for_status()

    def observe(self, observation: Observation) -> None:
        """Collect frames continuously; MTU3D consumes up to N frames per decision."""
        super().observe(observation)

        # Lazily initialize oracle top-down map after NavMesh is baked.
        if self._frontier is None:
            bounds = resolve_navmesh_bake_bounds()
            try:
                topdown = generate_oracle_topdown_map(bounds=bounds, resolution_m=self.topdown_resolution_m)
            except Exception as e:
                raise RuntimeError(f"Error generating oracle top-down map: {e}") from e
            self._frontier = FogOfWarFrontier(
                topdown=topdown,
                hfov_deg=42.0,
                max_range_m=self.visible_radius_m,
                rays=121,
            )

        # Update fog-of-war on every frame (continuous perception during motion).
        yaw = _quat_to_yaw(observation.orientation)
        self._frontier.update(position_xyz=observation.position, yaw_rad=yaw)

        cameras_list = observation.extra.get("cameras", [])
        verbose = not self._camera_check_logged
        if verbose:
            print(
                f"[MTU3DHTTPPolicy] Step {getattr(self, '_policy_step_count', 0)}: "
                f"Found {len(cameras_list)} cameras in observation.extra['cameras']"
            )

        if not cameras_list:
            if verbose:
                print("[MTU3DHTTPPolicy] ❌ ERROR: No cameras found in observation.extra['cameras']")
                print(
                    f"[MTU3DHTTPPolicy] observation.extra keys: "
                    f"{list(observation.extra.keys()) if observation.extra else 'None'}"
                )
                self._camera_check_logged = True
            return

        required_cameras = [f"camera_{i}" for i in range(NUM_CAMERAS)]
        cameras_by_name: Dict[str, Dict[str, Any]] = {}
        for camera_data in cameras_list:
            if not isinstance(camera_data, dict):
                continue
            name = camera_data.get("name")
            if isinstance(name, str) and name:
                cameras_by_name[name] = camera_data

        found_cameras: List[Dict[str, Any]] = []
        if verbose:
            print(f"[MTU3DHTTPPolicy] Looking for cameras: {required_cameras}")
        for cam_name in required_cameras:
            cam_data = cameras_by_name.get(cam_name)
            if cam_data is not None:
                found_cameras.append(cam_data)
                if verbose:
                    print(f"[MTU3DHTTPPolicy] ✅ Found camera: {cam_name}")
            else:
                if verbose:
                    print(f"[MTU3DHTTPPolicy] ❌ Missing camera: {cam_name}")

        if not found_cameras:
            if verbose:
                print("[MTU3DHTTPPolicy] No cameras found, cannot proceed")
                self._camera_check_logged = True
            return

        cameras_list = found_cameras
        camera_names = [cam.get("name", "") for cam in cameras_list]
        all_cameras_ok = True
        if verbose:
            print(f"[MTU3DHTTPPolicy] Using cameras: {camera_names}")
            print(f"[MTU3DHTTPPolicy] === Camera Status Check ===")

        if verbose:
            for i, camera_data in enumerate(cameras_list):
                name = camera_data.get("name", f"camera_{i}")
                rgb = camera_data.get("rgb")
                depth = camera_data.get("depth")

                rgb_ok = rgb is not None
                if rgb_ok:
                    if isinstance(rgb, np.ndarray):
                        rgb_shape_ok = rgb.ndim == 3 and rgb.shape[2] == 3
                        rgb_dtype_ok = rgb.dtype == np.uint8
                        rgb_ok = rgb_ok and rgb_shape_ok and rgb_dtype_ok
                        rgb_info = f"shape={rgb.shape}, dtype={rgb.dtype}"
                    else:
                        rgb_ok = False
                        rgb_info = f"type={type(rgb)}"
                else:
                    rgb_info = "None"

                depth_ok = depth is not None
                if depth_ok:
                    if isinstance(depth, np.ndarray):
                        depth_shape_ok = depth.ndim == 2
                        depth_dtype_ok = np.issubdtype(depth.dtype, np.floating)
                        depth_ok = depth_ok and depth_shape_ok and depth_dtype_ok
                        depth_info = f"shape={depth.shape}, dtype={depth.dtype}"
                    else:
                        depth_ok = False
                        depth_info = f"type={type(depth)}"
                else:
                    depth_info = "None"

                camera_ok = rgb_ok and depth_ok
                status_icon = "✅" if camera_ok else "❌"
                all_cameras_ok = all_cameras_ok and camera_ok

                print(f"  {status_icon} {name}: RGB={rgb_ok}({rgb_info}), Depth={depth_ok}({depth_info})")

            if all_cameras_ok:
                print(f"[MTU3DHTTPPolicy] 🎉 SUCCESS: All {len(cameras_list)} cameras are working properly!")
            else:
                print(f"[MTU3DHTTPPolicy] ⚠️ WARNING: Some cameras have issues. Check above details.")

            self._camera_check_logged = True

        # Once we've confirmed camera_0 is producing valid RGB-D, enforce it every frame.
        if cameras_list and all_cameras_ok:
            self._camera_check_passed = True
        elif self._camera_check_passed:
            raise RuntimeError(
                "MTU3D requires camera_0 with valid RGB-D every frame; "
                f"got cameras={len(cameras_list)}"
            )

        multi_frame = self._build_multi_frame(cameras_list, observation)
        if self._spinning_active:
            self._spin_frames.append(multi_frame)
            # Rotate controller takes multiple physics steps; downsample to spin_steps frames.
            if len(self._spin_frames) > self.spin_steps:
                idxs = np.linspace(0, len(self._spin_frames) - 1, self.spin_steps).astype(int).tolist()
                self._spin_frames = [self._spin_frames[i] for i in idxs]
        else:
            self._goto_frames.append(multi_frame)
            # Subsample down to at most goto_max_frames (interval sampling like MTU3D scripts).
            if len(self._goto_frames) > self.goto_max_frames:
                idxs = np.linspace(0, len(self._goto_frames) - 1, self.goto_max_frames).astype(int).tolist()
                self._goto_frames = [self._goto_frames[i] for i in idxs]

    def _build_frame(self, observation: Observation) -> _Frame:
        rgb = observation.rgb
        if not isinstance(rgb, np.ndarray):
            rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8:
            raise ValueError(f"MTU3D requires rgb uint8, got {rgb.dtype}")
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"MTU3D requires rgb shape (H,W,3), got {rgb.shape}")

        depth = observation.depth
        if not isinstance(depth, np.ndarray):
            depth = np.asarray(depth)
        if depth.ndim != 2:
            raise ValueError(f"MTU3D requires depth shape (H,W), got {depth.shape}")
        depth_m = np.asarray(depth, dtype=np.float32)

        # Strict requirement: caller must provide camera pose (position + quaternion).
        cam = observation.extra.get("camera_pose")
        if not isinstance(cam, dict):
            raise ValueError("MTU3D requires observation.extra['camera_pose']={position, orientation_wxyz}")
        pos = cam.get("position")
        quat = cam.get("orientation_wxyz")
        if not (isinstance(pos, (list, tuple)) and len(pos) == 3):
            raise ValueError("camera_pose.position must be [x,y,z]")
        if not (isinstance(quat, (list, tuple)) and len(quat) == 4):
            raise ValueError("camera_pose.orientation_wxyz must be [w,x,y,z]")

        sensor_pos = (float(pos[0]), float(pos[1]), float(pos[2]))
        sensor_quat = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        return _Frame(rgb=rgb, depth_m=depth_m, sensor_pos=sensor_pos, sensor_quat_wxyz=sensor_quat)

    def _build_multi_frame(self, cameras_list: List[Dict[str, Any]], observation: Observation) -> _MultiFrame:
        """Build multi-view frame from all available cameras."""
        rgb_list = []
        depth_list = []
        sensor_pos_list = []
        sensor_quat_list = []
        camera_names = []
        camera_params_list = []

        for camera_data in cameras_list:
            camera_name = camera_data.get("name", "")
            rgb = camera_data.get("rgb")
            depth = camera_data.get("depth")
            cam_pose = camera_data.get("camera_pose")  # Camera pose dict
            cam_params = camera_data.get("camera_params")  # Camera parameters

            # Extract position and orientation from camera_pose
            cam_pos = cam_quat = None
            if cam_pose is not None and isinstance(cam_pose, dict):
                cam_pos = cam_pose.get("position")
                cam_quat = cam_pose.get("orientation_wxyz")

            if rgb is None or depth is None or cam_pos is None or cam_quat is None:
                if not self._camera_check_logged:
                    print(f"[MTU3DHTTPPolicy] Skipping camera {camera_name}: missing data")
                continue

            # Process RGB
            if not isinstance(rgb, np.ndarray):
                rgb = np.asarray(rgb)
            if rgb.dtype != np.uint8:
                raise ValueError(f"MTU3D requires rgb uint8, got {rgb.dtype}")
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"MTU3D requires rgb shape (H,W,3), got {rgb.shape}")

            # Process depth
            if not isinstance(depth, np.ndarray):
                depth = np.asarray(depth)
            if depth.ndim != 2:
                raise ValueError(f"MTU3D requires depth shape (H,W), got {depth.shape}")
            depth_m = np.asarray(depth, dtype=np.float32)

            # Process camera pose
            if not (isinstance(cam_pos, (list, tuple)) and len(cam_pos) == 3):
                raise ValueError(f"Camera '{camera_name}' position must be [x,y,z]")
            if not (isinstance(cam_quat, (list, tuple)) and len(cam_quat) == 4):
                raise ValueError(f"Camera '{camera_name}' orientation_wxyz must be [w,x,y,z]")

            rgb_list.append(rgb)
            depth_list.append(depth_m)
            sensor_pos_list.append((float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])))
            sensor_quat_list.append((float(cam_quat[0]), float(cam_quat[1]), float(cam_quat[2]), float(cam_quat[3])))
            camera_names.append(camera_name)
            camera_params_list.append(cam_params)  # May be None

        if not rgb_list:
            raise ValueError("No valid camera data found")

        return _MultiFrame(
            rgb_list=rgb_list,
            depth_list=depth_list,
            sensor_pos_list=sensor_pos_list,
            sensor_quat_wxyz_list=sensor_quat_list,
            camera_names=camera_names,
            camera_params_list=camera_params_list
        )

    def act(self, observation: Observation) -> Action:
        """State machine: spin -> (frontier/target) decision -> goto.

        With NUM_CAMERAS == 1 (original MTU3D paper pattern) the robot first
        rotates in place for `spin_steps` yaw steps and the single forward
        camera collects 12 views before the first decision. With NUM_CAMERAS
        > 1 (ring-camera mode) the spin phase is skipped because the cameras
        already cover the corresponding angular extent in one frame.
        """
        if NUM_CAMERAS == 1 and self._phase == "spin":
            return self._act_spin(observation)
        return self._act_decide(observation)

    def _act_spin(self, observation: Observation) -> Action:
        """Rotate in-place for 360° scan (MTU3D-style)."""
        if self._spin_remaining <= 0:
            self._phase = "decide"
            self._spinning_active = False
            return self._act_decide(observation)

        yaw_now = _quat_to_yaw(observation.orientation)
        if self._spin_target_yaw is None:
            self._spin_target_yaw = yaw_now

        delta = (2.0 * np.pi) / float(self.spin_steps)
        self._spin_target_yaw = float(self._spin_target_yaw + delta)
        goal_q = _yaw_to_quat_wxyz(self._spin_target_yaw)
        self._spin_remaining -= 1
        self._spinning_active = True

        return Action(
            extra={
                "controller": "rotate",
                "goal_orientation_wxyz": list(goal_q),
                "threshold_rad": self.rotate_threshold_rad,
            }
        )

    def _act_decide(self, observation: Observation) -> Action:
        """Ask MTU3D server for a 3D target position, then output a move_to_point waypoint action."""
        # Original MTU3D feeds the spin frames as the visual context for each decision; the latest
        # goto frame (motion towards the previous target) is appended once any has been collected.
        current_frames: List[_MultiFrame] = list(self._spin_frames)
        if self._goto_frames:
            current_frames.append(self._goto_frames[-1])
        if not current_frames:
            raise RuntimeError("MTU3D has no current frames; ensure policy.observe() is called before act()")
        if self._frontier is None:
            raise RuntimeError("MTU3D frontier system not initialized")

        frontiers = self._frontier.sample_frontiers(
            max_candidates=self.frontier_max_candidates,
            min_separation_m=self.frontier_min_separation_m,
        )
        # Filter visited frontier targets (match MTU3D's rounding-based visitation).
        filtered: List[Tuple[float, float, float]] = []
        for wp in frontiers:
            key = (round(wp[0], 1), round(wp[1], 1), round(wp[2], 1))
            if key in self._visited_frontier:
                continue
            filtered.append(wp)

        frames_payload: List[Dict[str, Any]] = []
        camera_params_list = []
        for multi_fr in current_frames:
            # Emit one frame entry per camera in the multi-view frame.
            for i, (rgb, depth, pos, quat, cam_params) in enumerate(zip(
                multi_fr.rgb_list,
                multi_fr.depth_list,
                multi_fr.sensor_pos_list,
                multi_fr.sensor_quat_wxyz_list,
                multi_fr.camera_params_list
            )):
                frame_entry = {
                    "rgb_b64": _encode_png_b64(rgb),
                    "depth_npy_b64": _encode_npy_b64(depth),
                    "sensor_pos": list(pos),
                    "sensor_quat_wxyz": list(quat),
                }
                if hasattr(multi_fr, 'camera_names') and i < len(multi_fr.camera_names):
                    frame_entry["camera_name"] = multi_fr.camera_names[i]
                frames_payload.append(frame_entry)
                camera_params_list.append(cam_params)

        serializable_camera_params = [_make_json_serializable(params) for params in camera_params_list]
        print(f"[MTU3DHTTPPolicy] Sending {len(serializable_camera_params)} camera params to server")

        payload = {
            "sentence": observation.instruction,
            "decision_num": self._decision_num,
            "frontier_waypoints": [list(wp) for wp in filtered],
            "frames": frames_payload,
            "camera_params_list": serializable_camera_params,
        }
        resp = self._session.post(f"{self.server_url}/decision", json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected MTU3D response type: {type(data)}")

        target = data.get("target_position")
        if not (isinstance(target, list) and len(target) == 3):
            raise RuntimeError(f"MTU3D response missing target_position: {data}")

        is_object_decision = bool(data.get("is_object_decision", False))
        decision_type = data.get("decision_type", "Unknown")
        is_final = bool(data.get("is_final_decision", False))

        target_xyz = (float(target[0]), float(target[1]), float(target[2]))

        print(f"[MTU3DHTTPPolicy] Decision #{self._decision_num}: {decision_type}")
        print(f"  Target: ({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f})")
        print(f"  Is object decision: {is_object_decision}, Is final: {is_final}")
        print(f"  Instruction: '{observation.instruction}'")
        if not is_final:
            self._visited_frontier.add((round(target_xyz[0], 1), round(target_xyz[1], 1), round(target_xyz[2], 1)))

        self._decision_num += 1
        # Frames are intentionally not cleared so subsequent acts reuse the latest multi-view data.

        if is_final:
            return Action(stop=True)

        snapped = self._snap_to_navmesh(target_xyz)
        return Action(
            extra={
                "controller": "move_to_point",
                "goal_position": list(snapped),
                "threshold_m": self.waypoint_threshold_m,
            }
        )

    @staticmethod
    def _snap_to_navmesh(point_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Project a 3D point onto the baked NavMesh (strict)."""
        try:
            import carb  # type: ignore
            import omni.anim.navigation.core as nav  # type: ignore
        except ModuleNotFoundError as e:
            raise RuntimeError("NavMesh snap requires Isaac Sim (omni.anim.navigation.core)") from e

        interface = nav.acquire_interface()
        navmesh = interface.get_navmesh()
        if navmesh is None:
            raise RuntimeError("NavMesh not available; ensure NavMesh baking succeeded before running MTU3D policy")

        p = np.array([float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])], dtype=float)
        result = navmesh.query_closest_point(target=carb.Float3(float(p[0]), float(p[1]), float(p[2])))
        if result is None:
            raise RuntimeError("NavMesh query_closest_point returned None")

        if isinstance(result, (list, tuple)) and len(result) >= 1:
            pos = result[0]
        else:
            pos = result

        # carb.Float3 has x/y/z fields.
        if not (hasattr(pos, "x") and hasattr(pos, "y") and hasattr(pos, "z")):
            raise RuntimeError(f"Unexpected NavMesh closest point type: {type(pos)}")

        return (float(pos.x), float(pos.y), float(pos.z))
