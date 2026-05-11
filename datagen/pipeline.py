from __future__ import annotations

from dataclasses import dataclass
import json
import random
from shutil import rmtree
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from datagen.config import PipelineConfig
from datagen.core.geometry import NavMeshGeometry
from datagen.core.registry import ObjectRegistry
from datagen.core.room_zone import RoomZoning
from datagen.generation.blueprint import ChainSampler, TaskChain, VirtualHumansContext
from datagen.generation.instruction import InstructionContext, TemplateGenerator, VLMGenerator
from datagen.generation.trajectory import build_reference_waypoints
from datagen.io.object_annotations import load_object_annotations
from OmniNavExt.envset.recording import build_recording_payload
from OmniNavExt.envset.waypoint_recording import WaypointRecorder

@dataclass
class GeneratedEpisode:
    episode_id: str
    instruction: str
    output_dir: Path
    gt_path: List[Dict[str, Any]]
    recording: Dict[str, Any]
    metadata: Dict[str, Any]


def _log_info(msg: str) -> None:
    try:
        import carb  # type: ignore

        carb.log_info(msg)
    except Exception:
        print(msg)


def _log_warn(msg: str) -> None:
    try:
        import carb  # type: ignore

        carb.log_warn(msg)
    except Exception:
        print(f"[WARN] {msg}")


def _reset_dir(path: Path) -> None:
    if path.exists():
        rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _pose_to_payload(pose: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(pose, dict):
        return None
    pos = pose.get("position")
    yaw_deg = pose.get("orientation_deg")
    if not isinstance(pos, (list, tuple)) or len(pos) < 3 or yaw_deg is None:
        return None
    return {
        "position": [float(pos[0]), float(pos[1]), float(pos[2])],
        "orientation_deg": float(yaw_deg),
    }


def _target_to_payload(target_obj: Any) -> Optional[Dict[str, Any]]:
    if target_obj is None:
        return None
    pos = getattr(target_obj, "position", None)
    return {
        "object_id": getattr(target_obj, "object_id", None),
        "category": getattr(target_obj, "category", None),
        "prim_path": getattr(target_obj, "prim_path", None),
        "position": (
            pos.tolist()
            if hasattr(pos, "tolist")
            else ([float(pos[0]), float(pos[1]), float(pos[2])] if isinstance(pos, (list, tuple)) and len(pos) >= 3 else None)
        ),
        "room_id": getattr(target_obj, "room_id", None),
    }


def _build_standard_video_outputs(
    episode_dir: Path,
    camera_names: List[str],
    *,
    save_depth: bool,
) -> Dict[str, Dict[str, Optional[Path]]]:
    video_dir = episode_dir / "video"
    outputs: Dict[str, Dict[str, Optional[Path]]] = {}
    if len(camera_names) == 1:
        outputs[camera_names[0]] = {
            "rgb": video_dir / "rgb.mp4",
            "depth": (video_dir / "depth.mp4" if save_depth else None),
        }
        return outputs

    for camera_name in camera_names:
        camera_dir = video_dir / str(camera_name)
        outputs[str(camera_name)] = {
            "rgb": camera_dir / "rgb.mp4",
            "depth": (camera_dir / "depth.mp4" if save_depth else None),
        }
    return outputs


def _angle_diff_rad(target: float, current: float) -> float:
    return float((float(target) - float(current) + np.pi) % (2.0 * np.pi) - np.pi)


def _yaw_to_quat_wxyz(yaw_rad: float) -> np.ndarray:
    half = 0.5 * float(yaw_rad)
    return np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def _write_rgb_keyframe(path: Path, rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write keyframe image: {path}")


def _extract_robot_spawn_pose(scenario: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    robots = scenario.get("robots") if isinstance(scenario, dict) else None
    entries = robots.get("entries") if isinstance(robots, dict) else None
    if not isinstance(entries, list) or not entries:
        return None
    first = entries[0]
    if not isinstance(first, dict):
        return None
    initial_pose = first.get("initial_pose")
    return initial_pose if isinstance(initial_pose, dict) else None


def _extract_episode_initial_pose(scenario: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    datagen_cfg = scenario.get("datagen") if isinstance(scenario, dict) else None
    if isinstance(datagen_cfg, dict):
        pose = datagen_cfg.get("episode_start_pose")
        if isinstance(pose, dict):
            return pose
    return None


class DataGenerationPipeline:
    """
    Orchestrates the entire data generation process.
    Follows a strict tree-like execution path:
    Bootstrap -> Cache -> Sample -> Execute -> Generate -> Export
    """

    def __init__(self, config: PipelineConfig):
        self._cfg = config

        self._geometry: Optional[NavMeshGeometry] = None
        self._registry: Optional[ObjectRegistry] = None
        self._room_zoning: Optional[RoomZoning] = None
        self._sampler: Optional[ChainSampler] = None
        self._start_pool: List[np.ndarray] = []

    def run_with_runner(self, runner: Any, scenario: Dict[str, Any]) -> List[GeneratedEpisode]:
        """Run datagen using an already-initialized SimulatorRunner.

        This is the preferred mode (reuses runReplay/standalone initialization).
        """
        print("[datagen.pipeline] run_with_runner(): entered", flush=True)
        stage = getattr(runner, "_stage", None)
        if stage is None:
            raise RuntimeError("Runner has no stage; ensure runner.reset() was called")
        print("[datagen.pipeline] runner stage resolved", flush=True)

        # NavMesh interface is only available after SimulationApp init + navmesh bake.
        try:
            import omni.anim.navigation.core as nav  # type: ignore
        except Exception as exc:
            raise RuntimeError("Failed to import omni.anim.navigation.core; must run inside Isaac Sim") from exc
        print("[datagen.pipeline] imported omni.anim.navigation.core", flush=True)

        nav_interface = nav.acquire_interface()
        print("[datagen.pipeline] acquired nav interface", flush=True)

        _log_info("[DataGen] Phase 1/3: Build caches")
        print("[datagen.pipeline] creating NavMeshGeometry", flush=True)
        self._geometry = NavMeshGeometry(nav_interface)
        print("[datagen.pipeline] creating ObjectRegistry", flush=True)
        self._registry = ObjectRegistry(stage, self._cfg.robot)
        external_objects = None
        if self._cfg.object_annotations_path is not None:
            print(f"[datagen.pipeline] loading object annotations: {self._cfg.object_annotations_path}", flush=True)
            external_objects = load_object_annotations(objects_path=self._cfg.object_annotations_path)
            _log_info(
                f"[DataGen] Loaded {len(external_objects)} external objects from "
                f"{self._cfg.object_annotations_path}"
            )
            print(f"[datagen.pipeline] loaded {len(external_objects)} object annotations", flush=True)
        print("[datagen.pipeline] building object registry", flush=True)
        self._registry.build(external_objects=external_objects)
        print(f"[datagen.pipeline] object registry built: {len(self._registry.query())} objects", flush=True)

        # Fail-fast: if tasks need semantic targets, an empty registry means the scene is not labeled correctly.
        task_types_norm = {str(t).strip().lower() for t in (self._cfg.task.types or []) if t}
        requires_objects = any(t in {"vln", "objectnav", "objnav", "eqa"} for t in task_types_norm)
        if requires_objects and not self._registry.query():
            raise RuntimeError(
                "ObjectRegistry is empty (no semantic targets found). "
                "Ensure scene assets have semantic class labels readable by ObjectRegistry."
            )

        print("[datagen.pipeline] creating RoomZoning", flush=True)
        self._room_zoning = RoomZoning(self._geometry, self._registry)
        print("[datagen.pipeline] computing room zones", flush=True)
        self._room_zoning.compute_zones()
        print("[datagen.pipeline] room zones computed", flush=True)

        _log_info("[DataGen] Phase 2/3: Initialize samplers")
        print("[datagen.pipeline] setting nav random seed", flush=True)
        self._geometry.set_random_seed(int(self._cfg.random_seed))
        print("[datagen.pipeline] nav random seed set", flush=True)

        rng = random.Random(int(self._cfg.random_seed))
        print("[datagen.pipeline] creating ChainSampler", flush=True)
        self._sampler = ChainSampler(
            self._registry,
            self._geometry,
            rng=rng,
            follow_cfg=self._cfg.follow,
        )
        print("[datagen.pipeline] ChainSampler created", flush=True)

        if self._cfg.task.random_start:
            print("[datagen.pipeline] building random start pool", flush=True)
            _rs_units_in_meters = float((scenario.get("scene") or {}).get("units_in_meters") or 1.0)
            _rs_units_in_meters = _rs_units_in_meters if _rs_units_in_meters > 0 else 1.0
            if self._cfg.navmesh.min_clearance_m is None or self._cfg.navmesh.min_clearance_m <= 0:
                raise RuntimeError(
                    "NavMeshConfig.min_clearance_m is required for random_start (must be > 0). "
                    "Set spec.navmesh.min_clearance_m in your datagen spec."
                )
            _rs_min_clearance_m = self._cfg.navmesh.min_clearance_m
            self._start_pool = self._sampler.build_start_pool(
                num_episodes=int(self._cfg.task.num_episodes),
                min_clearance=_rs_min_clearance_m / _rs_units_in_meters,
                min_dist=float(self._cfg.task.start_min_dist_m) / _rs_units_in_meters,
                task_types=list(self._cfg.task.types),
                object_start_min_dist=float(self._cfg.task.object_start_min_dist_m) / _rs_units_in_meters,
            )
            _log_info(f"[DataGen] Random start pool built: {len(self._start_pool)} positions")
            print(f"[datagen.pipeline] random start pool built: {len(self._start_pool)}", flush=True)

        _log_info("[DataGen] Phase 3/3: Generate episodes")
        print("[datagen.pipeline] importing exporter and validator", flush=True)
        from datagen.export.exporter import EpisodeExporter
        from datagen.validation.validators import EpisodeValidator

        exporter = EpisodeExporter()
        validator = EpisodeValidator()
        print("[datagen.pipeline] exporter and validator ready", flush=True)
        room_zone_payload = self._room_zoning.as_room_zone_payload() if self._room_zoning is not None else None

        episodes: List[GeneratedEpisode] = []
        failed_episode_ids: List[int] = []
        for ep_idx in range(int(self._cfg.task.num_episodes)):
            # DEBUG MODE: Single attempt, fail fast.
            max_attempts = 1
            last_error: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    episode = self._generate_episode(ep_idx, runner=runner, scenario=scenario)
                    result = validator.validate(episode)
                    if not result.ok:
                        _log_warn(f"[DataGen] Ep {ep_idx} rejected: reasons={list(result.reasons)}")
                        # Force fail in debug mode
                        raise RuntimeError(f"Validation failed: {list(result.reasons)}")
                    exporter.export(episode=episode, template_scenario=scenario, room_zone=room_zone_payload)
                    self._update_attempt_payload(
                        ep_idx,
                        {
                            "status": "succeeded",
                            "validation_ok": True,
                            "instruction": episode.instruction,
                            "metadata": episode.metadata,
                        },
                    )
                    episodes.append(episode)
                    break
                except Exception as exc:
                    last_error = exc
                    _log_warn(f"[DataGen] Ep {ep_idx} failed: {exc}")
                    self._update_attempt_payload(
                        ep_idx,
                        {
                            "status": "failed",
                            "error": str(exc),
                            "attempt": int(attempt),
                        },
                    )
                    failed_episode_ids.append(int(ep_idx))
                    break
            else:
                failed_episode_ids.append(int(ep_idx))
                self._update_attempt_payload(
                    ep_idx,
                    {
                        "status": "failed",
                        "error": str(last_error) if last_error is not None else "unknown_error",
                    },
                )
        if failed_episode_ids:
            _log_warn(
                f"[DataGen] Completed with failed episodes={failed_episode_ids}; "
                "see episode_xxxxxx/attempt.json for sampled start poses and errors."
            )
        return episodes

    def _attempt_json_path(self, episode_idx: int) -> Path:
        return self._cfg.output_dir / f"episode_{episode_idx:06d}" / "attempt.json"

    def _update_attempt_payload(self, episode_idx: int, updates: Dict[str, Any]) -> None:
        path = self._attempt_json_path(episode_idx)
        payload = _read_json(path)
        payload.update(updates)
        _write_json(path, payload)

    def _generate_episode(self, episode_idx: int, runner: Any, scenario: Dict[str, Any]) -> GeneratedEpisode:
        if self._sampler is None:
            raise RuntimeError("Pipeline not initialized; call run_with_runner first")

        from bench.replay.replay_runner import ReplayCameraOutput, ReplayConfig, ReplayRunner, TrajectoryReplayer
        from bench.utils.visualizer import MultiCameraAsyncVideoWriter

        # B) Resolve robot/controller (capture cameras are prepared per-node below)
        task = list(getattr(runner, "current_tasks", {}).values())[0]
        robot_name = list(getattr(task, "robots", {}).keys())[0]
        robot = task.robots[robot_name]

        controller = robot.controllers.get("move_by_speed")
        if controller is None:
            raise RuntimeError("Robot missing move_by_speed controller; required for replay-style tracking")
        move_along_path = robot.controllers.get("move_along_path")
        rotate_controller = robot.controllers.get("rotate")

        forward_scale, rot_scale = ReplayRunner._extract_controller_scales(controller)
        action_builder = ReplayRunner._action_builder(controller)

        # Optional: attach an extra camera stream for instance-id segmentation visibility checks.
        seg_cam = None
        camera_sensor = None
        if hasattr(robot, "sensors"):
            try:
                sensor_map = getattr(robot, "sensors")
                if isinstance(sensor_map, dict):
                    if "camera" in sensor_map:
                        camera_sensor = sensor_map["camera"]
                    elif "front" in sensor_map:
                        camera_sensor = sensor_map["front"]
            except Exception as exc:
                raise RuntimeError("Failed to resolve robot camera sensor") from exc

        if self._cfg.min_pixels_visible and camera_sensor is None:
            raise RuntimeError("Visibility check requires robot sensor named 'camera' or 'front'")

        if self._cfg.min_pixels_visible and camera_sensor is not None:
            try:
                from OmniNav.core.sensor.isaacsim.camera import IsaacsimCamera

                cam_prim = getattr(camera_sensor, "camera_prim_path", None)
                cam_res = getattr(camera_sensor, "resolution", None)
                if cam_prim and cam_res:
                    seg_cam = IsaacsimCamera(
                        name="seg_camera",
                        prim_path=str(cam_prim),
                        rgba=False,
                        semantic_segmentation=False,
                        instance_segmentation=False,
                        instance_id_segmentation=True,
                        resolution=tuple(cam_res),
                    )
                else:
                    raise RuntimeError("Camera sensor missing camera_prim_path/resolution; cannot enable segmentation")
            except Exception as exc:
                raise RuntimeError("Failed to create IsaacsimCamera for instance_id_segmentation") from exc

        dt = float(getattr(runner, "dt", getattr(runner.config.simulator, "physics_dt", 1.0 / 60.0)))
        scenario_id = str(scenario.get("id", f"episode_{episode_idx}"))
        scenario_spawn_pose = _extract_robot_spawn_pose(scenario)
        scenario_initial_pose = _extract_episode_initial_pose(scenario)

        if self._cfg.task.random_start:
            if not self._start_pool:
                raise RuntimeError(
                    f"random_start=True but start pool is empty at episode {episode_idx}. "
                    "Ensure build_start_pool ran successfully in run_with_runner."
                )
            configured_start_pos = self._start_pool.pop(0)
            configured_yaw_deg = None  # derived from first path segment below
        else:
            if scenario_initial_pose is None:
                raise RuntimeError("Missing explicit episode start pose; spec.robot.initial_pose is required")
            configured_start_pos = (
                np.asarray(scenario_initial_pose.get("position"), dtype=np.float32)
                if scenario_initial_pose
                and isinstance(scenario_initial_pose.get("position"), (list, tuple))
                and len(scenario_initial_pose.get("position")) >= 3
                else None
            )
            configured_yaw_deg = (
                float(scenario_initial_pose.get("orientation_deg"))
                if scenario_initial_pose and scenario_initial_pose.get("orientation_deg") is not None
                else None
            )
            if configured_start_pos is None:
                raise RuntimeError("Episode start pose must include position=[x, y, z]")
            if configured_yaw_deg is None:
                raise RuntimeError("Episode start pose must include orientation_deg")
        spawn_start_pos = (
            np.asarray(scenario_spawn_pose.get("position"), dtype=np.float32)
            if scenario_spawn_pose
            and isinstance(scenario_spawn_pose.get("position"), (list, tuple))
            and len(scenario_spawn_pose.get("position")) >= 3
            else None
        )
        spawn_yaw_deg = (
            float(scenario_spawn_pose.get("orientation_deg"))
            if scenario_spawn_pose and scenario_spawn_pose.get("orientation_deg") is not None
            else None
        )

        replay_cfg = ReplayConfig(
            uninav_config=Path("."),  # unused by _replay_track
            envset_path=Path("."),  # unused by _replay_track
            output_dir=self._cfg.output_dir,
            headless=True,
            fps=float(self._cfg.capture.fps),
            output_video=False,
            save_depth=True,
            num_cameras=1,
            track_kp_pos=float(self._cfg.capture.track_kp_pos),
            track_kp_yaw=float(self._cfg.capture.track_kp_yaw),
            track_lookahead_steps=int(self._cfg.capture.track_lookahead_steps),
            track_smoothing_alpha=float(self._cfg.capture.track_smoothing_alpha),
        )
        replay = ReplayRunner(replay_cfg)

        episode_dir = self._cfg.output_dir / f"episode_{episode_idx:06d}"
        _reset_dir(episode_dir)
        attempt_payload: Dict[str, Any] = {
            "episode_id": str(episode_idx),
            "scenario_id": str(scenario_id),
            "status": "initializing",
            "task_types": list(self._cfg.task.types),
            "final_goal_threshold_m": float(self._cfg.task.final_goal_threshold_m),
            "spawn_pose": _pose_to_payload(scenario_spawn_pose),
            "requested_start_pose": _pose_to_payload(scenario_initial_pose),
        }
        _write_json(episode_dir / "attempt.json", attempt_payload)

        captured_gt_path: List[Dict[str, Any]] = []
        capture_frame_offset = 0
        sim_step_offset = 0

        from datagen.validation.visibility import count_visible_pixels_by_category, count_visible_pixels_instance_id
        from datagen.validation.follow import (
            compute_follow_band_stats,
            compute_personal_space_stats,
            detect_stop_events_from_stop_flags,
        )
        from datagen.generation.follow import (
            build_reference_waypoints_from_timed_poses,
            compute_follow_robot_positions,
            compute_yaws_from_positions,
            parse_route_segments,
            simulate_route_positions,
            simulate_route_timeline,
        )

        # C) Execute chain
        instruction: Optional[str] = None
        last_target: Optional[Dict[str, Any]] = None
        follow_payload: Optional[Dict[str, Any]] = None
        eqa_payload: Optional[Dict[str, Any]] = None
        vln_payload: Optional[Dict[str, Any]] = None
        start_pose_payload: Optional[Dict[str, Any]] = None

        units_in_meters = float((scenario.get("scene") or {}).get("units_in_meters") or 1.0)
        units_in_meters = units_in_meters if units_in_meters > 0 else 1.0

        def _m_to_units(val_m: float) -> float:
            return float(val_m) / float(units_in_meters)

        min_clearance_units = None
        grid_size_m = float(self._cfg.task.grid_size_m)
        if grid_size_m <= 0:
            raise RuntimeError("grid_size_m must be > 0")
        grid_size_units = _m_to_units(grid_size_m)

        pointnav_steps = max(1, int(self._cfg.task.pointnav_steps))
        pointnav_step_min = _m_to_units(float(self._cfg.task.pointnav_step_min_m))
        pointnav_step_max = _m_to_units(float(self._cfg.task.pointnav_step_max_m))
        if pointnav_step_min > pointnav_step_max:
            pointnav_step_min, pointnav_step_max = pointnav_step_max, pointnav_step_min
        pointnav_step_attempts = max(1, int(self._cfg.task.pointnav_step_attempts))
        object_goal_min = _m_to_units(float(self._cfg.task.object_goal_min_m))
        object_goal_max = _m_to_units(float(self._cfg.task.object_goal_max_m))
        object_start_min_dist = _m_to_units(float(self._cfg.task.object_start_min_dist_m))
        final_goal_threshold_units = _m_to_units(float(self._cfg.task.final_goal_threshold_m))

        # A) Blueprinting
        vh_ctx = _build_virtual_humans_context(scenario)
        chain: TaskChain = self._sampler.sample_chain(
            chain_length=int(self._cfg.task.chain_length),
            task_types=list(self._cfg.task.types),
            start_pos=configured_start_pos,
            virtual_humans=vh_ctx,
            min_clearance=min_clearance_units,
            grid_size=grid_size_units,
            pointnav_steps=pointnav_steps,
            pointnav_step_min=pointnav_step_min,
            pointnav_step_max=pointnav_step_max,
            pointnav_step_attempts=pointnav_step_attempts,
            object_goal_min=object_goal_min,
            object_goal_max=object_goal_max,
            object_start_min_dist=object_start_min_dist,
        )
        if chain.nodes:
            first_target_payload = _target_to_payload(chain.nodes[0].target_object)
            if first_target_payload is not None:
                attempt_payload["target"] = first_target_payload
            attempt_payload["chain"] = [
                {
                    "index": int(idx),
                    "task_type": str(node.task_type),
                    "start_pos": [float(node.start_pos[0]), float(node.start_pos[1]), float(node.start_pos[2])],
                    "end_pos": [float(node.end_pos[0]), float(node.end_pos[1]), float(node.end_pos[2])],
                    "target": _target_to_payload(node.target_object),
                }
                for idx, node in enumerate(chain.nodes)
            ]
            _write_json(episode_dir / "attempt.json", attempt_payload)

        # Fail-fast: we must start each episode from a known pose, otherwise tracking/capture are not reliable.
        try:
            _, _, z0_live, _ = ReplayRunner._get_robot_pose_xyyaw(robot)
            first = chain.nodes[0]
            first_path = list(first.path_points or [])
            start_position = np.asarray(configured_start_pos, dtype=np.float32)
            # For random_start the navmesh z is floor-level (~0m), not the robot's
            # physical spawn height. Use spawn_start_pos[2] which is already
            # configured correctly per robot model (e.g. 0.29m for Carter v1).
            if self._cfg.task.random_start:
                if spawn_start_pos is not None and len(spawn_start_pos) >= 3:
                    z0 = float(spawn_start_pos[2])
                else:
                    z0 = float(z0_live)
            elif configured_start_pos is not None and len(configured_start_pos) >= 3:
                z0 = float(configured_start_pos[2])
            elif spawn_start_pos is not None and len(spawn_start_pos) >= 3:
                z0 = float(spawn_start_pos[2])
            elif len(start_position) >= 3:
                z0 = float(start_position[2])
            else:
                z0 = float(z0_live)
            if configured_yaw_deg is not None:
                yaw0 = float(np.deg2rad(configured_yaw_deg))
            elif len(first_path) >= 2:
                dx = float(first_path[1][0] - first_path[0][0])
                dy = float(first_path[1][1] - first_path[0][1])
                yaw0 = float(np.arctan2(dy, dx))
            else:
                yaw0 = float(np.arctan2(first.end_pos[1] - first.start_pos[1], first.end_pos[0] - first.start_pos[0]))
            replay._teleport_robot_xyyaw(
                robot=robot,
                x=float(start_position[0]),
                y=float(start_position[1]),
                yaw_rad=yaw0,
                z_keep=float(z0),
            )
            if hasattr(controller, "forward_command"):
                controller.forward_command = 0.0
            if hasattr(controller, "turn_command"):
                controller.turn_command = 0.0

            start_pose_payload = {
                "position": [float(start_position[0]), float(start_position[1]), float(z0)],
                "orientation_deg": float(np.degrees(yaw0)),
            }
            if spawn_start_pos is not None or spawn_yaw_deg is not None:
                start_pose_payload["spawn_pose"] = {
                    "position": [
                        float(spawn_start_pos[0]) if spawn_start_pos is not None else float(start_position[0]),
                        float(spawn_start_pos[1]) if spawn_start_pos is not None else float(start_position[1]),
                        float(spawn_start_pos[2]) if spawn_start_pos is not None else float(z0),
                    ],
                    "orientation_deg": float(
                        spawn_yaw_deg if spawn_yaw_deg is not None else np.degrees(yaw0)
                    ),
                }
            attempt_payload["status"] = "running"
            attempt_payload["start_pose"] = start_pose_payload
            _write_json(episode_dir / "attempt.json", attempt_payload)

            _log_info("[DataGen] Settling physics for 30 frames...")
            for _ in range(30):
                runner.step(render=False)
            # Flush stale render-product buffers after teleport so frame_000000
            # belongs to the current episode rather than the previous one.
            _log_info("[DataGen] Warming camera buffers for 3 render frames...")
            runner.warm_up(steps=3, render=True, physics=False)

        except Exception as exc:
            raise RuntimeError("Failed to teleport robot to episode start pose") from exc

        available_categories: List[str] = []
        if self._registry is not None:
            try:
                available_categories = sorted({o.category for o in self._registry.query() if o.category})
            except Exception as exc:
                raise RuntimeError("Failed to query registry categories") from exc

        cameras = replay._resolve_replay_cameras(robot)
        if not cameras:
            raise RuntimeError("Replay resolved no cameras for datagen capture")
        camera_output_paths = _build_standard_video_outputs(
            episode_dir,
            [str(cam.name) for cam in cameras],
            save_depth=bool(replay.config.save_depth),
        )
        camera_outputs = {
            name: ReplayCameraOutput(
                rgb=Path(paths["rgb"]),
                depth=(Path(paths["depth"]) if paths["depth"] is not None else None),
            )
            for name, paths in camera_output_paths.items()
        }
        video_writer = MultiCameraAsyncVideoWriter(
            camera_outputs=camera_output_paths,
            fps=int(round(float(self._cfg.capture.fps))),
            recording_json_path=None,
            recording_instruction="",
        )
        primary_camera = cameras[0].camera
        temp_keyframe_root = episode_dir / ".tmp_keyframes"
        _reset_dir(temp_keyframe_root)

        try:
            for node_idx, node in enumerate(chain.nodes):
                _log_info(f"[DataGen] Ep {episode_idx} node {node_idx}: {node.task_type}")

                task_type_norm = str(node.task_type).strip().lower()
                follow_agent_name: Optional[str] = None
                route_stop_events: Optional[List[Dict[str, Any]]] = None

                track_start_capture_frame = int(capture_frame_offset)
                track_start_sim_step = int(sim_step_offset)
                reference_waypoints: List[Dict[str, Any]] = []

                if task_type_norm in {"follow", "follow_human", "followhuman"}:
                    if self._geometry is None:
                        raise RuntimeError("Geometry unavailable for Follow")
                    agent_name_obj = node.follow_human_name or (vh_ctx.names[0] if vh_ctx and vh_ctx.names else None)
                    if not agent_name_obj:
                        raise RuntimeError("Follow task requires at least one virtual human")
                    follow_agent_name = str(agent_name_obj)
                    commands = node.follow_route_commands or (vh_ctx.routes_by_name.get(follow_agent_name) if vh_ctx else None) or []
                    if not commands:
                        raise RuntimeError(f"Follow task missing move_routes commands for agent {follow_agent_name}")

                    # Fail-fast: route execution is part of the Follow ground truth, so command injection must succeed.
                    from OmniNavExt.envset.agent_manager import AgentManager

                    mgr = AgentManager.get_instance()
                    # Wait a few frames for BehaviorScript registration.
                    agent_inst = None
                    for _ in range(120):
                        agent_inst = mgr.get_agent_script_instance_by_name(follow_agent_name)
                        if agent_inst is not None:
                            break
                        runner.step(actions=None, render=True)
                    if agent_inst is None:
                        raise RuntimeError(f"Virtual human not ready for command injection: {follow_agent_name}")
                    mgr.inject_command(follow_agent_name, list(commands), force_inject=True, instant=True)

                    human_start = None
                    if vh_ctx is not None and follow_agent_name in vh_ctx.spawn_by_name:
                        human_start = np.asarray(vh_ctx.spawn_by_name[follow_agent_name], dtype=np.float32)
                    if human_start is None:
                        from OmniNavExt.envset.agent_manager import AgentManager

                        mgr = AgentManager.get_instance()
                        pos = mgr.get_agent_position(follow_agent_name)
                        if pos is not None:
                            human_start = np.asarray([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)
                    if human_start is None:
                        raise RuntimeError("Failed to resolve virtual human start position for Follow")

                    segments = parse_route_segments(commands, agent_name=str(follow_agent_name))
                    speed_units = _m_to_units(float(self._cfg.follow.human_speed_mps))
                    route_timeline = simulate_route_timeline(
                        geometry=self._geometry,
                        start_pos=human_start,
                        segments=segments,
                        dt=dt,
                        speed_units_per_s=speed_units,
                    )
                    human_positions_ref = route_timeline.positions
                    human_yaws_ref = compute_yaws_from_positions(human_positions_ref)

                    _, _, robot_z, _ = ReplayRunner._get_robot_pose_xyyaw(robot)
                    follow_dist_units = _m_to_units(float(self._cfg.follow.target_distance_m))
                    robot_positions_ref = compute_follow_robot_positions(
                        geometry=self._geometry,
                        human_positions=human_positions_ref,
                        human_yaws_rad=human_yaws_ref,
                        follow_distance_units=follow_dist_units,
                        z_keep=float(robot_z),
                    )
                    robot_yaws_ref = compute_yaws_from_positions(robot_positions_ref)

                    reference_waypoints = build_reference_waypoints_from_timed_poses(
                        positions=robot_positions_ref,
                        yaws_rad=robot_yaws_ref,
                        dt=dt,
                        start_frame=track_start_sim_step,
                        stride_frames=int(self._cfg.follow.waypoint_stride_frames),
                    )

                    # Strict stop-events derived from route command boundaries (idle/lookaround).
                    min_stop_steps = max(1, int(round(float(self._cfg.follow.min_stop_event_s) / max(dt, 1e-9))))
                    route_stop_events = detect_stop_events_from_stop_flags(
                        stop_flags=route_timeline.stop_flags,
                        start_frame=track_start_sim_step,
                        dt=dt,
                        min_duration_steps=min_stop_steps,
                        segment_kinds=route_timeline.segment_kinds,
                    )
                else:
                    if move_along_path is None:
                        raise RuntimeError(
                            "Robot missing move_along_path controller; required for PointNav/ObjectNav-style datagen"
                        )
                    if not node.path_points:
                        raise RuntimeError(f"Task node {node_idx} missing path_points")
                    path = [
                        np.asarray([float(p[0]), float(p[1]), 0.0], dtype=np.float32)
                        for p in node.path_points
                    ]
                    if len(path) < 2:
                        raise RuntimeError(f"Task node {node_idx} has insufficient path points")
                    if final_goal_threshold_units <= 0.0:
                        raise RuntimeError("final_goal_threshold_m must be > 0")
                    final_goal_xy = np.asarray([float(path[-1][0]), float(path[-1][1])], dtype=np.float32)
                    nominal_speed = max(0.05, float(forward_scale) * float(self._cfg.capture.nominal_speed_ratio))
                    reference_waypoints = build_reference_waypoints(
                        path_points=path,
                        dt=dt,
                        nominal_speed=nominal_speed,
                        start_frame=track_start_sim_step,
                    )
                if not reference_waypoints:
                    raise RuntimeError(f"Task node {node_idx} produced no reference waypoints")

                track_end_sim_step = int(reference_waypoints[-1]["frame"])
                traj = TrajectoryReplayer(waypoints=reference_waypoints, dt=dt, unit_scale=1.0)
                backend = str(getattr(self._cfg, "instruction_backend", "template") or "template").strip().lower()
                collect_visual_keyframes = backend == "vlm" and task_type_norm == "vln"
                keyframe_paths: List[Path] = []
                keyframe_interval_frames = max(
                    1,
                    int(round(float(self._cfg.capture.keyframe_interval_s) * float(self._cfg.capture.fps))),
                )
                node_keyframe_dir = temp_keyframe_root / f"node_{node_idx:02d}"
                if collect_visual_keyframes:
                    _reset_dir(node_keyframe_dir)

                node_waypoint_recorder = WaypointRecorder(
                    forward_scale=float(forward_scale),
                    rot_scale=float(rot_scale),
                )
                def _record_robot_sample(*, capture_frame: int, sim_step: int) -> None:
                    rec_x, rec_y, rec_z, rec_yaw = ReplayRunner._get_robot_pose_xyyaw(robot)
                    node_waypoint_recorder.add_sample(
                        frame=int(capture_frame),
                        sim_step=int(sim_step),
                        time_s=float(sim_step) * float(dt),
                        xyz=(float(rec_x), float(rec_y), float(rec_z)),
                        yaw_rad=float(rec_yaw),
                    )

                visibility = {"frames_checked": 0, "max_pixels": 0}
                evidence_frames: List[int] = []
                evidence_capture_frames: List[int] = []
                target_prim = getattr(node.target_object, "prim_path", None) if node.target_object else None
                target_cat = getattr(node.target_object, "category", None) if node.target_object else None
                saw_target_instance_id = False

                vln_max_pixels: Dict[str, int] = {}
                vln_evidence_capture_frames: Dict[str, List[int]] = {}
                vln_evidence_frames: Dict[str, List[int]] = {}
                vln_min_pixels = int(self._cfg.min_pixels_visible) if int(self._cfg.min_pixels_visible) > 0 else 100
                vln_stoplist = {
                    "wall",
                    "floor",
                    "ceiling",
                    "background",
                    "unknown",
                    "character",
                    "robot",
                }
                allowed_set = {c for c in available_categories if c} if available_categories else None

                def _on_capture(step_idx: int, frame_idx: int, elapsed_s: float, _camera_obj) -> None:
                    capture_frame = int(track_start_capture_frame) + int(frame_idx)
                    sampled_sim_step = int(track_start_sim_step) + int(step_idx) + 1
                    _record_robot_sample(capture_frame=int(capture_frame), sim_step=int(sampled_sim_step))

                    if (
                        collect_visual_keyframes
                        and _camera_obj is primary_camera
                        and int(frame_idx) % int(keyframe_interval_frames) == 0
                    ):
                        rgba = _camera_obj.get_rgba()
                        if rgba is None:
                            raise RuntimeError("Primary camera get_rgba() returned None during keyframe capture")
                        keyframe_path = node_keyframe_dir / f"frame_{int(frame_idx):06d}.jpg"
                        _write_rgb_keyframe(keyframe_path, rgba[:, :, :3])
                        keyframe_paths.append(keyframe_path)

                    nonlocal saw_target_instance_id
                    if seg_cam is None or not target_prim:
                        return
                    data = seg_cam.get_instance_id_segmentation()

                    # Fail-fast: prim-path instance matching requires idToLabels mapping.
                    if self._cfg.strict_instance_id_prim_path:
                        if not isinstance(data, dict) or not isinstance((data.get("info") or {}).get("idToLabels"), dict):
                            raise RuntimeError("instance_id_segmentation missing info.idToLabels; cannot map prim_path")

                    pixels, _ids = count_visible_pixels_instance_id(
                        data,
                        target_prim_path=str(target_prim),
                        target_category=str(target_cat) if target_cat else None,
                        allow_category_fallback=not bool(self._cfg.strict_instance_id_prim_path),
                    )
                    if _ids:
                        saw_target_instance_id = True
                    visibility["frames_checked"] += 1
                    if int(pixels) > int(visibility["max_pixels"]):
                        visibility["max_pixels"] = int(pixels)
                    if (
                        str(task_type_norm) == "eqa"
                        and int(pixels) >= int(self._cfg.min_pixels_visible)
                        and len(evidence_frames) < int(self._cfg.eqa.max_evidence_frames)
                    ):
                        # Unify evidence output to the physics frame axis.
                        evidence_frames.append(int(capture_frame))
                        evidence_capture_frames.append(int(capture_frame))

                    if str(task_type_norm) == "vln":
                        # Landmark extraction: aggregate visible pixels per category.
                        per_cat = count_visible_pixels_by_category(
                            data, allowed_categories=set(allowed_set) if allowed_set else None
                        )
                        if per_cat:
                            for cat, pix in per_cat.items():
                                if not cat:
                                    continue
                                cat_l = str(cat).lower()
                                if cat_l in vln_stoplist:
                                    continue
                                if target_cat and cat_l == str(target_cat).lower():
                                    continue
                                if int(pix) > int(vln_max_pixels.get(cat, 0)):
                                    vln_max_pixels[cat] = int(pix)
                                if int(pix) >= int(vln_min_pixels):
                                    vln_evidence_capture_frames.setdefault(cat, []).append(int(capture_frame))
                                    vln_evidence_frames.setdefault(cat, []).append(int(capture_frame))

                start_frame_for_track = int(track_start_sim_step)
                actual_track_end_sim_step = int(track_end_sim_step)

                on_step_cb = None
                # FOLLOW: record virtual human traces per physics step (no heuristic idle parsing).
                if task_type_norm in {"follow", "follow_human", "followhuman"}:
                    robot_positions_actual: List[np.ndarray] = []
                    human_positions_actual: List[np.ndarray] = []

                    def _on_step(step_idx: int, elapsed_s: float) -> None:
                        # Robot pose (after stepping) is required for follow validation.
                        rx, ry, rz, _ = ReplayRunner._get_robot_pose_xyyaw(robot)
                        robot_positions_actual.append(np.asarray([rx, ry, rz], dtype=np.float32))

                        from OmniNavExt.envset.agent_manager import AgentManager

                        mgr = AgentManager.get_instance()
                        pos = mgr.get_agent_position(str(follow_agent_name))
                        if pos is None:
                            raise RuntimeError(f"Failed to read virtual human position: {follow_agent_name}")
                        human_positions_actual.append(
                            np.asarray([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)
                        )

                    on_step_cb = _on_step

                if task_type_norm in {"follow", "follow_human", "followhuman"}:
                    replay._replay_track(
                        runner=runner,
                        robot=robot,
                        robot_name=robot_name,
                        controller=controller,
                        action_builder=action_builder,
                        forward_scale=forward_scale,
                        rot_scale=rot_scale,
                        traj=traj,
                        cameras=cameras,
                        camera_outputs=camera_outputs,
                        dt=dt,
                        out_dir=episode_dir,
                        scenario_id=f"{scenario_id}/node_{node_idx}",
                        on_capture=_on_capture,
                        on_step=on_step_cb,
                        video_writer=video_writer,
                    )
                else:
                    fps_interval = 1.0 / float(replay.config.fps)
                    next_capture = 0.0
                    elapsed = 0.0
                    frame_idx = 0
                    last_index = None
                    stall_steps = 0
                    executed_step_idx = -1
                    expected_steps = max(1, int(track_end_sim_step) - int(track_start_sim_step))
                    max_steps = max(
                        int(expected_steps) * replay._MAX_STEPS_MULTIPLIER,
                        int(expected_steps) + replay._MAX_STEPS_BUFFER,
                        replay._MAX_STEPS_MIN,
                    )
                    for step_idx in range(int(max_steps)):
                        actions = [{robot_name: {"move_along_path": [path]}}]
                        runner.step(actions=actions, render=True)

                        capture = elapsed >= next_capture - 1e-9
                        if capture:
                            replay._capture_frame_group(
                                cameras=cameras,
                                camera_outputs=camera_outputs,
                                frame_idx=frame_idx,
                                step_idx=step_idx,
                                elapsed=elapsed,
                                on_capture=_on_capture,
                                video_writer=video_writer,
                                robot=robot,
                            )
                            frame_idx += 1
                            next_capture += fps_interval

                        elapsed += float(dt)
                        executed_step_idx = int(step_idx)
                        actual_track_end_sim_step = int(track_start_sim_step) + int(step_idx) + 1

                        try:
                            obs = move_along_path.get_obs()
                        except Exception as exc:
                            raise RuntimeError("Failed to query move_along_path progress") from exc

                        cur_index = int(obs.get("current_index", -1)) if isinstance(obs, dict) else -1
                        finished = bool(obs.get("finished", False)) if isinstance(obs, dict) else False

                        if last_index is None:
                            last_index = cur_index
                            stall_steps = 0
                        elif cur_index != last_index:
                            last_index = cur_index
                            stall_steps = 0
                        else:
                            stall_steps += 1

                        if step_idx % 500 == 0:
                            rx, ry, _, _ = ReplayRunner._get_robot_pose_xyyaw(robot)
                            _log_info(
                                f"[DataGen] move_along_path step {step_idx}/{max_steps} "
                                f"idx={cur_index} robot_xy=({rx:.2f}, {ry:.2f})"
                            )

                        rx, ry, _, _ = ReplayRunner._get_robot_pose_xyyaw(robot)
                        dist_to_final_goal = float(
                            np.linalg.norm(np.asarray([rx, ry], dtype=np.float32) - final_goal_xy)
                        )
                        if dist_to_final_goal <= float(final_goal_threshold_units):
                            _log_info(
                                f"[DataGen] final goal threshold reached at step={step_idx} "
                                f"frame={actual_track_end_sim_step} dist={dist_to_final_goal:.3f} "
                                f"threshold={float(final_goal_threshold_units):.3f}"
                            )
                            break

                        if finished:
                            _log_info(
                                f"[DataGen] move_along_path finished at step={step_idx} "
                                f"frame={actual_track_end_sim_step}"
                            )
                            break

                        if stall_steps > 1500:
                            raise RuntimeError(
                                f"move_along_path stalled for {stall_steps} steps "
                                f"(index={cur_index})"
                            )
                    else:
                        raise RuntimeError(
                            f"move_along_path did not finish within max_steps={max_steps}"
                        )

                    if node.target_object is not None:
                        if rotate_controller is None:
                            raise RuntimeError(
                                "Robot missing rotate controller; required for final face-target alignment"
                            )
                        target_pos = np.asarray(node.target_object.position, dtype=np.float32)
                        rx, ry, _, yaw_now = ReplayRunner._get_robot_pose_xyyaw(robot)
                        face_vec_xy = np.asarray(
                            [float(target_pos[0]) - float(rx), float(target_pos[1]) - float(ry)],
                            dtype=np.float32,
                        )
                        if float(np.linalg.norm(face_vec_xy)) > 1e-6:
                            goal_yaw = float(np.arctan2(face_vec_xy[1], face_vec_xy[0]))
                            yaw_err = abs(_angle_diff_rad(goal_yaw, yaw_now))
                            _log_info(
                                f"[DataGen] face target start: yaw_err={yaw_err:.3f} "
                                f"goal_yaw_deg={float(np.degrees(goal_yaw)):.2f}"
                            )
                            goal_orientation = _yaw_to_quat_wxyz(goal_yaw)
                            max_rotate_steps = max(180, int(np.ceil(6.0 / max(dt, 1e-6))))
                            rotate_start_step = int(executed_step_idx) + 1
                            for rotate_step in range(max_rotate_steps):
                                step_count = int(rotate_start_step) + int(rotate_step)
                                actions = [{robot_name: {"rotate": [goal_orientation]}}]
                                runner.step(actions=actions, render=True)

                                capture = elapsed >= next_capture - 1e-9
                                if capture:
                                    replay._capture_frame_group(
                                        cameras=cameras,
                                        camera_outputs=camera_outputs,
                                        frame_idx=frame_idx,
                                        step_idx=step_count,
                                        elapsed=elapsed,
                                        on_capture=_on_capture,
                                        video_writer=video_writer,
                                        robot=robot,
                                    )
                                    frame_idx += 1
                                    next_capture += fps_interval

                                elapsed += float(dt)
                                executed_step_idx = int(step_count)
                                actual_track_end_sim_step = int(track_start_sim_step) + int(step_count) + 1
                                rotate_obs = rotate_controller.get_obs()
                                rotate_finished = (
                                    bool(rotate_obs.get("finished", False))
                                    if isinstance(rotate_obs, dict)
                                    else False
                                )
                                if rotate_finished:
                                    _log_info(
                                        f"[DataGen] face target finished at step={step_count} "
                                        f"frame={actual_track_end_sim_step}"
                                    )
                                    break
                            else:
                                raise RuntimeError(
                                    f"final face-target rotation did not finish within max_steps={max_rotate_steps}"
                                )

                node_gt_path = node_waypoint_recorder.build()
                if not node_gt_path:
                    raise RuntimeError(f"Task node {node_idx} produced no captured recording samples")
                captured_gt_path.extend(node_gt_path)
                last_recorded_waypoint = node_gt_path[-1]
                capture_frame_offset = int(last_recorded_waypoint["frame"]) + 1
                sim_step_offset = int(last_recorded_waypoint.get("sim_step", track_start_sim_step))

                if self._cfg.strict_instance_id_prim_path and self._cfg.min_pixels_visible and target_prim:
                    # Fail-fast: if we never observed any instance id for the target prim, prim-path mapping is broken
                    # (or the target never entered the view across all captures), so resample the episode.
                    if not saw_target_instance_id:
                        raise RuntimeError(f"Target instance id was never observed for prim_path={target_prim}")

                if task_type_norm in {"follow", "follow_human", "followhuman"}:
                    # Compute follow/social validations from actual traces.
                    if len(human_positions_actual) <= 1:
                        raise RuntimeError("Follow node did not record virtual human positions")
                    if len(robot_positions_actual) <= 1:
                        raise RuntimeError("Follow node did not record robot positions")

                    band_min = _m_to_units(float(self._cfg.follow.band_min_m))
                    band_max = _m_to_units(float(self._cfg.follow.band_max_m))
                    band_stats = compute_follow_band_stats(
                        robot_positions=robot_positions_actual,
                        human_positions=human_positions_actual,
                        band_min_units=band_min,
                        band_max_units=band_max,
                    )
                    if float(band_stats.violation_ratio) > float(self._cfg.follow.max_violation_ratio):
                        raise RuntimeError(f"Follow band violation too high: {band_stats}")

                    human_yaws_actual = compute_yaws_from_positions(human_positions_actual)
                    ps = compute_personal_space_stats(
                        robot_positions=robot_positions_actual,
                        human_positions=human_positions_actual,
                        human_yaws_rad=human_yaws_actual,
                        a_units=_m_to_units(float(self._cfg.follow.personal_space_a_m)),
                        b_units=_m_to_units(float(self._cfg.follow.personal_space_b_m)),
                    )
                    if float(ps.violation_ratio) > float(self._cfg.follow.max_personal_space_violation_ratio):
                        raise RuntimeError(f"Personal space violation too high: {ps}")

                    # Primary stop-events used for labeling: strict route-derived boundaries.
                    # Use route_stop_events computed during blueprinting/simulation (aligned to the reference sim-step axis).
                    stop_events = route_stop_events or []
                    if not stop_events:
                        raise RuntimeError("Follow node has no route-derived stop_events (Idle/LookAround segments)")

                    vh_threshold_units = _m_to_units(float(self._cfg.follow.vh_distance_threshold_m))
                    vh_gt_waypoints: List[Dict[str, Any]] = []
                    last_vh = None
                    total_xy = 0.0
                    for step, p in enumerate(human_positions_actual):
                        p = np.asarray(p, dtype=np.float32)
                        if last_vh is not None:
                            dxy = (p - last_vh)[:2]
                            dist_xy = float(np.linalg.norm(dxy))
                            if dist_xy < float(vh_threshold_units):
                                continue
                            total_xy += dist_xy
                        else:
                            dist_xy = 0.0
                        frame = int(start_frame_for_track) + int(step)
                        vh_gt_waypoints.append(
                            {
                                "frame": int(frame),
                                "time_s": float(frame) * float(dt),
                                "xyz": [float(p[0]), float(p[1]), float(p[2])],
                                "distance_xy": float(dist_xy),
                                "distance_total_xy": float(total_xy),
                            }
                        )
                        last_vh = p

                    follow_payload = {
                        "human_name": str(follow_agent_name),
                        "move_routes": [{"name": str(follow_agent_name), "commands": list(commands)}],
                        "vh_gt_waypoints": {str(follow_agent_name): vh_gt_waypoints},
                        "stop_events": {str(follow_agent_name): stop_events},
                        "band_stats": {
                            "total_steps": band_stats.total_steps,
                            "violations": band_stats.violations,
                            "violation_ratio": band_stats.violation_ratio,
                            "min_distance": band_stats.min_distance,
                            "max_distance": band_stats.max_distance,
                            "mean_distance": band_stats.mean_distance,
                        },
                        "personal_space": {
                            "total_steps": ps.total_steps,
                            "violations": ps.violations,
                            "violation_ratio": ps.violation_ratio,
                        },
                    }

                if self._cfg.min_pixels_visible and target_prim:
                    if visibility["frames_checked"] <= 0:
                        raise RuntimeError("Visibility check enabled but no capture frames were checked")
                    if int(visibility["max_pixels"]) < int(self._cfg.min_pixels_visible):
                        raise RuntimeError(
                            f"Target not visible enough: max_pixels={visibility['max_pixels']} < "
                            f"min_pixels_visible={self._cfg.min_pixels_visible}"
                        )

                keyframes = list(keyframe_paths)

                target_room_name: Optional[str] = None
                if self._room_zoning is not None and node.target_object is not None and node.target_object.room_id:
                    target_room_name = self._room_zoning.get_room_name(node.target_object.room_id)

                ctx = InstructionContext(
                    task_type=str(node.task_type),
                    target_category=getattr(node.target_object, "category", None) if node.target_object else None,
                    target_room_name=target_room_name,
                    available_categories=available_categories,
                    vln_landmarks=(),
                    vln_landmark_evidence={},
                )

                if task_type_norm == "eqa":
                    # MVP EQA: question is instruction, answer is category; bind evidence frames to captured indices.
                    q = f"What is the {ctx.target_category or 'object'}?"
                    a = str(ctx.target_category or "")
                    instruction = q
                    eqa_payload = {
                        "question": q,
                        "answer": a,
                        "evidence_frames": [int(i) for i in evidence_frames],
                        "evidence_capture_frames": [int(i) for i in evidence_capture_frames],
                        "target": {
                            "category": getattr(node.target_object, "category", None) if node.target_object else None,
                            "prim_path": getattr(node.target_object, "prim_path", None) if node.target_object else None,
                        },
                    }
                    if self._cfg.min_pixels_visible and not evidence_frames:
                        raise RuntimeError("EQA requires at least one evidence frame but none were collected")
                else:
                    if task_type_norm == "vln":
                        # Choose stable landmarks (seen >=2 times) sorted by max_pixels.
                        candidates: List[str] = []
                        for cat, max_pix in sorted(vln_max_pixels.items(), key=lambda kv: int(kv[1]), reverse=True):
                            cap_frames = vln_evidence_capture_frames.get(cat) or []
                            if len(cap_frames) < 2:
                                continue
                            candidates.append(str(cat))
                            if len(candidates) >= 3:
                                break
                        ctx = InstructionContext(
                            task_type=str(node.task_type),
                            target_category=getattr(node.target_object, "category", None) if node.target_object else None,
                            target_room_name=target_room_name,
                            available_categories=available_categories,
                            vln_landmarks=tuple(candidates),
                            vln_landmark_evidence={k: tuple(v) for k, v in vln_evidence_capture_frames.items()},
                        )

                    if backend == "vlm" and task_type_norm == "vln":
                        generator = VLMGenerator()
                    else:
                        generator = TemplateGenerator()
                    instruction = generator.generate([str(p) for p in keyframes], ctx)

                    if task_type_norm == "vln":
                        vln_payload = {
                            "landmarks": list(ctx.vln_landmarks),
                            "landmarks_evidence": {
                                str(cat): {
                                    "capture_frames": list(vln_evidence_capture_frames.get(cat) or []),
                                    "frames": list(vln_evidence_frames.get(cat) or []),
                                    "max_pixels": int(vln_max_pixels.get(cat, 0)),
                                }
                                for cat in ctx.vln_landmarks
                            },
                            "min_pixels": int(vln_min_pixels),
                        }

                if node.target_object is not None:
                    pos = getattr(node.target_object, "position", None)
                    last_target = {
                        "object_id": getattr(node.target_object, "object_id", None),
                        "category": getattr(node.target_object, "category", None),
                        "prim_path": getattr(node.target_object, "prim_path", None),
                        "position": pos.tolist() if hasattr(pos, "tolist") else (list(pos) if pos is not None else None),
                        "room_id": getattr(node.target_object, "room_id", None),
                        "room_name": target_room_name,
                        "visibility": visibility,
                    }
        finally:
            video_writer.close()
            if temp_keyframe_root.exists():
                rmtree(temp_keyframe_root)
            if seg_cam is not None:
                try:
                    seg_cam.cleanup()
                except Exception as exc:
                    _log_warn(f"[DataGen] seg_cam.cleanup failed: {exc}")

        recording_metadata: Dict[str, Any] = {
            "source": "datagen",
            "distance_threshold_xy": 0.0,
            "task_types": list(self._cfg.task.types),
        }
        if captured_gt_path:
            first_wp = captured_gt_path[0]
            last_wp = captured_gt_path[-1]
            recording_metadata["robot_initial_pose"] = {
                "xyz": list(first_wp.get("xyz") or []),
                "yaw_deg": float(first_wp.get("yaw_deg", 0.0)),
            }
            recording_metadata["robot_final_pose"] = {
                "xyz": list(last_wp.get("xyz") or []),
                "yaw_deg": float(last_wp.get("yaw_deg", 0.0)),
            }
        recording = build_recording_payload(
            instruction=instruction or "",
            gt_path=captured_gt_path,
            metadata=recording_metadata,
        )

        return GeneratedEpisode(
            episode_id=str(episode_idx),
            instruction=instruction or "",
            output_dir=episode_dir,
            gt_path=captured_gt_path,
            recording=recording,
            metadata={
                "scene_id": scenario.get("scene", {}).get("id"),
                "scenario_template_id": scenario_id,
                "task_types": list(self._cfg.task.types),
                "start_pose": start_pose_payload,
                "target": last_target,
                "follow": follow_payload,
                "eqa": eqa_payload,
                "vln": vln_payload,
            },
        )

# Example usage entry point (if run directly)
if __name__ == "__main__":
    pass


def _build_virtual_humans_context(scenario: Dict[str, Any]) -> Optional[VirtualHumansContext]:
    vh = scenario.get("virtual_humans") if isinstance(scenario, dict) else None
    if not isinstance(vh, dict):
        return None
    units_in_meters = float((scenario.get("scene") or {}).get("units_in_meters") or 1.0)
    units_in_meters = units_in_meters if units_in_meters > 0 else 1.0
    names = [str(n) for n in (vh.get("name_sequence") or []) if n]
    routes = vh.get("move_routes") or vh.get("routes") or []
    routes_by_name: Dict[str, List[str]] = {}
    if isinstance(routes, list):
        for entry in routes:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            cmds = entry.get("commands") or []
            if not name:
                continue
            if isinstance(cmds, list):
                routes_by_name[str(name)] = [str(c) for c in cmds if c]

    spawn_by_name: Dict[str, np.ndarray] = {}
    spawns = vh.get("spawn_points") or []
    if isinstance(spawns, list):
        for entry in spawns:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            pos = entry.get("position")
            if not name or not isinstance(pos, (list, tuple)) or len(pos) < 3:
                continue
            spawn_by_name[str(name)] = np.asarray([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)

    if not names and not routes_by_name:
        return None
    if not names:
        names = sorted(routes_by_name.keys())
    return VirtualHumansContext(
        names=tuple(names),
        routes_by_name=routes_by_name,
        spawn_by_name=spawn_by_name,
        units_in_meters=float(units_in_meters),
    )
