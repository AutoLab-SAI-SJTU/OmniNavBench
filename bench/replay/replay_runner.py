from __future__ import annotations

import math
import json
import time
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from OmniNavExt.envset.config_loader import EnvsetConfigLoader
from OmniNavExt.envset.core.scene_manager import is_matterport_scenario
from OmniNavExt.envset.recording import (
    CANONICAL_RECORDING_KEY,
    build_recording_payload,
    resolve_recording_waypoints,
    write_recording_sidecar,
)
from bench.utils.visualizer import MultiCameraAsyncVideoWriter

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ReplayConfig:
    uninav_config: Path
    envset_path: Path
    output_dir: Path
    scene_root: Optional[Path] = None
    scenario_ids: Optional[List[str]] = None
    headless: bool = True
    fps: float = 30.0
    track_kp_pos: float = 2.0
    track_kp_yaw: float = 3.0
    track_lookahead_steps: int = 10
    track_smoothing_alpha: float = 0.25
    # Periodic correction parameters (low-frequency correction approach)
    track_correction_period: int = 50  # Frames between corrections
    track_correction_frames: int = 10  # Frames for smooth transition
    # Skip completed scenarios
    skip_completed: bool = True  # Skip scenarios with existing output
    skip_min_frames: int = 50  # Minimum frames to consider complete
    # Video output (instead of individual images)
    output_video: bool = True  # Output as MP4 video instead of images
    save_depth: bool = True  # Also save depth video
    num_cameras: int = 1  # 1 -> front/primary only, 3 -> front/left/right


def _angle_diff_rad(a: float, b: float) -> float:
    """Shortest signed angular difference a-b."""
    d = a - b
    return (d + math.pi) % (2 * math.pi) - math.pi


def _quat_wxyz_to_yaw_rad(quat: Tuple[float, float, float, float]) -> float:
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _yaw_rad_to_quat_wxyz(yaw_rad: float) -> Tuple[float, float, float, float]:
    half = yaw_rad / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _min_jerk(t: float) -> float:
    """Minimum-jerk smoothing curve s(t) with s(0)=0, s(1)=1, zero vel/acc at ends."""
    t = _clamp(float(t), 0.0, 1.0)
    t3 = t * t * t
    t4 = t3 * t
    t5 = t4 * t
    return 10.0 * t3 - 15.0 * t4 + 6.0 * t5


def scan_json_files(root: Path) -> List[Path]:
    """Scan root directory for JSON files (root/*.json + root/*/*.json).

    If root is a file, returns [root] if it's a JSON file.
    """
    def _is_envset_json(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() == ".json" and not path.name.endswith(".backup.json")

    root = root.expanduser().resolve()
    if not root.exists():
        return []
    if root.is_file():
        return [root] if _is_envset_json(root) else []
    if not root.is_dir():
        return []

    json_files: List[Path] = []
    json_files.extend(sorted(p for p in root.iterdir() if _is_envset_json(p)))
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        json_files.extend(sorted(p for p in sub.iterdir() if _is_envset_json(p)))
    return json_files


def get_articulation_z(articulation) -> Optional[float]:
    """Get current Z position from articulation, handling both pose APIs.

    Returns None only if both API calls fail, with errors logged.
    """
    first_error = None
    try:
        pos, _ = articulation.get_world_pose()
        return float(pos[2])
    except Exception as e:
        first_error = e

    try:
        pos_arr, _ = articulation.get_world_poses()
        return float(pos_arr[0][2])
    except Exception as e:
        # Log both errors for debugging
        print(f"[get_articulation_z] Failed to get Z position. "
              f"get_world_pose error: {first_error}, get_world_poses error: {e}", flush=True)
        return None


@dataclass(frozen=True)
class ScenarioRef:
    source_envset_path: Path
    scenario: Dict[str, Any]
    scenario_id: str
    is_matterport: bool

    @property
    def source_stem(self) -> str:
        return self.source_envset_path.stem

    @property
    def scene_dir_name(self) -> str:
        # Backward-compatible: default to the parent folder name.
        return self.source_envset_path.parent.name

    @property
    def unique_key(self) -> str:
        """Generate a unique key combining scenario_id and source file path.

        This is necessary because multiple scenarios from different envset files
        can have the same scenario_id.
        """
        return f"{self.scenario_id}:{self.source_envset_path}"


@dataclass(frozen=True)
class ReplayCamera:
    name: str
    camera: Any


@dataclass(frozen=True)
class ReplayCameraOutput:
    rgb: Path
    depth: Optional[Path]


@dataclass
class CorrectionState:
    """Mutable state for periodic pose correction during replay."""

    correcting: bool = False
    step: int = 0
    start_x: float = 0.0
    start_y: float = 0.0
    start_yaw: float = 0.0
    goal_x: float = 0.0
    goal_y: float = 0.0
    goal_yaw: float = 0.0


class TrajectoryReplayer:
    """Reconstruct per-step velocities from pose waypoints."""

    def __init__(self, waypoints: List[Dict[str, Any]], dt: float, unit_scale: float):
        self._wps = sorted(waypoints, key=lambda wp: int(wp.get("frame", 0)))
        self._dt = float(dt)
        self._unit_scale = float(unit_scale)
        if not self._wps:
            raise ValueError("No replay waypoints provided")
        self._has_time_s = all("time_s" in wp for wp in self._wps)
        self._has_command = all("command" in wp for wp in self._wps)
        for wp in self._wps:
            if "yaw_deg" not in wp or "xyz" not in wp or "frame" not in wp:
                raise ValueError(f"Waypoint missing required fields: {wp}")

    def _check_timing(self) -> None:
        """Ensure frame gaps match time gaps within tolerance."""
        if not self._has_time_s:
            return
        tol_ratio = 0.05
        tol_abs = 1e-3
        for a, b in zip(self._wps[:-1], self._wps[1:]):
            frame_gap = int(b["frame"]) - int(a["frame"])
            if frame_gap <= 0:
                raise ValueError(f"Non-increasing frame sequence: {a['frame']} -> {b['frame']}")
            expected = frame_gap * self._dt
            delta_t = float(b["time_s"]) - float(a["time_s"])
            if abs(expected - delta_t) > max(tol_abs, tol_ratio * expected):
                raise ValueError(
                    f"Frame/time mismatch between frames {a['frame']} and {b['frame']}: "
                    f"expected {expected:.6f}s from dt, got {delta_t:.6f}s"
                )

    def _build_pose_sequence(self, *, validate_timing: bool = True) -> List[Tuple[float, float, float, float]]:
        """Interpolate poses for every physics step (frame)."""
        if validate_timing:
            self._check_timing()
        poses: List[Tuple[float, float, float, float]] = []

        start_frame = int(self._wps[0]["frame"])
        wps = self._wps

        # Normalize frames to start at 0
        normalized = []
        for wp in wps:
            frame = int(wp["frame"]) - start_frame
            pos = tuple(float(v) * self._unit_scale for v in wp["xyz"])
            yaw = math.radians(float(wp["yaw_deg"]))
            normalized.append((frame, pos, yaw))

        poses.append((*normalized[0][1], normalized[0][2]))
        for (f0, p0, y0), (f1, p1, y1) in zip(normalized[:-1], normalized[1:]):
            frame_gap = f1 - f0
            if frame_gap <= 0:
                raise ValueError(f"Invalid frame gap {frame_gap} between {f0} and {f1}")
            dyaw = _angle_diff_rad(y1, y0)
            for step in range(1, frame_gap + 1):
                alpha = step / frame_gap
                x = p0[0] + (p1[0] - p0[0]) * alpha
                y = p0[1] + (p1[1] - p0[1]) * alpha
                z = p0[2] + (p1[2] - p0[2]) * alpha
                yaw = y0 + dyaw * alpha
                poses.append((x, y, z, yaw))

        return poses

    def velocities(self) -> List[Tuple[float, float, float]]:
        """Return per-step (forward, lateral, angular) in stage units/sec."""
        poses = self._build_pose_sequence(validate_timing=True)
        if len(poses) < 2:
            raise ValueError("Insufficient waypoints to build velocities")
        vels: List[Tuple[float, float, float]] = []
        for p0, p1 in zip(poses[:-1], poses[1:]):
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            dyaw = _angle_diff_rad(p1[3], p0[3])
            world_vx = dx / self._dt
            world_vy = dy / self._dt
            yaw = p0[3]
            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)
            forward = world_vx * cos_y + world_vy * sin_y
            lateral = -world_vx * sin_y + world_vy * cos_y
            w = dyaw / self._dt
            vels.append((forward, lateral, w))
        return vels

    def pose_sequence(self, *, validate_timing: bool = True) -> List[Tuple[float, float, float, float]]:
        """Return per-frame (x, y, z, yaw_rad) in stage units."""
        return self._build_pose_sequence(validate_timing=validate_timing)

    def command_sequence(self) -> List[Tuple[float, float, float]]:
        """Return per-frame (v, lateral, w) from recorded commands.

        Uses the command from the most recent waypoint for each interpolated frame.
        Returns normalized commands (typically in [-1, 1] range).
        """
        if not self._has_command:
            raise ValueError("Waypoints do not contain 'command' field; cannot use command replay mode")

        start_frame = int(self._wps[0]["frame"])
        end_frame = int(self._wps[-1]["frame"])
        total_steps = end_frame - start_frame

        if total_steps <= 0:
            cmd = self._wps[0].get("command", {})
            return [(float(cmd.get("v", 0.0)), float(cmd.get("lateral", 0.0)), float(cmd.get("w", 0.0)))]

        # Build a list of (frame, v, lateral, w) for quick lookup
        wp_commands = []
        for wp in self._wps:
            frame = int(wp["frame"]) - start_frame
            cmd = wp.get("command", {})
            v = float(cmd.get("v", 0.0))
            lateral = float(cmd.get("lateral", 0.0))
            w = float(cmd.get("w", 0.0))
            wp_commands.append((frame, v, lateral, w))

        # For each step, use the command from the most recent waypoint
        commands: List[Tuple[float, float, float]] = []
        wp_idx = 0
        for step in range(total_steps):
            # Advance to the latest waypoint at or before this step
            while wp_idx + 1 < len(wp_commands) and wp_commands[wp_idx + 1][0] <= step:
                wp_idx += 1
            _, v, lateral, w = wp_commands[wp_idx]
            commands.append((v, lateral, w))

        return commands

    def has_command(self) -> bool:
        """Check if waypoints contain recorded commands."""
        return self._has_command


class ReplayRunner:
    """Replay recorded waypoints and export RGB frames."""

    # Waypoint writeback distance threshold (env units)
    # Keep writeback sampling roughly constant in real distance across datasets.
    # Convert to env units via: env_units = meters / units_in_meters.
    _WRITEBACK_DISTANCE_M: float = 0.05
    _WAIT_GAP_K: float = 5.0
    _WAIT_YAW_SKIP_DEG: float = 20.0
    _WAIT_MIN_S: float = 1.0

    # Correction thresholds
    _CORRECTION_THRESHOLD_M: float = 0.02  # 2cm position error threshold
    _CORRECTION_YAW_THRESHOLD_RAD: float = 0.05  # ~3 degrees yaw error threshold

    # Warmup steps
    _WARMUP_PHYSICS_STEPS: int = 2
    _WARMUP_RENDER_STEPS: int = 12

    # Articulation wait
    _ARTICULATION_WAIT_MAX_FRAMES: int = 50

    # Max steps safety multipliers
    _MAX_STEPS_MULTIPLIER: int = 3
    _MAX_STEPS_BUFFER: int = 2000
    _MAX_STEPS_MIN: int = 3000

    def __init__(self, config: ReplayConfig):
        self.config = config
        self._results_dir = config.output_dir

    @staticmethod
    def _log(message: str) -> None:
        print(message, flush=True)

    @classmethod
    def _writeback_distance_env_units(cls, unit_scale: float) -> float:
        """Convert the meter threshold to env units using units_in_meters."""
        try:
            scale = float(unit_scale)
        except Exception:
            scale = 0.0
        if not math.isfinite(scale) or scale <= 0.0:
            return float(cls._WRITEBACK_DISTANCE_M)
        return float(cls._WRITEBACK_DISTANCE_M) / scale

    def run(self) -> None:
        scenario_refs = self._load_scenarios()
        self._log(f"[Replay] Loaded {len(scenario_refs)} scenarios")
        self._results_dir.mkdir(parents=True, exist_ok=True)

        per_file_counts: Dict[Path, int] = {}
        for ref in scenario_refs:
            per_file_counts[ref.source_envset_path] = per_file_counts.get(ref.source_envset_path, 0) + 1

        groups = self._group_scenarios(scenario_refs)

        if len(groups) > 1:
            # Multiple folder groups require separate processes (Isaac Sim limitation:
            # SimulationApp can only be initialized once per process)
            self._log(f"[Replay] Detected {len(groups)} folder groups, using multiprocess mode")
            self._run_groups_multiprocess(groups, per_file_counts)
        else:
            # Single group can run in current process
            for idx, (group_dir, group_refs) in enumerate(groups):
                self._log(
                    f"[Replay] Group {idx + 1}/{len(groups)}: "
                    f"group_dir={group_dir} scenarios={len(group_refs)}"
                )
                self._run_scenario_group(group_refs, per_file_counts)

    def _run_groups_multiprocess(
        self,
        groups: List[Tuple[Path, List[ScenarioRef]]],
        per_file_counts: Dict[Path, int],
    ) -> None:
        """Run each scene group in a separate subprocess.

        Isaac Sim's SimulationApp can only be initialized once per Python process.
        When we have multiple folder groups, we spawn a new process for each group.
        """
        import subprocess
        import sys

        total_groups = len(groups)
        for idx, (group_dir, group_refs) in enumerate(groups):
            scenario_ids = [ref.scenario_id for ref in group_refs]

            # Collect unique envset files for this group
            envset_files = list(set(ref.source_envset_path for ref in group_refs))

            self._log(
                f"\n[Replay] Group {idx + 1}/{total_groups}: "
                f"group_dir={group_dir} scenarios={len(group_refs)}"
            )

            # Check if all scenarios in this group are already completed
            all_completed = True
            for ref in group_refs:
                out_dir = self._scenario_output_dir(
                    ref, scenarios_in_file=per_file_counts.get(ref.source_envset_path, 1)
                )
                if not self._is_scenario_completed(out_dir, ref.scenario_id):
                    all_completed = False
                    break

            if all_completed:
                self._log(f"[Replay] Group {idx + 1}/{total_groups}: all scenarios completed, skipping")
                continue

            # Build subprocess command
            cmd = self._build_subprocess_cmd(envset_files, scenario_ids)
            self._log(f"[Replay] Subprocess cmd: {' '.join(cmd)}")

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                self._log(f"[Replay] ERROR: Group {idx + 1}/{total_groups} failed: {e}")
                # Continue to next group instead of failing entirely
                continue
            except KeyboardInterrupt:
                self._log("\n[Replay] Interrupted by user")
                raise

        self._log(f"\n[Replay] All {total_groups} groups processed")

    def _build_subprocess_cmd(
        self,
        envset_files: List[Path],
        scenario_ids: List[str],
    ) -> List[str]:
        """Build the subprocess command for running a single group."""
        import sys

        # Find runReplay.py relative to this file
        # replay_runner.py is at bench/replay/replay_runner.py
        # runReplay.py is at the repo root
        run_replay_script = Path(__file__).resolve().parent.parent.parent / "runReplay.py"

        cmd = [sys.executable, str(run_replay_script)]

        # Required args
        cmd.extend(["--config", str(self.config.uninav_config.resolve())])
        cmd.extend(["--output", str(self.config.output_dir.resolve())])

        # Pass envset files - if single file, pass directly; otherwise pass first one
        # and use --scenario to filter
        if len(envset_files) == 1:
            cmd.extend(["--envset", str(envset_files[0].resolve())])
        else:
            # Multiple envset files for same scene - pass parent directory
            # and rely on scenario filtering
            common_parent = envset_files[0].parent
            cmd.extend(["--envset", str(common_parent.resolve())])

        # Pass scenario IDs to filter
        for sid in scenario_ids:
            cmd.extend(["--scenario", sid])

        # Optional args
        if self.config.scene_root:
            cmd.extend(["--scene-root", str(self.config.scene_root.resolve())])
        if self.config.headless:
            cmd.append("--headless")

        cmd.extend(["--fps", str(self.config.fps)])
        cmd.extend(["--num-cameras", str(self._normalize_num_cameras())])
        cmd.extend(["--kp-pos", str(self.config.track_kp_pos)])
        cmd.extend(["--kp-yaw", str(self.config.track_kp_yaw)])
        cmd.extend(["--lookahead-steps", str(self.config.track_lookahead_steps)])
        cmd.extend(["--smooth-alpha", str(self.config.track_smoothing_alpha)])
        cmd.extend(["--correction-period", str(self.config.track_correction_period)])
        cmd.extend(["--correction-frames", str(self.config.track_correction_frames)])

        if not self.config.skip_completed:
            cmd.append("--no-skip")
        cmd.extend(["--skip-min-frames", str(self.config.skip_min_frames)])

        # Video output options
        if not self.config.output_video:
            cmd.append("--no-video")
        if not self.config.save_depth:
            cmd.append("--no-depth")

        return cmd

    @staticmethod
    def _group_scenarios(scenario_refs: List[ScenarioRef]) -> List[Tuple[Path, List[ScenarioRef]]]:
        """Group scenarios strictly by their parent folder.

        For this repo's envset layout, each scene/group lives under a folder:
        <root>/<group_folder>/*.json
        Each group is replayed in a single SimulationApp lifecycle to maximize reuse.
        """
        groups: Dict[Path, List[ScenarioRef]] = {}
        ordered_keys: List[Path] = []
        for ref in scenario_refs:
            key = ref.source_envset_path.parent
            if key not in groups:
                groups[key] = []
                ordered_keys.append(key)
            groups[key].append(ref)
        return [(key, groups[key]) for key in ordered_keys]

    def _iter_envset_files(self) -> List[Path]:
        """Resolve envset JSON files from envset_path (file or directory).

        Directory scanning matches batch_replay_driver.py: root/*.json + root/*/*.json.
        """
        path = self.config.envset_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Envset not found: {path}")

        json_files = scan_json_files(path)
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in: {path}")
        return json_files

    def _load_scenarios(self) -> List[ScenarioRef]:
        scenario_refs: List[ScenarioRef] = []
        id_filter = set(self.config.scenario_ids or [])

        for envset_file in self._iter_envset_files():
            with envset_file.open("r", encoding="utf-8") as f:
                envset = json.load(f)
            scenarios = envset.get("scenarios", [])
            if id_filter:
                scenarios = [s for s in scenarios if s.get("id") in id_filter]
            for scenario in scenarios:
                if not isinstance(scenario, dict):
                    raise ValueError(f"Invalid scenario entry in {envset_file}: expected dict, got {type(scenario)}")

                # Normalize envset file paths before computing keys.
                EnvsetConfigLoader.normalize_scenario_paths(scenario, self.config.scene_root)

                scenario_id = str(scenario.get("id") or "unknown")
                mp = is_matterport_scenario(scenario.get("scene", {}) if isinstance(scenario.get("scene"), dict) else {})

                scenario_refs.append(
                    ScenarioRef(
                        source_envset_path=envset_file,
                        scenario=scenario,
                        scenario_id=scenario_id,
                        is_matterport=mp,
                    )
                )

        return scenario_refs

    def _scenario_output_dir(self, ref: ScenarioRef, *, scenarios_in_file: int) -> Path:
        """Return per-scenario output directory.

        Backward compatibility:
        - If the source file yields exactly one scenario, output matches historical layout:
          output_root/<envset_stem>/<scene_dir>/
        - Otherwise, include scenario_id to avoid collisions:
          output_root/<envset_stem>/<scene_dir>/<scenario_id>/
        """
        base = self._results_dir / ref.source_stem / ref.scene_dir_name
        if scenarios_in_file == 1:
            return base
        return base / ref.scenario_id

    def _build_group_config_model(self, group_refs: List[ScenarioRef]):
        from OmniNav.core.config import Config

        merged_configs: List[Dict[str, Any]] = []
        for ref in group_refs:
            loader = EnvsetConfigLoader(
                config_path=self.config.uninav_config,
                envset_path=ref.source_envset_path,
                scenario_id=ref.scenario_id,
                scene_root=self.config.scene_root,
            )
            bundle = loader.load()
            merged_config = bundle.config
            merged_config.setdefault("simulator", {})["headless"] = self.config.headless
            merged_configs.append(merged_config)

        if not merged_configs:
            raise ValueError("No scenarios provided for group config")

        base_config = dict(merged_configs[0])
        task_configs: List[Any] = []
        for merged in merged_configs:
            tasks = merged.get("task_configs")
            if not isinstance(tasks, list) or not tasks:
                raise ValueError("Replay expects task_configs in merged config")
            if len(tasks) != 1:
                raise ValueError(
                    f"Replay expects one task_config per scenario, got {len(tasks)}"
                )
            task_configs.extend(tasks)

        base_config["task_configs"] = task_configs
        base_config.setdefault("simulator", {})["headless"] = self.config.headless

        try:
            config_model = Config.model_validate(base_config)
        except AttributeError:
            config_model = Config.parse_obj(base_config)
        self._apply_replay_robot_sensor_config(config_model)
        return config_model

    def _apply_replay_robot_sensor_config(self, config_model) -> None:
        try:
            from bench.replay.camera_config import configure_replay_robot_sensors
        except Exception as exc:
            raise RuntimeError("Failed to import bench.replay.camera_config for replay sensor setup") from exc

        for task_cfg in (getattr(config_model, "task_configs", None) or []):
            robots = getattr(task_cfg, "robots", None) or []
            for robot_cfg in robots:
                try:
                    configure_replay_robot_sensors(
                        robot_cfg,
                        enable_depth=bool(self.config.save_depth),
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to configure replay sensors for robot '{getattr(robot_cfg, 'name', '?')}'"
                    ) from exc

                sensors = getattr(robot_cfg, "sensors", None) or []
                self._log(
                    f"[Replay] Applied replay camera config to robot '{getattr(robot_cfg, 'name', '?')}': "
                    f"{[getattr(s, 'name', '?') for s in sensors]}"
                )

    def _setup_first_scenario_navmesh(
        self,
        runner,
        scenario: Dict[str, Any],
        scenario_id: str,
        scene_cfg: Dict[str, Any],
        navmesh_cfg: Dict[str, Any],
        is_matterport: bool,
        simulation_app,
    ) -> None:
        """Setup NavMesh for the first scenario in a group."""
        from OmniNavExt.envset.core.scene_manager import find_scene_root
        from OmniNavExt.envset.core import PhysicsManager, NavMeshManager
        from OmniNavExt.envset.runtime_hooks import EnvsetTaskRuntime

        self._log(f"[Replay][{scenario_id}] Step: find_scene_root + fix_grscenes_physics")
        scene_root = find_scene_root(runner._stage, scenario)
        self._log(f"[Replay][{scenario_id}] scene_root={scene_root}")
        PhysicsManager.fix_grscenes_physics(scene_cfg, scene_root)

        navmesh_success = True
        if not is_matterport:
            self._log(f"[Replay][{scenario_id}] Step: exclude robots from NavMesh")
            for task_name, task in runner.current_tasks.items():
                robots = getattr(task, "robots", {}) or {}
                self._log(f"[Replay][{scenario_id}] task={task_name} robots={list(robots.keys())}")
                for robot_name, robot in robots.items():
                    if hasattr(robot, "config") and hasattr(robot.config, "prim_path"):
                        prim_path = robot.config.prim_path
                        self._log(f"[Replay][{scenario_id}]   - exclude {robot_name} prim={prim_path}")
                        EnvsetTaskRuntime._exclude_from_navmesh(prim_path)

            self._log(f"[Replay][{scenario_id}] Step: bake NavMesh")
            navmesh_manager = NavMeshManager.from_scenario(navmesh_cfg, scene_cfg, scene_root)
            try:
                cfg = getattr(navmesh_manager, "_config", None)
                if cfg is not None:
                    self._log(
                        f"[Replay][{scenario_id}] NavMesh resolved root={cfg.root_prim_path} "
                        f"z_padding={cfg.z_padding} agent_radius={cfg.agent_radius} "
                        f"max_step_height={cfg.max_step_height}"
                    )
            except Exception as e:
                self._log(f"[Replay][{scenario_id}] Warning: could not log NavMesh config: {e}")
            navmesh_success = navmesh_manager.bake_sync(simulation_app, envset_cfg=scenario)
            self._log(f"[Replay][{scenario_id}] NavMesh success={navmesh_success}")

        if not navmesh_success:
            raise RuntimeError("NavMesh baking failed")

        self._log(f"[Replay][{scenario_id}] Step: disable NavMesh visualization")
        import omni.kit.commands
        omni.kit.commands.execute(
            "ChangeSetting",
            path="/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh",
            value=False,
        )

    @staticmethod
    def _apply_sdf_to_scene_meshes(stage, scene_root_path: str, *, log_prefix: str = "[SDF]") -> None:
        from pxr import PhysxSchema, UsdPhysics, UsdGeom, Usd

        def _has_points(mesh_prim):
            pts = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
            return pts is not None and len(pts) > 0

        scene_prim = stage.GetPrimAtPath(scene_root_path)
        if not scene_prim or not scene_prim.IsValid():
            print(f"{log_prefix} Scene root not found: {scene_root_path}")
            return

        target_roots = {}
        for prim in Usd.PrimRange(scene_prim):
            path_str = str(prim.GetPath())
            if "/robots/" in path_str:
                continue
            if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(PhysxSchema.PhysxMeshMergeCollisionAPI):
                target_roots[path_str] = prim

        converted_count = 0
        skipped_small = 0
        skipped_empty = 0
        seen_meshes = set()
        root_count = 0
        for root in target_roots.values():
            root_count += 1
            for child in Usd.PrimRange(root):
                if child == root:
                    continue
                child_path = str(child.GetPath())
                if "/robots/" in child_path or child_path in seen_meshes:
                    continue
                if not child.IsA(UsdGeom.Mesh):
                    continue
                if not _has_points(child):
                    skipped_empty += 1
                    continue
                mesh = UsdGeom.Mesh(child)
                face_counts = mesh.GetFaceVertexCountsAttr().Get()
                num_faces = len(face_counts) if face_counts else 0
                if num_faces < 100000:
                    skipped_small += 1
                    continue
                mesh_api = UsdPhysics.MeshCollisionAPI.Apply(child)
                mesh_api.CreateApproximationAttr().Set("convexHull")
                PhysxSchema.PhysxConvexHullCollisionAPI.Apply(child)
                converted_count += 1
                seen_meshes.add(child_path)
                print(f"[FIX] {child.GetPath()} faces={num_faces} -> convexHull")
        print(
            f"[FIX] Total: {converted_count} large mesh(es) converted from {root_count} collision root(s), "
            f"skipped {skipped_small} small mesh(es), skipped {skipped_empty} empty mesh(es) "
            f"under {scene_root_path}"
        )

    def _run_single_scenario(
        self,
        runner,
        ref: ScenarioRef,
        out_dir: Path,
        is_first_scenario: bool,
        is_matterport: bool,
        scene_cfg: Dict[str, Any],
        navmesh_cfg: Dict[str, Any],
        simulation_app,
    ) -> None:
        """Run replay for a single scenario within a group."""
        from OmniNavExt.envset.runtime_hooks import EnvsetTaskRuntime
        from OmniNavExt.envset.core import PhysicsManager
        import omni.timeline

        scenario = ref.scenario
        scenario_id = ref.scenario_id

        self._log(f"[Replay][{scenario_id}] Begin scenario")
        self._log(f"[Replay][{scenario_id}] config={self.config.uninav_config}")
        self._log(f"[Replay][{scenario_id}] envset={ref.source_envset_path}")
        self._log(f"[Replay][{scenario_id}] scene_root={self.config.scene_root}")
        self._log(f"[Replay][{scenario_id}] output_dir={out_dir}")

        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing():
            self._log(f"[Replay][{scenario_id}] Step: timeline.pause before reset")
            timeline.pause()

        EnvsetTaskRuntime.reset_episode_state(stage=runner._stage)

        self._log(f"[Replay][{scenario_id}] Step: runner.reset(start_timeline=False)")
        runner.reset(start_timeline=False)

        if is_first_scenario:
            self._setup_first_scenario_navmesh(
                runner, scenario, scenario_id, scene_cfg, navmesh_cfg, is_matterport, simulation_app
            )

        self._log(f"[Replay][{scenario_id}] Step: timeline.play + init vhumans + wait articulations")
        timeline.play()
        EnvsetTaskRuntime.reconcile_virtual_humans(
            scenario,
            stage=runner._stage,
            defer_routes=True,
        )
        EnvsetTaskRuntime.register_robots_as_dynamic_obstacles(runner)
        ok = PhysicsManager.wait_for_articulations(runner, max_frames=self._ARTICULATION_WAIT_MAX_FRAMES)
        self._log(f"[Replay][{scenario_id}] articulations_ready={ok}")

        # Set robot to the episode start pose before warmup/capture.
        try:
            robot_name = self._get_robot_name(scenario)
            task = list(runner.current_tasks.values())[0]
            robot = task.robots.get(robot_name)
            waypoints = resolve_recording_waypoints(scenario, envset_path=source_envset_path)
            unit_scale = scenario.get("scene", {}).get("units_in_meters")
            if robot is not None and waypoints and unit_scale is not None:
                self._log(f"[Replay][{scenario_id}] Step: set robot to start pose")
                self._maybe_snap_robot_to_first_waypoint(
                    scenario=scenario,
                    waypoints=waypoints,
                    unit_scale=float(unit_scale),
                    robot=robot,
                )
        except Exception as exc:
            self._log(f"[Replay][{scenario_id}] Set start pose failed: {exc}")

        self._log(f"[Replay][{scenario_id}] Step: warmup ({self._WARMUP_PHYSICS_STEPS} physics, {self._WARMUP_RENDER_STEPS} render)")
        runner.render_interval = 0
        runner.render_trigger = 0
        runner.warm_up(steps=self._WARMUP_PHYSICS_STEPS, render=False, physics=True)
        runner.warm_up(steps=self._WARMUP_RENDER_STEPS, render=True, physics=False)
        self._log(f"[Replay][{scenario_id}] Step: inject vhuman routes after warmup")
        EnvsetTaskRuntime._setup_virtual_routes(scenario)

        self._replay(
            runner,
            scenario,
            bundle=None,
            out_dir=out_dir,
            source_envset_path=ref.source_envset_path,
            pose_already_set=True,
        )
        self._log(f"[Replay][{scenario_id}] Done")

    def _run_scenario_group(
        self,
        group_refs: List[ScenarioRef],
        per_file_counts: Dict[Path, int],
        *,
        output_overrides: Optional[Dict[str, Path]] = None,
    ) -> None:
        from OmniNav.core.task_config_manager.base import create_task_config_manager
        from OmniNavExt import import_extensions
        from OmniNavExt.envset.core import SimulationBootstrap, SimulationConfig
        from OmniNavExt.envset.core.scene_manager import SceneManager
        import traceback

        if not group_refs:
            return

        # === Pre-filter: skip completed scenarios BEFORE starting SimulationApp ===
        pending_refs: List[ScenarioRef] = []
        pending_out_dirs: Dict[str, Path] = {}  # unique_key -> out_dir

        for ref in group_refs:
            out_dir = None
            if output_overrides:
                out_dir = output_overrides.get(ref.scenario_id)
            if out_dir is None:
                out_dir = self._scenario_output_dir(
                    ref,
                    scenarios_in_file=per_file_counts.get(ref.source_envset_path, 1),
                )

            if self._is_scenario_completed(out_dir, ref.scenario_id):
                # Already logged by _is_scenario_completed
                continue

            pending_refs.append(ref)
            pending_out_dirs[ref.unique_key] = out_dir

        if not pending_refs:
            self._log(f"[Replay][Group] All {len(group_refs)} scenarios already completed, skipping group")
            return

        self._log(
            f"[Replay][Group] {len(pending_refs)}/{len(group_refs)} scenarios pending "
            f"({len(group_refs) - len(pending_refs)} skipped)"
        )

        # === Now start SimulationApp with only pending scenarios ===
        first_ref = pending_refs[0]
        first_scenario = first_ref.scenario
        scene_cfg = first_scenario.get("scene", {}) if isinstance(first_scenario.get("scene"), dict) else {}
        navmesh_cfg = first_scenario.get("navmesh", {}) if isinstance(first_scenario.get("navmesh"), dict) else {}

        self._log(f"[Replay][Group] Begin group with {len(pending_refs)} scenario(s)")
        sim_bootstrap = None
        try:
            config_model = self._build_group_config_model(pending_refs)
            self._log(f"[Replay][Group] Init SimulationApp (headless={self.config.headless})")
            sim_bootstrap = SimulationBootstrap(SimulationConfig(headless=self.config.headless))
            simulation_app = sim_bootstrap.initialize()

            self._log("[Replay][Group] import_extensions")
            import_extensions()

            # Modules below depend on SimulationApp being initialized to load carb/omni.
            from OmniNavExt.envset.runtime_hooks import EnvsetTaskRuntime
            from OmniNavExt.envset.world_utils import bootstrap_world_if_needed

            self._log("[Replay][Group] Create SimulatorRunner")
            task_manager = create_task_config_manager(config_model)
            runner = self._create_runner_with_app(config_model, task_manager, simulation_app)

            scene_root_path = scene_cfg.get("root_prim_path", "/World")

            def _sdf_hook(stage):
                try:
                    self._apply_sdf_to_scene_meshes(stage, scene_root_path, log_prefix="[SDF-Hook]")
                except Exception as exc:
                    print(f"[SDF-Hook] Failed: {exc}")
                    import traceback
                    traceback.print_exc()

            runner._pre_physics_hook = _sdf_hook

            self._log("[Replay][Group] bootstrap_world_if_needed")
            bootstrap_world_if_needed()
            EnvsetTaskRuntime.reset_navmesh_cache()

            is_mp = first_ref.is_matterport
            self._log(f"[Replay][Group] is_matterport={is_mp}")
            if is_mp:
                self._log("[Replay][Group] Import matterport scene")
                matterport_prim = SceneManager.import_matterport_scene(scene_cfg, self.config.scene_root)
                self._log(f"[Replay][Group] matterport_prim={matterport_prim}")
                self._log("[Replay][Group] Prepare matterport navmesh")
                SceneManager.prepare_matterport_navmesh(
                    matterport_prim, navmesh_cfg, scene_cfg, simulation_app, envset_cfg=first_scenario
                )
                self._log("[Replay][Group] Set camera light")
                self._set_camera_light()

            # Run each pending scenario in the group
            first_scenario_run = False
            for idx, ref in enumerate(pending_refs):
                out_dir = pending_out_dirs[ref.unique_key]

                try:
                    self._run_single_scenario(
                        runner=runner,
                        ref=ref,
                        out_dir=out_dir,
                        is_first_scenario=(not first_scenario_run),
                        is_matterport=is_mp,
                        scene_cfg=scene_cfg,
                        navmesh_cfg=navmesh_cfg,
                        simulation_app=simulation_app,
                    )
                    first_scenario_run = True
                except BaseException as exc:
                    self._log(f"[Replay][{ref.scenario_id}] ERROR: {exc}")
                    self._log(traceback.format_exc())
                    raise

        except BaseException as exc:
            self._log(f"[Replay][Group] FATAL: {exc}")
            self._log(traceback.format_exc())
            raise
        finally:
            if sim_bootstrap is not None:
                self._log("[Replay][Group] Shutdown SimulationApp")
                sim_bootstrap.shutdown()

    def _run_scenario_isolated(self, ref: ScenarioRef, out_dir: Path) -> None:
        self._run_scenario_group(
            [ref],
            {ref.source_envset_path: 1},
            output_overrides={ref.scenario_id: out_dir},
        )

    def _is_scenario_completed(self, out_dir: Path, scenario_id: str) -> bool:
        """Check if a scenario has already been replayed with sufficient frames.

        Returns True if skip_completed is enabled and output exists:
        - Video mode: path/path.json exists, or video/front/rgb.mp4 exists
        - Image mode: front/rgb/ contains >= skip_min_frames files
        """
        if not self.config.skip_completed:
            return False

        if self.config.output_video:
            path_json_candidates = [
                out_dir / "path" / "path.json",
                out_dir / "trajectory.json",
            ]
            for path_json in path_json_candidates:
                if not path_json.exists():
                    continue
                self._log(
                    f"[Replay][{scenario_id}] SKIP: path recording already exists ({path_json})"
                )
                return True
            rgb_video_candidates = [
                out_dir / "video" / "front" / "rgb.mp4",
                out_dir / "front" / "rgb.mp4",
            ]
            for rgb_video in rgb_video_candidates:
                if not rgb_video.exists():
                    continue
                self._log(
                    f"[Replay][{scenario_id}] SKIP: video already exists ({rgb_video})"
                )
                return True
            return False
        else:
            rgb_dir = out_dir / "front" / "rgb"
            if not rgb_dir.exists():
                return False

            frame_count = sum(1 for f in rgb_dir.iterdir() if f.is_file() and f.name.startswith("frame_"))

            if frame_count >= self.config.skip_min_frames:
                self._log(
                    f"[Replay][{scenario_id}] SKIP: already completed "
                    f"({frame_count} frames >= {self.config.skip_min_frames})"
                )
                return True

            return False

    def _normalize_num_cameras(self) -> int:
        return 3 if int(getattr(self.config, "num_cameras", 1)) == 3 else 1

    def _prepare_camera_outputs(
        self,
        out_dir: Path,
        scenario_id: str,
        camera_names: List[str],
    ) -> Dict[str, ReplayCameraOutput]:
        out_dir.mkdir(parents=True, exist_ok=True)

        outputs: Dict[str, ReplayCameraOutput] = {}
        if self.config.output_video:
            video_root = out_dir / "video"
            for camera_name in camera_names:
                camera_dir = video_root / camera_name
                camera_dir.mkdir(parents=True, exist_ok=True)
                outputs[camera_name] = ReplayCameraOutput(
                    rgb=camera_dir / "rgb.mp4",
                    depth=(camera_dir / "depth.mp4" if self.config.save_depth else None),
                )
            self._log(
                f"[Replay][{scenario_id}] Output videos ready for cameras={camera_names}: out_dir={out_dir}"
            )
            return outputs

        for camera_name in camera_names:
            camera_dir = out_dir / camera_name
            rgb_dir = camera_dir / "rgb"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir = camera_dir / "depth" if self.config.save_depth else None
            if depth_dir is not None:
                depth_dir.mkdir(parents=True, exist_ok=True)
            outputs[camera_name] = ReplayCameraOutput(
                rgb=rgb_dir,
                depth=depth_dir,
            )
        self._log(
            f"[Replay][{scenario_id}] Output image dirs ready for cameras={camera_names}: out_dir={out_dir}"
        )
        return outputs

    def _should_skip_existing_replay(self, out_dir: Path, scenario_id: str) -> bool:
        if not out_dir.exists():
            return False
        rgb_dir = out_dir / "front" / "rgb"
        if not rgb_dir.exists():
            return False
        image_suffixes = {".png", ".jpg", ".jpeg"}
        rgb_count = sum(1 for p in rgb_dir.iterdir() if p.is_file() and p.suffix.lower() in image_suffixes)
        if rgb_count >= 40:
            self._log(
                f"[Replay][{scenario_id}] Skip replay: output exists and rgb has {rgb_count} images (>=70)"
            )
            return True
        return False

    def _maybe_snap_robot_to_first_waypoint(
        self, *, scenario: Dict[str, Any], waypoints: List[Dict[str, Any]], unit_scale: float, robot
    ) -> None:
        # Use initial_pose if present, otherwise fall back to first waypoint.
        init_pose = scenario.get("robots", {}).get("entries", [])[0].get("initial_pose") or {}
        init_pos = init_pose.get("position") or init_pose.get("xyz")
        init_yaw_deg = init_pose.get("orientation_deg")

        wp0 = sorted(waypoints, key=lambda wp: int(wp.get("frame", 0)))[0]
        xyz0 = wp0.get("xyz", (0.0, 0.0, 0.0))

        if isinstance(init_pos, (list, tuple)) and len(init_pos) >= 2:
            x0 = float(init_pos[0]) * float(unit_scale)
            y0 = float(init_pos[1]) * float(unit_scale)
            z0_logged = float(init_pos[2]) * float(unit_scale) if len(init_pos) >= 3 else None
        else:
            x0 = float(xyz0[0]) * float(unit_scale)
            y0 = float(xyz0[1]) * float(unit_scale)
            z0_logged = float(xyz0[2]) * float(unit_scale)

        yaw_deg = init_yaw_deg if init_yaw_deg is not None else wp0.get("yaw_deg", 0.0)
        yaw0 = math.radians(float(yaw_deg))
        quat0 = (math.cos(yaw0 / 2.0), 0.0, 0.0, math.sin(yaw0 / 2.0))
        articulation = getattr(robot, "articulation", None)
        if articulation is None:
            self._log("[Replay] Warning: robot has no articulation, cannot snap to first waypoint")
            return

        z_target = self._resolve_replay_fixed_z(
            scenario=scenario,
            waypoints=waypoints,
            unit_scale=float(unit_scale),
            articulation=articulation,
        )

        articulation.set_world_pose((x0, y0, z_target), quat0)
        try:
            articulation.set_linear_velocity(np.zeros(3))
            articulation.set_angular_velocity(np.zeros(3))
            articulation.set_joint_velocities(np.zeros(articulation.num_dof))
        except Exception as e:
            self._log(f"[Replay] Warning: failed to reset velocities: {e}")

    def _resolve_replay_fixed_z(
        self,
        *,
        scenario: Dict[str, Any],
        waypoints: List[Dict[str, Any]],
        unit_scale: float,
        articulation=None,
    ) -> float:
        """Resolve the fixed Z used by replay, preferring initial_pose.z.

        Order of preference:
        1. scenario.robots.entries[0].initial_pose.position[2]
        2. first waypoint xyz[2]
        3. articulation current z
        4. 0.0
        """
        init_pose = scenario.get("robots", {}).get("entries", [])[0].get("initial_pose") or {}
        init_pos = init_pose.get("position") or init_pose.get("xyz")
        if isinstance(init_pos, (list, tuple)) and len(init_pos) >= 3:
            z_target = float(init_pos[2]) * float(unit_scale)
            self._log(f"[Replay] Using initial_pose.z for replay height: {z_target:.4f}")
            return z_target

        wp0 = sorted(waypoints, key=lambda wp: int(wp.get("frame", 0)))[0] if waypoints else {}
        xyz0 = wp0.get("xyz", (0.0, 0.0, 0.0))
        if isinstance(xyz0, (list, tuple)) and len(xyz0) >= 3:
            z_target = float(xyz0[2]) * float(unit_scale)
            self._log(f"[Replay] initial_pose.z missing, using first waypoint z: {z_target:.4f}")
            return z_target

        if articulation is not None:
            art_z = get_articulation_z(articulation)
            if art_z is not None:
                self._log(f"[Replay] initial_pose.z missing, using articulation z: {art_z:.4f}")
                return float(art_z)

        self._log("[Replay] initial_pose.z and waypoint z missing, defaulting replay height to 0.0")
        return 0.0

    def _replay_carter_v1(
        self,
        *,
        runner: "SimulatorRunner",
        robot,
        scenario: Dict[str, Any],
        scenario_id: str,
        waypoints: List[Dict[str, Any]],
        unit_scale: float,
        dt: float,
        out_dir: Path,
        source_envset_path: Path,
    ) -> None:
        move_along_path = robot.controllers.get("move_along_path")
        if move_along_path is None:
            raise ValueError(
                "move_along_path controller not found on carter_v1; "
                "ensure envset controller injection includes move_along_path."
            )
        move_by_speed = robot.controllers.get("move_by_speed")
        if move_by_speed is None:
            raise ValueError("move_by_speed controller not found on robot (required for command scaling)")

        cameras = self._resolve_replay_cameras(robot)
        camera_outputs = self._prepare_camera_outputs(
            out_dir=out_dir,
            scenario_id=scenario_id,
            camera_names=[cam.name for cam in cameras],
        )
        self._log(f"[Replay][{scenario_id}] Cameras resolved: {[cam.name for cam in cameras]}")

        frames = [int(wp.get("frame", 0)) for wp in waypoints if isinstance(wp, dict)]
        start_frame = min(frames) if frames else 0
        end_frame = max(frames) if frames else 0
        total_steps = end_frame - start_frame
        steps = max(1, int(total_steps))
        self._log(f"[Replay][{scenario_id}] move_along_path replay: steps={steps} dt={dt:.6f}")

        replay_gt_path, vh_new = self._replay_move_along_path(
            runner=runner,
            robot=robot,
            robot_name=self._get_robot_name(scenario),
            move_along_path=move_along_path,
            move_by_speed=move_by_speed,
            waypoints=waypoints,
            unit_scale=float(unit_scale),
            steps=steps,
            dt=dt,
            cameras=cameras,
            camera_outputs=camera_outputs,
            out_dir=out_dir,
            scenario=scenario,
            scenario_id=scenario_id,
        )

        self._writeback_waypoints(
            envset_path=source_envset_path,
            scenario_id=scenario_id,
            gt_path=replay_gt_path,
            vh_gt_waypoints=vh_new,
        )
        self._log(
            f"[Replay][{scenario_id}] Recording written back: gt_path={len(replay_gt_path)} "
            f"vh_agents={len(vh_new) if vh_new else 0}"
        )

    def _replay(
        self,
        runner: "SimulatorRunner",
        scenario: Dict[str, Any],
        bundle,
        out_dir: Path,
        source_envset_path: Path,
        pose_already_set: bool = False,
    ) -> None:
        scenario_id = scenario.get("id", "unknown")
        self._log(f"[Replay][{scenario_id}] Enter _replay() out_dir={out_dir}")
        robot_name = self._get_robot_name(scenario)
        task = list(runner.current_tasks.values())[0]
        robot = task.robots.get(robot_name)
        if robot is None:
            raise ValueError(f"Robot '{robot_name}' not found in task")

        try:
            self._log(f"[Replay][{scenario_id}] robot={robot_name} controllers={list(getattr(robot, 'controllers', {}).keys())}")
        except Exception:
            pass

        waypoints = resolve_recording_waypoints(scenario, envset_path=source_envset_path)
        if not waypoints:
            raise ValueError("Scenario missing recording.gt_path and legacy rb_gt_waypoints for replay")
        try:
            frames = [int(wp.get("frame", 0)) for wp in waypoints if isinstance(wp, dict)]
            start_frame = min(frames) if frames else 0
            end_frame = max(frames) if frames else 0
            total_steps = end_frame - start_frame
            has_cmd = all(isinstance(wp, dict) and "command" in wp for wp in waypoints)
            self._log(
                f"[Replay][{scenario_id}] recording.gt_path: n={len(waypoints)} "
                f"frame_min={start_frame} frame_max={end_frame} total_steps={total_steps} has_command={has_cmd}"
            )
        except Exception:
            pass

        unit_scale = scenario.get("scene", {}).get("units_in_meters")
        if unit_scale is None:
            raise ValueError("scene.units_in_meters missing; cannot scale trajectory")

        robot_type = ""
        try:
            entries = scenario.get("robots", {}).get("entries", [])
            if entries and isinstance(entries[0], dict):
                robot_type = str(entries[0].get("type") or "").strip().lower()
        except Exception:
            robot_type = ""

        dt = float(getattr(runner, "dt", runner.config.simulator.physics_dt))

        # CarterV1 / Aliengo / H1: unified teleport replay.
        # Use kinematic pose playback for stable multi-robot/multi-camera recording.
        if robot_type in ("carter_v1", "h1robot", "h1", "aliengo", "aliengorobot"):
            if not pose_already_set:
                self._maybe_snap_robot_to_first_waypoint(
                    scenario=scenario,
                    waypoints=waypoints,
                    unit_scale=float(unit_scale),
                    robot=robot,
                )
            cameras = self._resolve_replay_cameras(robot)
            camera_outputs = self._prepare_camera_outputs(
                out_dir=out_dir,
                scenario_id=scenario_id,
                camera_names=[cam.name for cam in cameras],
            )
            traj = TrajectoryReplayer(waypoints=waypoints, dt=dt, unit_scale=unit_scale)
            self._log(f"[Replay][{scenario_id}] Enter _replay_teleport() for {robot_type}")
            self._replay_teleport(
                runner=runner,
                robot=robot,
                scenario=scenario,
                traj=traj,
                cameras=cameras,
                camera_outputs=camera_outputs,
                dt=dt,
                unit_scale=float(unit_scale),
                out_dir=out_dir,
                scenario_id=scenario_id,
            )
            self._log(f"[Replay][{scenario_id}] Exit _replay_teleport()")
            return

        # Ensure robot starts at the episode pose before recording.
        if not pose_already_set:
            self._maybe_snap_robot_to_first_waypoint(
                scenario=scenario,
                waypoints=waypoints,
                unit_scale=float(unit_scale),
                robot=robot,
            )

        controller = robot.controllers.get("move_by_speed")
        if controller is None:
            raise ValueError("move_by_speed controller not found on robot")
        forward_scale, rot_scale = self._extract_controller_scales(controller)
        action_builder = self._action_builder(controller)

        cameras = self._resolve_replay_cameras(robot)
        camera_outputs = self._prepare_camera_outputs(
            out_dir=out_dir,
            scenario_id=scenario_id,
            camera_names=[cam.name for cam in cameras],
        )
        self._log(f"[Replay][{scenario_id}] Cameras resolved: {[cam.name for cam in cameras]}")

        self._log(f"[Replay][{scenario_id}] Build TrajectoryReplayer")
        traj = TrajectoryReplayer(waypoints=waypoints, dt=dt, unit_scale=unit_scale)
        self._log(f"[Replay][{scenario_id}] Enter _replay_track()")
        self._replay_track(
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
            out_dir=out_dir,
            scenario_id=scenario_id,
        )
        self._log(f"[Replay][{scenario_id}] Exit _replay_track()")

    def _check_and_start_correction(
        self,
        correction_state: CorrectionState,
        step_idx: int,
        correction_period: int,
        anchor_x: float,
        anchor_y: float,
        anchor_yaw: float,
        cur_x: float,
        cur_y: float,
        cur_yaw: float,
        correction_frames: int,
    ) -> None:
        """Check if periodic correction is needed and initialize correction state."""
        if correction_state.correcting or step_idx == 0 or step_idx % correction_period != 0:
            return

        dx_a = anchor_x - cur_x
        dy_a = anchor_y - cur_y
        anchor_err = (dx_a * dx_a + dy_a * dy_a) ** 0.5
        yaw_err_abs = abs(_angle_diff_rad(anchor_yaw, cur_yaw))

        if anchor_err > self._CORRECTION_THRESHOLD_M or yaw_err_abs > self._CORRECTION_YAW_THRESHOLD_RAD:
            correction_state.correcting = True
            correction_state.step = 0
            correction_state.start_x = cur_x
            correction_state.start_y = cur_y
            correction_state.start_yaw = cur_yaw
            correction_state.goal_x = anchor_x
            correction_state.goal_y = anchor_y
            correction_state.goal_yaw = anchor_yaw
            self._log(
                f"[Replay] periodic correction start step={step_idx} "
                f"err_xy={anchor_err:.3f}m err_yaw={math.degrees(yaw_err_abs):.1f}deg "
                f"frames={correction_frames}"
            )

    def _apply_correction_step(
        self,
        correction_state: CorrectionState,
        correction_frames: int,
        robot,
        cur_z: float,
        step_idx: int,
    ) -> Tuple[float, float, float, bool]:
        """Apply one step of smooth correction and return updated pose.

        Returns:
            (cur_x, cur_y, cur_yaw, correcting_this_step)
        """
        if not correction_state.correcting:
            return 0.0, 0.0, 0.0, False

        t = (correction_state.step + 1) / float(correction_frames)
        s = _min_jerk(t)
        cur_x = correction_state.start_x + (correction_state.goal_x - correction_state.start_x) * s
        cur_y = correction_state.start_y + (correction_state.goal_y - correction_state.start_y) * s
        cur_yaw = correction_state.start_yaw + _angle_diff_rad(correction_state.goal_yaw, correction_state.start_yaw) * s

        self._teleport_robot_xyyaw(robot, x=cur_x, y=cur_y, yaw_rad=cur_yaw, z_keep=cur_z)
        correction_state.step += 1

        if correction_state.step >= correction_frames:
            correction_state.correcting = False
            self._log(f"[Replay] periodic correction done step={step_idx}")

        return cur_x, cur_y, cur_yaw, True

    def _compute_control_commands(
        self,
        *,
        target_x: float,
        target_y: float,
        target_yaw: float,
        cur_x: float,
        cur_y: float,
        cur_yaw: float,
        v_ff: float,
        lat_ff: float,
        w_ff: float,
        kp_pos: float,
        kp_yaw: float,
        forward_scale: float,
        rot_scale: float,
        supports_lateral: bool,
        use_command_mode: bool,
        correcting_this_step: bool,
    ) -> Tuple[float, float, float]:
        """Compute raw control commands based on tracking error and feedforward."""
        dx = target_x - cur_x
        dy = target_y - cur_y

        cos_y = math.cos(cur_yaw)
        sin_y = math.sin(cur_yaw)
        err_fwd = dx * cos_y + dy * sin_y
        err_lat = -dx * sin_y + dy * cos_y

        desired_yaw = target_yaw
        if not supports_lateral:
            steer = math.atan2(err_lat, max(abs(err_fwd), 1e-3))
            desired_yaw = target_yaw + steer
            lat_ff = 0.0
            err_lat = 0.0

        yaw_err = _angle_diff_rad(desired_yaw, cur_yaw)
        gain_scale = 0.2 if correcting_this_step else 1.0

        if use_command_mode:
            v_correction = (kp_pos * gain_scale) * err_fwd / forward_scale
            lat_correction = (kp_pos * gain_scale) * err_lat / forward_scale
            w_correction = (kp_yaw * gain_scale) * yaw_err / rot_scale
            v_cmd_raw = v_ff + v_correction
            lat_cmd_raw = lat_ff + lat_correction
            w_cmd_raw = w_ff + w_correction
        else:
            v_des = v_ff + (kp_pos * gain_scale) * err_fwd
            lat_des = lat_ff + (kp_pos * gain_scale) * err_lat
            w_des = w_ff + (kp_yaw * gain_scale) * yaw_err
            v_cmd_raw = v_des / forward_scale
            lat_cmd_raw = lat_des / forward_scale
            w_cmd_raw = w_des / rot_scale

        return v_cmd_raw, lat_cmd_raw, w_cmd_raw

    def _capture_frame_group(
        self,
        cameras: List[ReplayCamera],
        camera_outputs: Dict[str, ReplayCameraOutput],
        frame_idx: int,
        step_idx: int,
        elapsed: float,
        on_capture: Optional[Callable[[int, int, float, Any], None]],
        video_writer: Optional[MultiCameraAsyncVideoWriter] = None,
        robot=None,
    ) -> None:
        frames: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
        for replay_camera in cameras:
            camera = replay_camera.camera

            depth = None
            if hasattr(camera, "get_distance_to_image_plane"):
                try:
                    depth = camera.get_distance_to_image_plane()
                except Exception as e:
                    self._log(f"[Replay] Warning: failed to get depth for {replay_camera.name}: {e}")
                    depth = None

            rgba = camera.get_rgba()
            if rgba is None:
                raise RuntimeError(f"Camera '{replay_camera.name}'.get_rgba() returned None during capture")
            rgb = rgba[:, :, :3]
            frames[replay_camera.name] = (rgb, depth)

            if on_capture is not None:
                on_capture(step_idx, frame_idx, float(elapsed), camera)

        if video_writer is not None:
            metadata = self._build_video_frame_metadata(
                robot=robot,
                sim_time_s=float(elapsed),
                frame_idx=int(frame_idx),
                sim_step=int(step_idx) + 1,
            )
            video_writer.push(frame_idx, frames, metadata=metadata)
            return

        for camera_name, (rgb, depth) in frames.items():
            camera_output = camera_outputs[camera_name]
            self._save_png(camera_output.rgb / f"frame_{frame_idx:06d}.jpg", rgb)
            if depth is not None and camera_output.depth is not None:
                self._save_depth(camera_output.depth / f"frame_{frame_idx:06d}.png", depth)

    @staticmethod
    def _build_video_frame_metadata(
        *,
        robot,
        sim_time_s: float,
        frame_idx: int,
        sim_step: int,
    ) -> Optional[Dict[str, Any]]:
        if robot is None:
            return None

        ts = time.time()
        x, y, z, yaw = ReplayRunner._get_robot_pose_xyyaw(robot)
        return {
            "frame": int(frame_idx),
            "sim_step": int(sim_step),
            "timestamp": float(ts),
            "timestamp_ms": int(ts * 1000.0),
            "sim_time_s": float(sim_time_s),
            "pose": {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "yaw": float(yaw),
            },
        }

    def _create_multi_camera_video_writer(
        self,
        *,
        camera_outputs: Dict[str, ReplayCameraOutput],
        scenario_id: str,
        out_dir: Path,
        instruction: str,
    ) -> MultiCameraAsyncVideoWriter:
        path_dir = out_dir / "path"
        path_dir.mkdir(parents=True, exist_ok=True)
        writer_outputs = {
            name: {"rgb": output.rgb, "depth": output.depth}
            for name, output in camera_outputs.items()
        }
        writer = MultiCameraAsyncVideoWriter(
            camera_outputs=writer_outputs,
            fps=int(self.config.fps),
            recording_json_path=path_dir / "path.json",
            recording_instruction=str(instruction or ""),
        )
        self._log(
            f"[Replay][{scenario_id}] Multi-camera video writer started: cameras={list(camera_outputs.keys())}"
        )
        return writer

    def _replay_track(
        self,
        runner: "SimulatorRunner",
        robot,
        robot_name: str,
        controller,
        action_builder,
        forward_scale: float,
        rot_scale: float,
        traj: TrajectoryReplayer,
        cameras: List[ReplayCamera],
        camera_outputs: Dict[str, ReplayCameraOutput],
        dt: float,
        out_dir: Path,
        scenario_id: str,
        on_capture: Optional[Callable[[int, int, float, Any], None]] = None,
        on_step: Optional[Callable[[int, float], None]] = None,
        video_writer: Optional[MultiCameraAsyncVideoWriter] = None,
    ) -> None:
        ref_poses = traj.pose_sequence()

        owns_video_writer = False
        if video_writer is None and self.config.output_video:
            instruction = str(
                ((scenario.get("task") or {}).get("navigation") or {}).get("instruction")
                or (scenario.get("task") or {}).get("instruction")
                or ""
            )
            video_writer = self._create_multi_camera_video_writer(
                camera_outputs=camera_outputs,
                scenario_id=scenario_id,
                out_dir=out_dir,
                instruction=instruction,
            )
            owns_video_writer = True

        # Determine replay mode: command-based (preferred) or velocity-based (fallback)
        use_command_mode = traj.has_command()
        if use_command_mode:
            feedforward = traj.command_sequence()  # Normalized commands from recording
            self._log("[Replay] Using command-based replay (direct command feedforward)")
        else:
            feedforward = traj.velocities()  # Fallback to velocity-based
            self._log("[Replay] Using velocity-based replay (no commands in waypoints)")

        steps = len(feedforward)

        kp_pos = float(self.config.track_kp_pos)
        kp_yaw = float(self.config.track_kp_yaw)
        lookahead = max(0, int(self.config.track_lookahead_steps))
        alpha = _clamp(float(self.config.track_smoothing_alpha), 0.0, 1.0)

        supports_lateral = controller.__class__.__name__ not in {
            "DifferentialDriveController",
            "DifferentialDriveMoveBySpeedController",
        }
        clamp_inputs = hasattr(controller, "forward_speed") or hasattr(controller, "rotation_speed")

        fps_interval = 1.0 / float(self.config.fps)
        next_capture = 0.0
        elapsed = 0.0
        frame_idx = 0

        v_prev = 0.0
        lat_prev = 0.0
        w_prev = 0.0

        # Periodic correction parameters
        correction_period = max(1, int(self.config.track_correction_period))
        correction_frames = max(1, int(self.config.track_correction_frames))

        mode_str = "command" if use_command_mode else "velocity"
        self._log(
            f"[Replay] mode={mode_str}, steps={steps}, dt={dt:.6f}, "
            f"kp_pos={kp_pos:.3f}, kp_yaw={kp_yaw:.3f}, lookahead={lookahead}, "
            f"smooth_alpha={alpha:.3f}, correction_period={correction_period}, "
            f"correction_frames={correction_frames}"
        )

        # Periodic correction state
        correction_state = CorrectionState()

        for step_idx in range(steps):
            capture = elapsed >= next_capture - 1e-9

            target_idx = min(step_idx + lookahead, len(ref_poses) - 1)
            target_x, target_y, _, target_yaw = ref_poses[target_idx]
            anchor_x, anchor_y, _, anchor_yaw = ref_poses[min(step_idx, len(ref_poses) - 1)]

            cur_x, cur_y, cur_z, cur_yaw = self._get_robot_pose_xyyaw(robot)

            # Check if periodic correction is needed
            self._check_and_start_correction(
                correction_state, step_idx, correction_period,
                anchor_x, anchor_y, anchor_yaw,
                cur_x, cur_y, cur_yaw,
                correction_frames,
            )

            # Apply smooth correction if in progress
            correcting_this_step = False
            if correction_state.correcting:
                cur_x, cur_y, cur_yaw, correcting_this_step = self._apply_correction_step(
                    correction_state, correction_frames, robot, cur_z, step_idx
                )

            # Compute control commands
            v_ff, lat_ff, w_ff = feedforward[step_idx]
            v_cmd_raw, lat_cmd_raw, w_cmd_raw = self._compute_control_commands(
                target_x=target_x,
                target_y=target_y,
                target_yaw=target_yaw,
                cur_x=cur_x,
                cur_y=cur_y,
                cur_yaw=cur_yaw,
                v_ff=v_ff,
                lat_ff=lat_ff,
                w_ff=w_ff,
                kp_pos=kp_pos,
                kp_yaw=kp_yaw,
                forward_scale=forward_scale,
                rot_scale=rot_scale,
                supports_lateral=supports_lateral,
                use_command_mode=use_command_mode,
                correcting_this_step=correcting_this_step,
            )

            if clamp_inputs:
                v_cmd_raw = _clamp(v_cmd_raw, -1.0, 1.0)
                lat_cmd_raw = _clamp(lat_cmd_raw, -1.0, 1.0)
                w_cmd_raw = _clamp(w_cmd_raw, -1.0, 1.0)

            # Apply smoothing (less important in command mode since commands are already intended)
            v_cmd = alpha * v_cmd_raw + (1.0 - alpha) * v_prev
            lat_cmd = alpha * lat_cmd_raw + (1.0 - alpha) * lat_prev
            w_cmd = alpha * w_cmd_raw + (1.0 - alpha) * w_prev
            v_prev, lat_prev, w_prev = v_cmd, lat_cmd, w_cmd

            action_tuple = action_builder(v_cmd, lat_cmd, w_cmd)
            actions = [{robot_name: {"move_by_speed": action_tuple}}]
            runner.step(actions=actions, render=True)

            if on_step is not None:
                # Fail-fast: hook errors indicate inconsistent data capture/analysis.
                on_step(step_idx, float(elapsed))

            if capture:
                self._capture_frame_group(
                    cameras, camera_outputs, frame_idx, step_idx, elapsed, on_capture,
                    video_writer=video_writer, robot=robot,
                )
                frame_idx += 1
                next_capture += fps_interval
            elapsed += dt

            if step_idx % 500 == 0:
                dx = target_x - cur_x
                dy = target_y - cur_y
                err_xy = (dx * dx + dy * dy) ** 0.5
                self._log(
                    f"[Replay] step {step_idx}/{steps} elapsed={elapsed:.2f}s err_xy={err_xy:.3f} "
                    f"v={v_cmd:.3f} w={w_cmd:.3f}"
                )

        # Close video writer
        if video_writer is not None and owns_video_writer:
            video_writer.close()
            self._log(f"[Replay][{scenario_id}] Video writer closed, {frame_idx} frames written")

        self._log(f"[Replay] Finished scenario {scenario_id}: saved {frame_idx} frames")

    def _replay_teleport(
        self,
        runner: "SimulatorRunner",
        robot,
        scenario: Dict[str, Any],
        traj: TrajectoryReplayer,
        cameras: List[ReplayCamera],
        camera_outputs: Dict[str, ReplayCameraOutput],
        dt: float,
        unit_scale: float,
        out_dir: Path,
        scenario_id: str,
    ) -> None:
        """Replay by pure teleport — no physics controller, no walking animation.

        Designed for robots (e.g. H1 humanoid) that fall easily under
        physics-based replay.  Each frame: teleport base to the interpolated
        pose, reset joints to standing posture, zero all velocities, step the
        simulator (for rendering / camera update), and capture.
        """
        self._log(
            f"[Replay][{scenario_id}] teleport pose reconstruction uses frame-based interpolation; "
            f"time_s is treated as auxiliary metadata only"
        )
        ref_poses = traj.pose_sequence(validate_timing=False)
        steps = len(ref_poses)

        # Video writer
        video_writer: Optional[MultiCameraAsyncVideoWriter] = None
        if self.config.output_video:
            instruction = str(
                ((scenario.get("task") or {}).get("navigation") or {}).get("instruction")
                or (scenario.get("task") or {}).get("instruction")
                or ""
            )
            video_writer = self._create_multi_camera_video_writer(
                camera_outputs=camera_outputs,
                scenario_id=scenario_id,
                out_dir=out_dir,
                instruction=instruction,
            )

        articulation = getattr(robot, "articulation", None)

        standing_joints = self._teleport_joint_reset_positions(robot=robot, scenario=scenario)

        # Set articulation to kinematic and disable gravity so joints don't slump.
        # This prevents the robot from sinking during replay.
        # Unwrap to get the real Isaac Sim articulation (IsaacsimArticulation is a wrapper).
        _real_art = articulation.unwrap() if hasattr(articulation, 'unwrap') else articulation
        _kinematic_set = False
        if _real_art is not None:
            try:
                import omni.usd
                from pxr import UsdPhysics
                stage = omni.usd.get_context().get_stage()
                root_prim = stage.GetPrimAtPath(str(_real_art.prim_path))
                if root_prim.IsValid():
                    # Set root body to kinematic
                    rb_api = UsdPhysics.RigidBodyAPI(root_prim)
                    rb_api.GetKinematicEnabledAttr().Set(True)
                    _kinematic_set = True
                    self._log(f"[Replay][{scenario_id}] Set articulation to kinematic mode")
            except Exception as e:
                self._log(f"[Replay][{scenario_id}] Warning: failed to set kinematic: {e}")

            # Disable gravity on the entire articulation
            try:
                _real_art.disable_gravity()
                self._log(f"[Replay][{scenario_id}] Disabled gravity on articulation")
            except Exception as e:
                self._log(f"[Replay][{scenario_id}] Warning: failed to disable gravity: {e}")

        fps_interval = 1.0 / float(self.config.fps)
        next_capture = 0.0
        elapsed = 0.0
        frame_idx = 0

        # Use initial z from the first waypoint; ignore per-frame z (may be incorrect).
        init_z = self._resolve_replay_fixed_z(
            scenario=scenario,
            waypoints=traj._wps,
            unit_scale=float(unit_scale),
            articulation=articulation,
        )
        self._log(f"[Replay][{scenario_id}] Using fixed z={init_z:.4f} from replay height policy (ignoring per-frame z)")

        self._log(
            f"[Replay][{scenario_id}] teleport mode (kinematic): steps={steps}, dt={dt:.6f}, "
            f"fps={self.config.fps}"
        )

        for step_idx in range(steps):
            x, y, _z, yaw = ref_poses[step_idx]

            # 1. Teleport base — use fixed init_z instead of waypoint z
            self._teleport_robot_xyyaw(robot, x, y, yaw, z_keep=init_z)

            # 2. Reset velocities and joints to a stable fixed pose
            if _real_art is not None:
                try:
                    _real_art.set_linear_velocity(np.zeros(3))
                    _real_art.set_angular_velocity(np.zeros(3))
                    _real_art.set_joint_velocities(np.zeros(_real_art.num_dof))
                    if standing_joints is not None and _real_art.num_dof == len(standing_joints):
                        _real_art.set_joint_positions(standing_joints)
                except Exception as e:
                    if step_idx == 0:
                        self._log(f"[Replay] Warning: failed to reset robot state: {e}")

            # 3. Step simulation — kinematic body won't fall, but pose syncs to USD
            runner.step(actions=[], render=True)

            # 4. Capture at configured fps
            capture = elapsed >= next_capture - 1e-9
            if capture:
                self._capture_frame_group(
                    cameras, camera_outputs, frame_idx, step_idx, elapsed, None,
                    video_writer=video_writer, robot=robot,
                )
                frame_idx += 1
                next_capture += fps_interval
            elapsed += dt

            if step_idx % 500 == 0:
                self._log(
                    f"[Replay] step {step_idx}/{steps} elapsed={elapsed:.2f}s "
                    f"pos=({x:.3f}, {y:.3f}, {init_z:.3f}) yaw={math.degrees(yaw):.1f}deg"
                )

        if video_writer is not None:
            video_writer.close()
            self._log(f"[Replay][{scenario_id}] Video writer closed, {frame_idx} frames written")

        self._log(f"[Replay] Finished scenario {scenario_id} (teleport): saved {frame_idx} frames")

    def _teleport_joint_reset_positions(
        self,
        *,
        robot,
        scenario: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        """Return a fixed joint pose for teleport replay, matched to robot type."""
        robot_type = ""
        try:
            entries = scenario.get("robots", {}).get("entries", [])
            if entries and isinstance(entries[0], dict):
                robot_type = str(entries[0].get("type") or "").strip().lower()
        except Exception:
            robot_type = ""

        articulation = getattr(robot, "articulation", None)
        real_art = articulation.unwrap() if hasattr(articulation, "unwrap") else articulation
        num_dof = int(getattr(real_art, "num_dof", 0) or 0)

        if robot_type in ("h1", "h1robot"):
            joints = np.array(
                [
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    -0.4, -0.4, 0.0, 0.0, 0.8, 0.8,
                    0.0, 0.0, -0.4, -0.4, 0.0, 0.0,
                ],
                dtype=np.float32,
            )
            if num_dof == len(joints):
                self._log(f"[Replay] Teleport joint reset pose: H1 standing pose ({len(joints)} dof)")
                return joints
            self._log(f"[Replay] H1 standing pose skipped: articulation dof={num_dof}, expected={len(joints)}")
            return None

        if robot_type in ("aliengo", "aliengorobot"):
            joints = np.array(
                [0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 0.8, 0.8, -1.5, -1.5, -1.5, -1.5],
                dtype=np.float32,
            )
            if num_dof == len(joints):
                self._log(f"[Replay] Teleport joint reset pose: Aliengo default stance ({len(joints)} dof)")
                return joints
            self._log(
                f"[Replay] Aliengo default stance skipped: articulation dof={num_dof}, expected={len(joints)}"
            )
            return None

        if robot_type == "carter_v1":
            if num_dof > 0:
                self._log(f"[Replay] Teleport joint reset pose: Carter zero joint pose ({num_dof} dof)")
                return np.zeros(num_dof, dtype=np.float32)
            self._log("[Replay] Carter zero joint pose skipped: articulation dof unavailable")
            return None

        if num_dof > 0:
            self._log(f"[Replay] Teleport joint reset pose: generic zero joint pose ({num_dof} dof)")
            return np.zeros(num_dof, dtype=np.float32)
        return None

    def _replay_move_along_path(
        self,
        *,
        runner: "SimulatorRunner",
        robot,
        robot_name: str,
        move_along_path,
        move_by_speed,
        waypoints: List[Dict[str, Any]],
        unit_scale: float,
        steps: int,
        dt: float,
        cameras: List[ReplayCamera],
        camera_outputs: Dict[str, ReplayCameraOutput],
        out_dir: Path,
        scenario: Dict[str, Any],
        scenario_id: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        # Ensure robot starts at the episode pose before recording.
        self._maybe_snap_robot_to_first_waypoint(
            scenario=scenario,
            waypoints=waypoints,
            unit_scale=float(unit_scale),
            robot=robot,
        )

        video_writer: Optional[MultiCameraAsyncVideoWriter] = None
        if self.config.output_video:
            instruction = str(
                ((scenario.get("task") or {}).get("navigation") or {}).get("instruction")
                or (scenario.get("task") or {}).get("instruction")
                or ""
            )
            video_writer = self._create_multi_camera_video_writer(
                camera_outputs=camera_outputs,
                scenario_id=scenario_id,
                out_dir=out_dir,
                instruction=instruction,
            )

        # Recorder init
        distance_thresh = float(self._writeback_distance_env_units(unit_scale=float(unit_scale)))
        forward_scale, rot_scale = self._extract_controller_scales(move_by_speed)

        rb_waypoints: List[Dict[str, Any]] = []
        vh_waypoints: Dict[str, List[Dict[str, Any]]] = {}

        # Virtual human setup (optional)
        vh_cfg = scenario.get("virtual_humans") if isinstance(scenario.get("virtual_humans"), dict) else {}
        vh_count = 0
        try:
            vh_count = int(vh_cfg.get("count") or 0)
        except Exception:
            vh_count = 0
        vh_names = []
        if vh_count > 0:
            seq = vh_cfg.get("name_sequence") or []
            vh_names = [str(n) for n in seq if n]
            if not vh_names:
                vh_names = [f"Character_{i}" for i in range(vh_count)]
            vh_waypoints = {name: [] for name in vh_names}

        # Build a wait plan from large frame gaps in the original waypoint stream.
        # This preserves "pause/wait" segments without writing stationary waypoints back.
        wait_plan = self._build_wait_plan_from_waypoints(
            waypoints=waypoints,
            dt=float(dt),
            min_wait_s=float(self._WAIT_MIN_S),
            yaw_skip_deg=float(self._WAIT_YAW_SKIP_DEG),
            scenario_id=str(scenario_id),
        )

        # Use the full recorded waypoint stream as path points (no sparsification).
        path_points = self._build_path_points_from_waypoints(waypoints, unit_scale=float(unit_scale))

        # Initial samples (frame=0)
        elapsed = 0.0
        next_capture = 0.0
        fps_interval = 1.0 / float(self.config.fps)
        frame_idx = 0

        rb_state = {
            "last_logged": None,  # (x_env, y_env, yaw_rad)
            "total_xy": 0.0,
            "last_sample": None,  # (x_env, y_env)
            "since_last_log": 0.0,
            "dt": float(dt),
        }
        vh_state: Dict[str, Dict[str, Any]] = {
            name: {
                "last_logged": None,
                "total_xy": 0.0,
                "last_sample": None,
                "since_last_log": 0.0,
                "dt": float(dt),
            }
            for name in vh_waypoints.keys()
        }

        self._record_robot_waypoint(
            out=rb_waypoints,
            state=rb_state,
            robot=robot,
            frame=0,
            time_s=0.0,
            unit_scale=float(unit_scale),
            forward_scale=float(forward_scale),
            rot_scale=float(rot_scale),
        )

        if vh_waypoints:
            try:
                from OmniNavExt.envset.agent_manager import AgentManager

                mgr = AgentManager.get_instance()
            except Exception:
                mgr = None
            for name in vh_waypoints.keys():
                self._record_virtual_human_waypoint(
                    out=vh_waypoints[name],
                    state=vh_state[name],
                    mgr=mgr,
                    agent_name=name,
                    frame=0,
                    time_s=0.0,
                    unit_scale=float(unit_scale),
                )
        else:
            mgr = None

        # IMPORTANT: do NOT stop based on recorded frame count. Recorded paths can be fully reachable
        # but replay dynamics may take longer than the original timeline. We run until controller reports finished.
        total_wait_frames = int(sum(int(v) for v in wait_plan.values()))
        max_steps = max(
            int(steps) * self._MAX_STEPS_MULTIPLIER,
            int(steps) + total_wait_frames + self._MAX_STEPS_BUFFER,
            self._MAX_STEPS_MIN
        )
        self._log(f"[Replay][{scenario_id}] move_along_path max_steps={max_steps} (recorded_steps={int(steps)})")

        # Goal (optional; used only for diagnostics)
        goal_xy_stage = None
        try:
            nav = ((scenario.get("task") or {}).get("navigation") or {}) if isinstance(scenario.get("task"), dict) else {}
            goal = nav.get("goal_position")
            if isinstance(goal, (list, tuple)) and len(goal) >= 2:
                # goal_position is in env units; convert to stage units (meters).
                goal_xy_stage = (float(goal[0]) * float(unit_scale), float(goal[1]) * float(unit_scale))
        except Exception:
            goal_xy_stage = None

        waiting_remaining = int(wait_plan.get(0, 0))
        waited_indices = set()
        if waiting_remaining > 0:
            waited_indices.add(0)
            self._log(f"[Replay][{scenario_id}] wait@wp_idx=0 frames={waiting_remaining}")

        last_index = None
        stall_steps = 0
        step_idx = 0
        while step_idx < int(max_steps):
            capture = elapsed >= next_capture - 1e-9
            in_wait = waiting_remaining > 0
            if in_wait:
                actions = [{robot_name: {"move_by_speed": (0.0, 0.0, 0.0)}}]
                runner.step(actions=actions, render=True)
                waiting_remaining -= 1
            else:
                actions = [{robot_name: {"move_along_path": [path_points]}}]
                runner.step(actions=actions, render=True)

            cur_frame = int(step_idx) + 1
            cur_time_s = float(cur_frame) * float(dt)

            # Robot sampling + thresholded logging
            self._maybe_record_robot_waypoint(
                out=rb_waypoints,
                state=rb_state,
                robot=robot,
                frame=cur_frame,
                time_s=cur_time_s,
                unit_scale=float(unit_scale),
                forward_scale=float(forward_scale),
                rot_scale=float(rot_scale),
                distance_thresh=float(distance_thresh),
            )

            # Virtual human sampling + thresholded logging
            if vh_waypoints:
                for name in vh_waypoints.keys():
                    self._maybe_record_virtual_human_waypoint(
                        out=vh_waypoints[name],
                        state=vh_state[name],
                        mgr=mgr,
                        agent_name=name,
                        frame=cur_frame,
                        time_s=cur_time_s,
                        unit_scale=float(unit_scale),
                        distance_thresh=float(distance_thresh),
                    )

            if capture:
                self._capture_frame_group(
                    cameras=cameras,
                    camera_outputs=camera_outputs,
                    frame_idx=frame_idx,
                    step_idx=step_idx,
                    elapsed=elapsed,
                    on_capture=None,
                    video_writer=video_writer,
                    robot=robot,
                )
                frame_idx += 1
                next_capture += fps_interval

            elapsed += float(dt)

            # Track controller progress (do not terminate by frames; terminate by finished).
            cur_index = -1
            finished = False
            if not in_wait:
                try:
                    obs = move_along_path.get_obs()
                    cur_index = int(obs.get("current_index", -1)) if isinstance(obs, dict) else -1
                    finished = bool(obs.get("finished", False)) if isinstance(obs, dict) else False
                except Exception:
                    cur_index = -1
                    finished = False

                # Schedule waits only when controller index advances (meaning previous point was reached).
                if last_index is None:
                    last_index = cur_index
                    stall_steps = 0
                else:
                    if cur_index != last_index:
                        if cur_index > last_index:
                            reached_wp = int(cur_index) - 1
                            reached_wp = max(0, min(reached_wp, len(path_points) - 1))
                            frames = int(wait_plan.get(reached_wp, 0))
                            if frames > 0 and int(reached_wp) not in waited_indices:
                                waiting_remaining = frames
                                waited_indices.add(int(reached_wp))
                                self._log(f"[Replay][{scenario_id}] wait@wp_idx={reached_wp} frames={frames}")
                        stall_steps = 0
                        last_index = cur_index
                    else:
                        stall_steps += 1

                if finished:
                    self._log(f"[Replay][{scenario_id}] move_along_path finished at step={step_idx} frame={cur_frame}")
                    break

            if step_idx % 500 == 0:
                try:
                    rx, ry, _, _ = self._get_robot_pose_xyyaw(robot)
                    msg = f"[Replay] step {step_idx}/{max_steps} t={elapsed:.2f}s idx={cur_index} robot_xy=({rx:.2f},{ry:.2f})"
                    if goal_xy_stage is not None:
                        gx, gy = goal_xy_stage
                        msg += f" dist_to_goal={((rx-gx)**2+(ry-gy)**2)**0.5:.3f}m"
                    if in_wait:
                        msg += f" WAIT(rem={waiting_remaining})"
                    self._log(msg)
                except Exception as e:
                    self._log(f"[Replay] step {step_idx}/{max_steps} failed to get pose: {e}")
            # Stall warning: controller index hasn't advanced for a while.
            if (not in_wait) and stall_steps > 0 and stall_steps % 720 == 0:
                self._log(
                    f"[Replay][{scenario_id}] WARNING: move_along_path stalled "
                    f"index={cur_index} for {stall_steps} steps (~{stall_steps * float(dt):.1f}s)"
                )

            STALL_ABORT_THRESHOLD = 1500  # ~25 s of no progress
            if stall_steps > STALL_ABORT_THRESHOLD:
                self._log(f"[Replay][{scenario_id}] ABORT: stalled for {stall_steps} steps, breaking")
                break

            step_idx += 1

        else:
            # Close video writer before raising error
            if video_writer is not None:
                video_writer.close()
            raise RuntimeError(
                f"[Replay][{scenario_id}] move_along_path did not finish within max_steps={max_steps}; "
                f"last_index={last_index}"
            )

        # Close video writer
        if video_writer is not None:
            video_writer.close()
            self._log(f"[Replay][{scenario_id}] Video writer closed, {frame_idx} frames written")

        return rb_waypoints, vh_waypoints

    @classmethod
    def _build_wait_plan_from_waypoints(
        cls,
        *,
        waypoints: List[Dict[str, Any]],
        dt: float,
        min_wait_s: float,
        yaw_skip_deg: float,
        scenario_id: str,
    ) -> Dict[int, int]:
        """Infer wait segments from large frame gaps, excluding pure yaw-adjust segments.

        Returns a dict: reached_waypoint_index -> wait_frames_to_inject.
        When the path controller advances to index i+1 (meaning it reached i), we inject wait_plan[i] frames.
        """
        import statistics

        wps = sorted([wp for wp in waypoints if isinstance(wp, dict)], key=lambda wp: int(wp.get("frame", 0)))
        if len(wps) < 2:
            return {}

        gaps: List[int] = []
        for a, b in zip(wps[:-1], wps[1:]):
            try:
                gap = int(b.get("frame", 0)) - int(a.get("frame", 0))
            except Exception:
                continue
            if gap > 0:
                gaps.append(gap)
        if not gaps:
            return {}

        dt_s = max(1e-9, float(dt))
        min_wait_s = max(0.0, float(min_wait_s))

        # Estimate a "normal" gap between movement-triggered waypoints (in frames) from gaps below wait threshold.
        # This is used only to approximate the non-wait baseline duration, not to decide whether a gap is a wait.
        non_wait_gaps = [int(g) for g in gaps if (float(g) * dt_s) < float(min_wait_s)]
        if non_wait_gaps:
            base_gap_frames = max(1, int(round(statistics.median(non_wait_gaps))))
        else:
            base_gap_frames = 1

        plan: Dict[int, int] = {}
        candidates = 0
        for i, (a, b) in enumerate(zip(wps[:-1], wps[1:])):
            try:
                gap = int(b.get("frame", 0)) - int(a.get("frame", 0))
            except Exception:
                continue
            if gap <= 0 or (float(gap) * dt_s) < float(min_wait_s):
                continue
            candidates += 1

            yaw0 = a.get("yaw_deg")
            yaw1 = b.get("yaw_deg")
            dyaw = 0.0
            try:
                dyaw = float(yaw1) - float(yaw0)
                dyaw = (dyaw + 180.0) % 360.0 - 180.0
            except Exception:
                dyaw = 0.0

            if abs(float(dyaw)) > float(yaw_skip_deg):
                continue

            wait_frames = max(0, int(gap) - int(base_gap_frames))
            if wait_frames > 0:
                plan[int(i)] = int(wait_frames)

        try:
            cls._log(
                f"[Replay][{scenario_id}] wait_plan: dt={dt_s:.6f}s min_wait_s={min_wait_s:.3f} "
                f"base_gap_frames={base_gap_frames} candidates={candidates} waits={len(plan)} "
                f"total_wait_frames={sum(plan.values())}"
            )
        except Exception:
            pass

        return plan

    @staticmethod
    def _scenario_has_initial_pose(scenario: Dict[str, Any]) -> bool:
        try:
            entries = scenario.get("robots", {}).get("entries", [])
            if not entries or not isinstance(entries, list) or not isinstance(entries[0], dict):
                return False
            init_pose = entries[0].get("initial_pose") or {}
            if not isinstance(init_pose, dict):
                return False
            has_pos = isinstance(init_pose.get("position"), (list, tuple)) and len(init_pose.get("position")) >= 2
            has_yaw = init_pose.get("orientation_deg") is not None
            return bool(has_pos or has_yaw)
        except Exception:
            return False

    @staticmethod
    def _build_path_points_from_waypoints(waypoints: List[Dict[str, Any]], unit_scale: float) -> List[List[float]]:
        wps = sorted([wp for wp in waypoints if isinstance(wp, dict)], key=lambda wp: int(wp.get("frame", 0)))
        pts: List[List[float]] = []
        for wp in wps:
            xyz = wp.get("xyz") or (0.0, 0.0, 0.0)
            try:
                x = float(xyz[0]) * float(unit_scale)
                y = float(xyz[1]) * float(unit_scale)
            except Exception:
                continue
            pts.append([float(x), float(y), 0.0])
        if not pts:
            raise ValueError("No valid xyz points found in replay waypoints for move_along_path replay")
        return pts

    def _snap_robot_to_first_waypoint(self, *, robot, waypoints: List[Dict[str, Any]], unit_scale: float) -> None:
        wp0 = sorted(waypoints, key=lambda wp: int(wp.get("frame", 0)))[0]
        xyz0 = wp0.get("xyz", (0.0, 0.0, 0.0))
        x0 = float(xyz0[0]) * float(unit_scale)
        y0 = float(xyz0[1]) * float(unit_scale)
        z0_logged = float(xyz0[2]) * float(unit_scale)
        yaw0 = math.radians(float(wp0.get("yaw_deg", 0.0)))
        quat0 = (math.cos(yaw0 / 2.0), 0.0, 0.0, math.sin(yaw0 / 2.0))
        articulation = getattr(robot, "articulation", None)
        if articulation is None:
            self._log("[Replay] Warning: robot has no articulation, cannot snap to first waypoint")
            return

        z_target = get_articulation_z(articulation)
        if z_target is None:
            z_target = z0_logged
            self._log(f"[Replay] Warning: could not get articulation Z, using fallback z={z_target}")
        articulation.set_world_pose((x0, y0, z_target), quat0)

    @staticmethod
    def _stage_xyyaw_to_env(x: float, y: float, z: float, yaw_rad: float, unit_scale: float) -> Tuple[float, float, float, float]:
        inv = 1.0 / float(unit_scale) if float(unit_scale) != 0 else 1.0
        return float(x) * inv, float(y) * inv, float(z) * inv, float(yaw_rad)

    def _record_robot_waypoint(
        self,
        *,
        out: List[Dict[str, Any]],
        state: Dict[str, Any],
        robot,
        frame: int,
        time_s: float,
        unit_scale: float,
        forward_scale: float,
        rot_scale: float,
    ) -> None:
        x_m, y_m, z_m, yaw_rad = self._get_robot_pose_xyyaw(robot)
        x_env, y_env, _, _ = self._stage_xyyaw_to_env(x_m, y_m, 0.0, yaw_rad, unit_scale=float(unit_scale))

        prev = state.get("last_logged")
        if prev is None:
            dist_xy = 0.0
            total_xy = 0.0
            cmd_v = 0.0
            cmd_w = 0.0
        else:
            px, py, pyaw = prev
            dx = x_env - float(px)
            dy = y_env - float(py)
            dist_xy = float((dx * dx + dy * dy) ** 0.5)
            total_xy = float(state.get("total_xy", 0.0)) + dist_xy

            df = int(frame) - int(out[-1]["frame"])
            dt_seg = float(df) * float(state.get("dt", 0.0) or 0.0)
            if dt_seg <= 0:
                dt_seg = 1e-6
            # Velocity in stage units (meters/sec)
            dx_m = dx * float(unit_scale)
            dy_m = dy * float(unit_scale)
            vx = dx_m / dt_seg
            vy = dy_m / dt_seg
            forward_mps = vx * math.cos(float(yaw_rad)) + vy * math.sin(float(yaw_rad))
            w_rps = _angle_diff_rad(float(yaw_rad), float(pyaw)) / dt_seg
            cmd_v = forward_mps / float(forward_scale) if float(forward_scale) != 0 else 0.0
            cmd_w = w_rps / float(rot_scale) if float(rot_scale) != 0 else 0.0
            cmd_v = _clamp(cmd_v, -1.0, 1.0)
            cmd_w = _clamp(cmd_w, -1.0, 1.0)

        state["last_logged"] = (float(x_env), float(y_env), float(yaw_rad))
        state["last_sample"] = (float(x_env), float(y_env))
        state["since_last_log"] = 0.0
        state["total_xy"] = float(total_xy)

        out.append(
            {
                "frame": int(frame),
                "xyz": [float(x_env), float(y_env), 0.0],
                "yaw_deg": float(math.degrees(float(yaw_rad))),
                "time_s": float(time_s),
                "distance_xy": float(dist_xy),
                "distance_total_xy": float(total_xy),
                "command": {"v": float(cmd_v), "w": float(cmd_w), "lateral": 0.0},
            }
        )

    def _maybe_record_robot_waypoint(
        self,
        *,
        out: List[Dict[str, Any]],
        state: Dict[str, Any],
        robot,
        frame: int,
        time_s: float,
        unit_scale: float,
        forward_scale: float,
        rot_scale: float,
        distance_thresh: float,
    ) -> None:
        x_m, y_m, z_m, yaw_rad = self._get_robot_pose_xyyaw(robot)
        x_env, y_env, _, _ = self._stage_xyyaw_to_env(x_m, y_m, 0.0, yaw_rad, unit_scale=float(unit_scale))

        last_sample = state.get("last_sample")
        if last_sample is not None:
            dxs = x_env - float(last_sample[0])
            dys = y_env - float(last_sample[1])
            state["since_last_log"] = float(state.get("since_last_log", 0.0)) + float((dxs * dxs + dys * dys) ** 0.5)
        state["last_sample"] = (float(x_env), float(y_env))

        if float(state.get("since_last_log", 0.0)) < float(distance_thresh):
            return

        self._record_robot_waypoint(
            out=out,
            state=state,
            robot=robot,
            frame=int(frame),
            time_s=float(time_s),
            unit_scale=float(unit_scale),
            forward_scale=float(forward_scale),
            rot_scale=float(rot_scale),
        )

    def _record_virtual_human_waypoint(
        self,
        *,
        out: List[Dict[str, Any]],
        state: Dict[str, Any],
        mgr,
        agent_name: str,
        frame: int,
        time_s: float,
        unit_scale: float,
    ) -> None:
        if mgr is None:
            return
        pos = mgr.get_agent_position(str(agent_name))
        if pos is None:
            return
        x_m, y_m, z_m = float(pos[0]), float(pos[1]), float(pos[2])
        inv = 1.0 / float(unit_scale) if float(unit_scale) != 0 else 1.0
        x_env, y_env = x_m * inv, y_m * inv

        prev = state.get("last_logged")
        if prev is None:
            dist_xy = 0.0
            total_xy = 0.0
        else:
            px, py = prev
            dx = x_env - float(px)
            dy = y_env - float(py)
            dist_xy = float((dx * dx + dy * dy) ** 0.5)
            total_xy = float(state.get("total_xy", 0.0)) + dist_xy

        state["last_logged"] = (float(x_env), float(y_env))
        state["last_sample"] = (float(x_env), float(y_env))
        state["since_last_log"] = 0.0
        state["total_xy"] = float(total_xy)

        out.append(
            {
                "frame": int(frame),
                "xyz": [float(x_env), float(y_env), 0.0],
                "time_s": float(time_s),
                "distance_xy": float(dist_xy),
                "distance_total_xy": float(total_xy),
            }
        )

    def _maybe_record_virtual_human_waypoint(
        self,
        *,
        out: List[Dict[str, Any]],
        state: Dict[str, Any],
        mgr,
        agent_name: str,
        frame: int,
        time_s: float,
        unit_scale: float,
        distance_thresh: float,
    ) -> None:
        if mgr is None:
            return
        pos = mgr.get_agent_position(str(agent_name))
        if pos is None:
            return
        inv = 1.0 / float(unit_scale) if float(unit_scale) != 0 else 1.0
        x_env, y_env = float(pos[0]) * inv, float(pos[1]) * inv

        last_sample = state.get("last_sample")
        if last_sample is not None:
            dxs = x_env - float(last_sample[0])
            dys = y_env - float(last_sample[1])
            state["since_last_log"] = float(state.get("since_last_log", 0.0)) + float((dxs * dxs + dys * dys) ** 0.5)
        state["last_sample"] = (float(x_env), float(y_env))

        if float(state.get("since_last_log", 0.0)) < float(distance_thresh):
            return
        state["since_last_log"] = 0.0
        self._record_virtual_human_waypoint(
            out=out,
            state=state,
            mgr=mgr,
            agent_name=str(agent_name),
            frame=int(frame),
            time_s=float(time_s),
            unit_scale=float(unit_scale),
        )

    @staticmethod
    def _writeback_waypoints(
        *,
        envset_path: Path,
        scenario_id: str,
        gt_path: List[Dict[str, Any]],
        vh_gt_waypoints: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> None:
        import json
        import shutil
        from datetime import datetime

        envset_path = Path(envset_path)

        # Backup original file before overwriting
        if envset_path.exists():
            # Only create a backup once per envset file. Subsequent writebacks reuse the first backup.
            existing_backups = sorted(envset_path.parent.glob(f"{envset_path.stem}.*.backup.json"))
            if not existing_backups:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = envset_path.with_suffix(f".{timestamp}.backup.json")
                shutil.copy2(envset_path, backup_path)
                print(f"[Replay] Backed up original envset to: {backup_path}", flush=True)

        data = json.loads(envset_path.read_text(encoding="utf-8"))
        scenarios = data.get("scenarios") or []
        if not isinstance(scenarios, list):
            raise ValueError("Envset JSON has no 'scenarios' list; cannot write back waypoints")

        target = None
        for s in scenarios:
            if isinstance(s, dict) and str(s.get("id")) == str(scenario_id):
                target = s
                break
        if target is None:
            raise ValueError(f"Scenario id not found in envset for writeback: {scenario_id}")

        robots = target.setdefault("robots", {})
        entries = robots.get("entries") or []
        if not entries or not isinstance(entries, list) or not isinstance(entries[0], dict):
            raise ValueError("Scenario robots.entries[0] missing; cannot write back recording.gt_path")
        entries[0].pop("rb_gt_waypoints", None)
        target_recording = build_recording_payload(
            instruction=str(
                ((target.get("task") or {}).get("navigation") or {}).get("instruction")
                or (target.get("task") or {}).get("instruction")
                or ""
            ),
            gt_path=gt_path,
            metadata={"source": "replay_writeback"},
        )
        target[CANONICAL_RECORDING_KEY] = target_recording

        if vh_gt_waypoints:
            vh = target.setdefault("virtual_humans", {})
            if isinstance(vh, dict):
                vh["vh_gt_waypoints"] = vh_gt_waypoints

        envset_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_recording_sidecar(envset_path, target_recording)

    @staticmethod
    def _get_robot_pose_xyyaw(robot) -> Tuple[float, float, float, float]:
        pos = None
        quat = None
        try:
            pos, quat = robot.get_pose()
        except Exception:
            pos = None
            quat = None

        if pos is None or quat is None:
            articulation = getattr(robot, "articulation", None)
            if articulation is not None:
                try:
                    pos, quat = articulation.get_world_pose()
                except Exception:
                    try:
                        pos_arr, quat_arr = articulation.get_world_poses()
                        pos = pos_arr[0]
                        quat = quat_arr[0]
                    except Exception:
                        pos = None
                        quat = None

        if pos is None or quat is None:
            raise RuntimeError("Failed to get robot pose for tracking replay")

        pos_tuple = tuple(pos.tolist()) if hasattr(pos, "tolist") else tuple(pos)
        quat_tuple = tuple(quat.tolist()) if hasattr(quat, "tolist") else tuple(quat)

        x, y, z = float(pos_tuple[0]), float(pos_tuple[1]), float(pos_tuple[2])
        yaw = _quat_wxyz_to_yaw_rad(
            (float(quat_tuple[0]), float(quat_tuple[1]), float(quat_tuple[2]), float(quat_tuple[3]))
        )
        return x, y, z, yaw

    def _teleport_robot_xyyaw(self, robot, x: float, y: float, yaw_rad: float, z_keep: Optional[float] = None) -> None:
        articulation = getattr(robot, "articulation", None)
        if articulation is None:
            self._log("[Replay] Warning: robot has no articulation, cannot teleport")
            return
        z_target = z_keep
        if z_target is None:
            z_target = get_articulation_z(articulation)
            if z_target is None:
                z_target = 0.0
                self._log("[Replay] Warning: could not get articulation Z for teleport, using z=0.0")
        quat = _yaw_rad_to_quat_wxyz(float(yaw_rad))
        articulation.set_world_pose((float(x), float(y), float(z_target)), quat)

    @staticmethod
    def _save_png(path: Path, rgb) -> None:
        img = Image.fromarray(rgb)
        img.save(path, quality=95)

    @staticmethod
    def _save_depth(path: Path, depth) -> None:
        """Save as 16-bit PNG (millimetres); 0 means invalid depth."""
        depth_arr = np.asarray(depth, dtype=np.float32)
        invalid = ~np.isfinite(depth_arr)
        depth_mm = np.clip(depth_arr * 1000.0, 0.0, 65535.0)
        depth_u16 = depth_mm.astype(np.uint16)
        if np.any(invalid):
            depth_u16[invalid] = 0
        img = Image.fromarray(depth_u16, mode="I;16")
        img.save(path)

    @staticmethod
    def _get_robot_name(scenario: Dict[str, Any]) -> str:
        entries = scenario.get("robots", {}).get("entries", [])
        if entries:
            return entries[0].get("label", "robot")
        return "robot"

    @staticmethod
    def _create_runner_with_app(config, task_manager, simulation_app) -> "SimulatorRunner":
        from OmniNav.core.runner import SimulatorRunner

        original_setup = SimulatorRunner.setup_isaacsim

        def _reuse_setup(runner_self):
            runner_self._simulation_app = simulation_app
            runner_self._simulation_app._carb_settings.set("/physics/cooking/ujitsoCollisionCooking", False)

        SimulatorRunner.setup_isaacsim = _reuse_setup
        try:
            runner = SimulatorRunner(config=config, task_config_manager=task_manager)
        finally:
            SimulatorRunner.setup_isaacsim = original_setup
        return runner

    @staticmethod
    def _extract_controller_scales(controller) -> Tuple[float, float]:
        fwd = getattr(controller, "forward_speed", 1.0)
        rot = getattr(controller, "rotation_speed", 1.0)
        if fwd == 0 or rot == 0:
            raise ValueError("Controller scale cannot be zero")
        return float(fwd), float(rot)

    @staticmethod
    def _action_builder(controller):
        """Return a callable mapping (v, lat, w) -> controller input tuple."""
        name = controller.__class__.__name__
        if name == "DifferentialDriveController":
            return lambda v, lat, w: (v, w)
        return lambda v, lat, w: (v, lat, w)

    @staticmethod
    def _unwrap_camera_sensor(sensor: Any) -> Any:
        return getattr(sensor, "_camera", None) or sensor

    def _resolve_replay_cameras(self, robot) -> List[ReplayCamera]:
        sensors = getattr(robot, "sensors", None)
        sensor_map: Dict[str, Any] = {}
        if isinstance(sensors, dict):
            for name, sensor in sensors.items():
                sensor_map[str(name)] = self._unwrap_camera_sensor(sensor)

        requested = self._normalize_num_cameras()
        if requested == 3:
            ordered = ["front", "left", "right"]
            missing = [name for name in ordered if name not in sensor_map]
            if missing:
                raise RuntimeError(
                    f"Replay requested 3 cameras, but robot is missing sensors {missing}; "
                    f"available={list(sensor_map.keys())}"
                )
            return [ReplayCamera(name=name, camera=sensor_map[name]) for name in ordered]

        if "front" in sensor_map:
            return [ReplayCamera(name="front", camera=sensor_map["front"])]

        if "camera" in sensor_map:
            return [ReplayCamera(name="camera", camera=sensor_map["camera"])]

        raise RuntimeError(
            f"No replay camera found on robot; expected 'front' or 'camera' sensor, "
            f"available={list(sensor_map.keys())}"
        )

    @staticmethod
    def _set_camera_light() -> None:
        """Switch viewport lighting to camera light for Matterport scenes."""
        try:
            import omni.kit.actions.core as actions  # type: ignore
        except Exception as exc:
            raise RuntimeError("Failed to import omni.kit.actions.core for camera light control") from exc

        registry = actions.get_action_registry()
        action = registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera")
        if action is None:
            raise RuntimeError("Viewport lighting action not found; ensure lighting extension is enabled.")
        action.execute()
