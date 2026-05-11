# pyright: reportMissingImports=false

"""Episode runner for single episode evaluation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple

import numpy as np

from ..policy.base import BasePolicy, Observation, Action
from bench.configs.execution import ExecutionConfig, ExecutionMode, RobotExecutionProfile
from bench.execution.executor import create_executor
from bench.execution.modes.base import BaseActionExecutor
from bench.execution.policy_modes import resolve_policy_mode
from ..metrics.navigation import (
    MeasureSetup,
    add_measurement,
    compute_follow_human_task_success,
    compute_follow_human_success_ratio,
    compute_subtask_progress,
    compute_spl_offline,
    compute_eqa
)
from .termination import TerminationCondition, TerminationResult, StopActionCondition, CompositeCondition
from bench.utils.visualizer import Visualizer

if TYPE_CHECKING:
    from OmniNav.core.runner import SimulatorRunner


@dataclass
class EpisodeConfig:
    """Configuration for a single episode.

    Attributes:
        scenario_id: Unique identifier for the scenario
        instruction: Natural language task instruction
        goal_position: Target position (x, y, z) for navigation
        start_position: Initial robot position (optional, from scenario)
        start_orientation: Initial robot orientation as quaternion (optional)
        max_steps: Maximum steps before timeout
        success_threshold: Distance to goal for success (meters)
        require_leave_goal_first: If True, robot must leave goal area before success
        extra: Additional scenario-specific configuration
    """
    scenario_id: str
    instruction: str
    goal_position: tuple[float, float, float]
    start_position: Optional[tuple[float, float, float]] = None
    start_orientation: Optional[tuple[float, float, float, float]] = None
    max_steps: int = 500
    success_threshold: float = 2.0
    require_leave_goal_first: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryPoint:
    """Single point in trajectory recording."""
    step: int
    time_s: float
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    action: Optional[Action] = None


@dataclass
class EpisodeResult:
    """Result of episode evaluation.

    Attributes:
        scenario_id: Scenario identifier
        success: Whether goal was reached
        termination_reason: Why episode ended
        steps: Total steps taken
        time_s: Total time elapsed
        distance_to_goal: Final distance to goal
        path_length: Total path length traveled
        trajectory: List of trajectory points (if recorded)
        metrics: Additional computed metrics
        extra: Additional scenario-specific data (objects, room_zone, answer, etc.)
    """
    scenario_id: str
    success: bool
    termination_reason: str
    steps: int
    time_s: float
    distance_to_goal: float
    path_length: float
    trajectory: List[TrajectoryPoint] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    stop_step: int = -1




class EpisodeRunner:
    """Runs a single episode and collects results.

    Coordinates policy, environment, and termination conditions.
    """

    def __init__(
        self,
        policy: BasePolicy,
        termination_conditions: List[TerminationCondition],
        record_trajectory: bool = True,
        execution_config: Optional[ExecutionConfig] = None,
        robot_profile: Optional[RobotExecutionProfile] = None,
        visualizer: Optional[Visualizer] = None,
    ):
        """Initialize episode runner.

        Args:
            policy: Navigation policy to evaluate
            termination_conditions: List of conditions that end episode
            record_trajectory: Whether to record full trajectory
            visualizer: Optional Visualizer for recording/visualization
        """
        self.policy = policy
        self.record_trajectory = record_trajectory
        self.execution_config = execution_config or ExecutionConfig()
        self.robot_profile = robot_profile
        self._visualizer = visualizer
        self._mode: Optional[ExecutionMode] = None
        self._executor: Optional[BaseActionExecutor] = None
        self._eqa_result: Optional[str] = None  # Store EQA result
        self._eqa_accuracy: Optional[bool] = None  # Store EQA accuracy
        self._config: Optional[EpisodeConfig] = None  # Store config for EQA

        self._stop_step: int = -1
        # Add stop action condition and wrap in composite
        self._stop_condition = StopActionCondition()
        all_conditions = list(termination_conditions) + [self._stop_condition]
        self._termination = CompositeCondition(all_conditions)
        self._required_camera_names: List[str] = ["camera"]

    def run(
        self,
        runner: "SimulatorRunner",
        config: EpisodeConfig,
        robot_name: str = "robot",
    ) -> EpisodeResult:
        # Store config for EQA
        self._config = config
        """Execute single episode.

        Args:
            runner: OmniNav SimulatorRunner instance
            config: Episode configuration
            robot_name: Name of robot in task.robots dict

        Returns:
            EpisodeResult with evaluation metrics
        """
        print(f"[EpisodeRunner] Starting episode {config.scenario_id}", flush=True)
        # Reset state
        self.policy.reset(instruction=config.instruction)
        self._termination.reset()
        self._required_camera_names = self._parse_required_cameras(config)
        trajectory: List[TrajectoryPoint] = []
        path_length = 0.0
        prev_position: Optional[tuple[float, float, float]] = None
        step = 0
        measure_manager = add_measurement(self._build_measure_setup(config))
        measures_initialized = False

        # Collect human IDs that the agent must follow, used to record their paths.
        follow_humans: List[str] = []
        extra = config.extra or {}
        subtasks = extra.get("subtasks") or []
        if isinstance(subtasks, list):
            for st in subtasks:
                if not isinstance(st, dict):
                    continue
                st_type = str(st.get("type", "")).upper()
                if st_type != "FOLLOW_HUMAN":
                    continue
                for k, v in st.items():
                    if isinstance(k, str) and k.lower().endswith("_id"):
                        follow_humans.append(str(v))
                        break

        human_paths: Dict[str, List[Dict[str, Any]]] = {}
        agent_manager = None
        if follow_humans:
            from OmniNavExt.envset.agent_manager import AgentManager
            agent_manager = AgentManager.get_instance()
            for hid in follow_humans:
                human_paths[hid] = []

        try:
            mode = self._resolve_mode()
        except Exception as e:
            print(f"[EpisodeRunner] Failed to resolve mode: {e}")
            raise
        self._mode = mode
        dt = float(getattr(runner.config.simulator, "physics_dt", 0.00416666667))
        try:
            executor = create_executor(mode=mode, robot_profile=self.robot_profile, exec_config=self.execution_config)
        except Exception as e:
            print(f"[EpisodeRunner] Failed to create executor: {e}")
        self._executor = executor
        # Initial observation
        obs = self._build_observation(
            runner=runner,
            robot_name=robot_name,
            instruction=config.instruction,
            step=step,
            dt=dt,
        )
        # Stream frames to policy continuously (required by MTU3D-style pipelines).
        try:
            self.policy.observe(obs)
        except Exception as e:
            print(f"[EpisodeRunner] Failed to observe: {e}")
        # WAYPOINT execution requires a configured `move_to_point` controller on the robot.
        if mode == ExecutionMode.WAYPOINT:
            # Get the current task (assuming single task scenario like _build_observation does)
            print(runner.current_tasks)
            if not runner.current_tasks:
                raise RuntimeError("No current tasks found; expected at least one active task for WAYPOINT mode")
            task = list(runner.current_tasks.values())[0]
            if robot_name not in task.robots:
                raise RuntimeError(f"Robot '{robot_name}' not found in current task robots: {list(task.robots.keys())}")
            robot = task.robots[robot_name]
            controllers = getattr(robot, "controllers", None)
            if controllers is None or "move_to_point" not in controllers:
                raise RuntimeError(
                    "WAYPOINT mode requires robot.controllers['move_to_point'] to be configured "
                    "(see OmniNavExt/configs/robots/* for controller setup)"
                )

        while True:
            # Initialize / update measures with current pose
            if measure_manager:
                if not measures_initialized:
                    measure_manager.reset_measures(obs.position)
                    measures_initialized = True
                else:
                    measure_manager.update_measures(obs.position)
            # Check termination before acting
            term_result = self._termination.check(obs)
            if term_result.terminated:
                break
            # Start a new action if none active
            if executor.finished or executor.active_action is None:
                try:
                    action = self.policy.act(obs)
                except Exception as e:
                    print(f"[EpisodeRunner] Failed to act: {e}")
                try:
                    executor.start(action, obs, dt)
                except Exception as e:
                    print(f"[EpisodeRunner] Failed to start action: {e}")
                if os.getenv("OMNINAV_LOG_ACTIONS"):
                    print(
                        f"[ActionDebug] step={step} pos=({obs.position[0]:.3f}, {obs.position[1]:.3f}, {obs.position[2]:.3f}) "
                        f"action(lin={action.linear_velocity:.3f}, lat={action.lateral_velocity:.3f}, ang={action.angular_velocity:.3f}, stop={action.stop}, mode={mode.value})"
                    )

            action = executor.active_action
            if action is None:
                raise RuntimeError("Executor has no active action")
            # Execute one physics step
            self._execute_action(runner, robot_name, action, obs)
            step += 1
            # Build post-action observation for this frame
            obs_after = self._build_observation(
                runner=runner,
                robot_name=robot_name,
                instruction=config.instruction,
                step=step,
                dt=dt,
            )
            # Continuous observation during movement.
            self.policy.observe(obs_after)

            # Update measures/path with latest pose
            if measure_manager and measures_initialized:
                measure_manager.update_measures(obs_after.position)

            if prev_position is not None:
                dx = obs_after.position[0] - prev_position[0]
                dy = obs_after.position[1] - prev_position[1]
                path_length += np.sqrt(dx * dx + dy * dy)
            prev_position = obs_after.position

            # Record trajectory per frame if enabled
            if self.record_trajectory:
                trajectory.append(
                    TrajectoryPoint(
                        step=step,
                        time_s=obs_after.time_s,
                        position=obs_after.position,
                        orientation=obs_after.orientation,
                        action=action,
                    )
                )
            # Record video frame if visualizer is provided
            if self._visualizer is not None:
                self._visualizer.record_step(step, obs_after)
            if human_paths and agent_manager is not None:
                for hid in follow_humans:
                    human_pos = agent_manager.get_agent_pos_by_name(hid)
                    if human_pos is not None:
                        human_paths[hid].append({
                            "step": step,
                            "position": (
                                float(human_pos[0]),
                                float(human_pos[1]),
                                float(human_pos[2]),
                            ),
                        })

            # Stop action requested: break after first frame
            if action.stop:
                if self._stop_step < 0:
                    self._stop_step = step
                self._stop_condition.set_stop_requested(True)
            # Termination check on every frame (includes stop condition)
            term_result = self._termination.check(obs_after)
            if term_result.terminated:
                obs = obs_after
                break

            # Progress log
            if step % 50 == 0:
                dist = self._compute_distance(obs_after.position, config.goal_position)
                print(
                    f"  [Step {step}] pos=({obs_after.position[0]:.2f}, {obs_after.position[1]:.2f}, {obs_after.position[2]:.2f}), "
                    f"dist_to_goal={dist:.2f}m"
                )

            # Update executor progress
            finished = executor.step_finished(obs_after)
            obs = obs_after
            if finished:
                continue

        # Compute final metrics
        final_obs = self._build_observation(
            runner=runner,
            robot_name=robot_name,
            instruction=config.instruction,
            step=step,
            dt=dt,
        )

        distance_to_goal = self._compute_distance(
            final_obs.position, config.goal_position
        )

        if measure_manager and measures_initialized:
            measure_manager.update_measures(final_obs.position)

        # Success requires both a policy stop and reaching the goal.
        stopped = term_result.reason == "Policy requested stop"
        goal_radius = float((config.extra or {}).get("goal_radius", config.success_threshold))
        success = bool(stopped and distance_to_goal < goal_radius)

        print(
            "[EpisodeRunner] Finished episode "
            f"{config.scenario_id} with steps={step}, time_s={final_obs.time_s:.2f}"
        )
        metrics: Dict[str, Any] = {
            "path_length": path_length,
            "distance_to_goal": distance_to_goal,
        }
        metrics.update(measure_manager.get_measurements())

        shortest_path = metrics.get("shortest_path")
        print(shortest_path)
        if shortest_path is None:
            shortest_path = (config.extra or {}).get("shortest_path", path_length)

        spl_value = compute_spl_offline(
            success=success,
            path_length=path_length,
            shortest_path=float(shortest_path)
        )
        metrics["spl"] = spl_value

        # FOLLOW_HUMAN: offline overall success based on final stop position relative to human.
        if human_paths:
            try:
                follow_dist_cfg = (config.extra or {}).get("follow_distance", 3.0)
                try:
                    follow_dist = float(follow_dist_cfg)
                except (TypeError, ValueError):
                    follow_dist = 3.0

                fh_success = compute_follow_human_task_success(
                    human_paths=human_paths,
                    trajectory=trajectory,
                    distance_threshold=follow_dist,
                    inner_threshold=0.1,
                )
                metrics["follow_human_task_success"] = float(fh_success)

                # Per-step success ratio inside the follow segment (0..1).
                fh_ratio = compute_follow_human_success_ratio(
                    human_paths=human_paths,
                    trajectory=trajectory,
                    distance_threshold=follow_dist,
                    inner_threshold=0.1,
                )
                metrics["follow_human_success_ratio"] = float(fh_ratio)
            except Exception as e:
                # Don't let follow-human metric failures block episode result write-out.
                print(f"[EpisodeRunner][WARN] Failed to compute follow-human offline metrics: {e}")
        # ------------------------------------------------------------------
        # Offline subtask metrics: CSR / SoftSR
        # ------------------------------------------------------------------
        config_extra = config.extra or {}
        subtasks = config_extra.get("subtasks") or []

        def compute_goto_point_distance(
            trajectory: List[TrajectoryPoint],
            target_position: Tuple[float, float, float]
        ) -> float:
            """Minimum distance from any trajectory point to the target position."""
            if not trajectory:
                return float("inf")

            min_dist = float("inf")
            for point in trajectory:
                dist = self._compute_distance(point.position, target_position)
                min_dist = min(min_dist, dist)
            return min_dist

        def first_goto_point_success_timestamp(
            trajectory: List[TrajectoryPoint],
            target_position: Tuple[float, float, float],
            threshold_m: float,
        ) -> int:
            """Return the first trajectory index that enters the target radius."""
            for index, point in enumerate(trajectory):
                if self._compute_distance(point.position, target_position) <= threshold_m:
                    return index
            return -1

        # Pull object / room / follow_human online/offline status up front.
        object_status_raw = metrics.get("object_reach_status") or {}
        room_status_raw = metrics.get("room_zone_status") or {}
        follow_human_task_success = bool(metrics.get("follow_human_task_success", 0.0))
        follow_human_success_ratio = float(metrics.get("follow_human_success_ratio", 0.0))

        # Build case-insensitive lookup tables.
        object_status = {
            str(name).lower(): value for name, value in object_status_raw.items()
        }
        room_status = {
            str(name).lower(): value for name, value in room_status_raw.items()
        }

        goal_radius = float(config_extra.get("goal_radius", config.success_threshold))
        object_threshold = float(config_extra.get("object_success_threshold", goal_radius))

        subtask_states: List[Dict[str, Any]] = []

        if isinstance(subtasks, list):
            for st in subtasks:
                if not isinstance(st, dict):
                    continue
                st_type = str(st.get("type", "")).upper()

                # Use the first *_id field as the target name.
                target_id = None
                for k, v in st.items():
                    if isinstance(k, str) and k.lower().endswith("_id"):
                        target_id = str(v)
                        break

                target_name_lc = target_id.lower() if target_id is not None else None

                success_flag = False
                timestamp = -1
                progress = 0.0

                if st_type == "FOLLOW_HUMAN":
                    # FOLLOW_HUMAN uses overall offline success; per-step timestamp is not meaningful.
                    success_flag = follow_human_task_success
                    timestamp = -1
                    progress = compute_subtask_progress(st_type, {"success_ratio": follow_human_success_ratio}, 0.0)
                elif st_type == "GOTO_ROOM":
                    if target_name_lc is not None and target_name_lc in room_status:
                        info = room_status[target_name_lc]
                        success_flag = bool(info.get("entered", 0))
                        timestamps = info.get("timestamp", [])
                        if isinstance(timestamps, list):
                            # Multiple entries: compute progress per timestamp, keep the max.
                            progress_values = []
                            for ts in timestamps:
                                temp_info = info.copy()
                                temp_info["timestamp"] = ts
                                progress_values.append(compute_subtask_progress(st_type, temp_info, goal_radius))
                            progress = max(progress_values) if progress_values else 0.0
                            timestamp = max(timestamps) if timestamps else -1
                        else:
                            timestamp = int(timestamps)
                            progress = compute_subtask_progress(st_type, info, goal_radius)
                    else:
                        progress = 0.0
                elif st_type == "GOTO_POINT":
                    # GOTO_POINT: compute min trajectory-to-target distance directly.
                    point_position = st.get("position")
                    if isinstance(point_position, (list, tuple)) and len(point_position) >= 3:
                        target_pos = (float(point_position[0]), float(point_position[1]), float(point_position[2]))
                        min_dist = compute_goto_point_distance(trajectory, target_pos)
                        threshold_m = float(st.get("radius", 0.36))  # default 0.36 m

                        success_flag = bool(min_dist <= threshold_m)
                        progress = compute_subtask_progress(st_type, {"min_distance": min_dist}, threshold_m)
                        timestamp = (
                            first_goto_point_success_timestamp(trajectory, target_pos, threshold_m)
                            if success_flag
                            else -1
                        )
                    else:
                        progress = 0.0
                        success_flag = False
                        timestamp = -1
                elif st_type == "RETURN_TO":
                    # RETURN_TO inherits the episode-level success flag; no per-step timestamp.
                    success_flag = bool(success)
                    progress = 1.0 if success_flag else 0.0
                    timestamp = -1
                else:
                    # All other subtask types (GOTO_OBJECT, GOTO_LANDMARK) are "approach target" style.
                    if target_name_lc is not None and target_name_lc in object_status:
                        info = object_status[target_name_lc]
                        thresholds_by_type = {
                            "GOTO_OBJECT": 1.0,
                            "GOTO_LANDMARK": 3.0,
                        }
                        if st_type not in thresholds_by_type:
                            raise ValueError(f"Unknown subtask type for distance-threshold evaluation: {st_type}")
                        threshold_m = float(thresholds_by_type[st_type])

                        min_dist = info.get("min_distance", float("inf"))
                        success_flag = bool(float(min_dist) <= threshold_m)
                        progress = compute_subtask_progress(st_type, info, threshold_m)

                        ts_by_thr = info.get("timestamp_by_threshold") if isinstance(info, dict) else None
                        if isinstance(ts_by_thr, dict):
                            key = str(threshold_m)
                            if key in ts_by_thr:
                                timestamp = int(ts_by_thr.get(key, -1))
                    else:
                        progress = 0.0

                subtask_states.append(
                    {
                        "type": st_type,
                        "id": target_id,
                        "success": bool(success_flag),
                        "timestamp": int(timestamp),
                        "progress": 1.0 if success_flag else float(progress),
                    }
                )
        # SoftSR: order-agnostic mean of per-subtask progress (continuous 0..1).
        if subtask_states:
            softsr = sum(float(s["progress"]) for s in subtask_states) / len(subtask_states)
        else:
            softsr = 0.0
        metrics["softsr"] = float(softsr)

        # CSR: all subtasks + access-to-goal succeed AND (excluding FOLLOW_HUMAN and access-to-goal)
        # subtasks succeed in the declared `subtasks` order.
        all_subtasks_success = all(s["success"] for s in subtask_states)
        order_ok = True
        prev_ts = -1
        for s in subtask_states:
            if s["type"] == "FOLLOW_HUMAN":
                continue
            ts = s["timestamp"]
            if not s["success"] or ts < 0:
                order_ok = False
                break
            if prev_ts >= 0 and not (ts > prev_ts):
                order_ok = False
                break
            prev_ts = ts

        csr = 1.0 if (all_subtasks_success and order_ok) else 0.0
        metrics["csr"] = csr
    
        metrics["subtask_details"] = [
            {
                "type": s["type"],
                "id": s["id"],
                "success": s["success"],
                "timestamp": s["timestamp"],
                "progress": s["progress"]
            }
            for s in subtask_states
        ]
        # Personal-space violation ratio = violation_steps / total_steps.
        human_personal_space = metrics.get("human_personal_space", {})
        violation_steps = int(human_personal_space.get("violation_steps", 0)) if isinstance(human_personal_space, dict) else 0
        total_steps = max(step, 1)  # avoid divide-by-zero
        social_violation_ratio = float(violation_steps / total_steps) if total_steps > 0 else 0.0
        metrics["social_violation_ratio"] = social_violation_ratio

        # Only whitelisted fields are passed through to result.extra
        # to prevent private data (objects, room_zone, subtasks, expert_path, etc.) leaking.
        extra: Dict[str, Any] = {}
        if human_paths:
            extra["human_paths"] = human_paths

        # Call EQA for UniNaVid models after episode completion
        self._call_eqa_if_available(final_obs, config.instruction)

        # Add EQA result and accuracy to metrics/extra if available
        if self._eqa_result is not None:
            extra["eqa_answer"] = self._eqa_result
        # Always include eqa_accuracy in metrics, even if None (for no EQA task)
        metrics["eqa_accuracy"] = self._eqa_accuracy

        return EpisodeResult(
            scenario_id=config.scenario_id,
            success=success,
            termination_reason=term_result.reason,
            steps=step,
            time_s=final_obs.time_s,
            distance_to_goal=distance_to_goal,
            path_length=path_length,
            trajectory=trajectory if self.record_trajectory else [],
            metrics=metrics,
            extra=extra,
            stop_step=self._stop_step,
        )

    def _build_measure_setup(self, config: EpisodeConfig) -> MeasureSetup:
        extra = config.extra or {}
        goal_radius = float(extra.get("goal_radius", config.success_threshold))

        raw_waypoints = extra.get("gt_waypoints") or []
        waypoints: List[tuple[float, float, float]] = []
        for wp in raw_waypoints:
            if isinstance(wp, (list, tuple)) and len(wp) >= 3:
                waypoints.append((float(wp[0]), float(wp[1]), float(wp[2])))
        if not waypoints:
            if config.start_position is not None:
                waypoints = [config.start_position, config.goal_position]
            else:
                waypoints = [config.goal_position]

        shortest_path = extra.get("shortest_path")
        if not isinstance(shortest_path, (float, int)):
            shortest_path = self._estimate_shortest_path(waypoints)

        # Object subtasks (entries whose name starts with "Human" go to humans[]).
        objects: List[tuple[str, tuple[float, float, float]]] = []
        humans: List[tuple[str, tuple[float, float, float]]] = []
        raw_objects = extra.get("objects")
        if isinstance(raw_objects, dict):
            for name, pos in raw_objects.items():
                if (
                    isinstance(pos, (list, tuple))
                    and len(pos) >= 3
                ):
                    entry = (str(name), (float(pos[0]), float(pos[1]), float(pos[2])))
                    if str(name).startswith("Human"):
                        humans.append(entry)
                    else:
                        objects.append(entry)

        object_threshold = extra.get("object_success_threshold", goal_radius)
        if not isinstance(object_threshold, (float, int)):
            object_threshold = goal_radius

        social_distance = extra.get("social_distance_threshold", goal_radius)
        if not isinstance(social_distance, (float, int)):
            social_distance = goal_radius

        # NavMesh enables geodesic object-proximity checks; otherwise fall back to plain 3D Euclidean.
        navmesh = None
        try:
            import omni.anim.navigation.core as nav
            inav = nav.acquire_interface()
            if inav is not None:
                navmesh = inav.get_navmesh()
                if navmesh is None:
                    print(f"[EpisodeRunner][WARN] NavMesh unavailable; object proximity falls back to 3D Euclidean distance.")
            else:
                print(f"[EpisodeRunner][WARN] NavMesh interface unavailable; object proximity falls back to 3D Euclidean distance.")
        except Exception as e:
            print(f"[EpisodeRunner][WARN] NavMesh acquisition failed ({type(e).__name__}: {e}); falling back to 3D Euclidean distance.")
            navmesh = None

        return MeasureSetup(
            goal_position=config.goal_position,
            goal_radius=goal_radius,
            waypoints=waypoints,
            shortest_path=float(shortest_path),
            objects=objects,
            object_threshold=float(object_threshold),
            navmesh=navmesh,            
            humans=humans,
            social_distance=float(social_distance),     
            room_zones=extra.get("room_zone"),       
        )

    @staticmethod
    def _estimate_shortest_path(
        waypoints: List[tuple[float, float, float]]
    ) -> float:
        if len(waypoints) < 2:
            return 0.0
        total = 0.0
        for i in range(len(waypoints) - 1):
            dx = waypoints[i + 1][0] - waypoints[i][0]
            dy = waypoints[i + 1][1] - waypoints[i][1]
            total += np.sqrt(dx * dx + dy * dy)
        return total

    def _parse_required_cameras(self, config: EpisodeConfig) -> List[str]:
        """Return required camera sensor names for this episode (fail-fast)."""
        extra = config.extra or {}
        required = extra.get("required_cameras")
        if required is None:
            return ["camera"]
        if not isinstance(required, list) or not required:
            raise ValueError("EpisodeConfig.extra['required_cameras'] must be a non-empty list of camera names")

        unique: List[str] = []
        for item in required:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("EpisodeConfig.extra['required_cameras'] must contain non-empty strings")
            name = item.strip()
            if name not in unique:
                unique.append(name)
        return unique

    @staticmethod
    def _resolve_camera_for_name(robot: Any, camera_name: str):
        """Resolve a camera-like object for the given name from robot.{camera,sensors}."""
        if camera_name == "camera":
            direct = getattr(robot, "camera", None)
            if direct is not None:
                return direct

        sensors = getattr(robot, "sensors", None)
        if isinstance(sensors, dict):
            sensor = sensors.get(camera_name)
            if sensor is not None:
                return getattr(sensor, "_camera", None) or sensor
        available = list(sensors.keys()) if isinstance(sensors, dict) else []
        raise RuntimeError(f"No camera '{camera_name}' found on robot; available sensors={available}")

    def _build_observation(
        self,
        runner: "SimulatorRunner",
        robot_name: str,
        instruction: str,
        step: int,
        dt: float,
    ) -> Observation:
        """Build observation from current environment state."""
        # Get robot from runner
        task = list(runner.current_tasks.values())[0]  # Single task assumption
        robot = task.robots.get(robot_name)

        if robot is None:
            raise RuntimeError(f"Robot '{robot_name}' not found in task")

        required = list(self._required_camera_names or ["camera"])
        primary_name = "camera" if "camera" in required else required[0]

        cameras: List[Dict[str, Any]] = []
        primary_rgb = None
        primary_depth = None
        primary_pose = None
        primary_params = None

        for name in required:
            camera = self._resolve_camera_for_name(robot, name)

            rgba = camera.get_rgba()
            if rgba is None:
                raise RuntimeError(f"Camera '{name}'.get_rgba() returned None")
            rgb = rgba[:, :, :3]

            depth = None
            if hasattr(camera, "get_depth"):
                depth = camera.get_depth()
            elif hasattr(camera, "get_distance_to_image_plane"):
                depth = camera.get_distance_to_image_plane()

            cam_pose = None
            try:
                if hasattr(camera, "get_pose"):
                    cpos, cquat = camera.get_pose()
                elif hasattr(camera, "get_world_pose"):
                    cpos, cquat = camera.get_world_pose()
                else:
                    cpos, cquat = None, None
                if cpos is not None and cquat is not None:
                    cam_pos = tuple(cpos.tolist() if isinstance(cpos, np.ndarray) else cpos)
                    cam_quat = tuple(cquat.tolist() if isinstance(cquat, np.ndarray) else cquat)
                    if len(cam_pos) == 3 and len(cam_quat) == 4:
                        cam_pose = {
                            "position": [float(v) for v in cam_pos],
                            "orientation_wxyz": [float(v) for v in cam_quat],
                        }
            except Exception:
                cam_pose = None

            cam_params = None
            try:
                if hasattr(camera, "get_camera_params"):
                    cam_params = camera.get_camera_params()
            except Exception:
                cam_params = None

            payload: Dict[str, Any] = {
                "name": name,
                "rgb": rgb,
                "depth": depth,
            }
            if cam_pose is not None:
                payload["camera_pose"] = cam_pose
            if cam_params is not None:
                payload["camera_params"] = cam_params
            cameras.append(payload)

            if name == primary_name:
                primary_rgb = rgb
                primary_depth = depth
                primary_pose = cam_pose
                primary_params = cam_params

        # Get robot pose
        position = (0.0, 0.0, 0.0)
        orientation = (1.0, 0.0, 0.0, 0.0)
        try:
            pos, quat = robot.get_pose()
            position = tuple(pos.tolist() if isinstance(pos, np.ndarray) else pos)
            orientation = tuple(quat.tolist() if isinstance(quat, np.ndarray) else quat)
        except Exception:
            if hasattr(robot, 'articulation') and robot.articulation is not None:
                try:
                    pos, quat = robot.articulation.get_world_poses()
                    position = tuple(pos[0].tolist())
                    orientation = tuple(quat[0].tolist())
                except Exception:
                    pass

        # GroundProbe replaces the robot's z with the projected ground height under (x, y).
        ground_probe = getattr(task, '_ground_probe', None)
        if ground_probe is not None:
            x, y, z = position
            ground_z, success = ground_probe.project(x, y, z)
            if success:
                position = (x, y, ground_z)

        extra: Dict[str, Any] = {}
        extra["cameras"] = cameras
        # Backward-compat: keep primary camera pose under camera_pose if available.
        if primary_pose is not None:
            extra["camera_pose"] = primary_pose
        if primary_params is not None:
            extra["camera_params"] = primary_params

        return Observation(
            rgb=primary_rgb,
            depth=primary_depth,
            position=position,
            orientation=orientation,
            instruction=instruction,
            step=step,
            time_s=step * dt,
            extra=extra,
        )

    def _execute_action(
        self,
        runner: "SimulatorRunner",
        robot_name: str,
        action: Action,
        obs: Observation,
    ):
        """Execute action in environment."""
        if self._executor is None:
            raise RuntimeError("Executor is not initialized")
        self._executor.execute(runner, robot_name, action)

    @staticmethod
    def _compute_distance(
        pos1: tuple[float, float, float],
        pos2: tuple[float, float, float],
    ) -> float:
        """Compute 2D Euclidean distance."""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return np.sqrt(dx * dx + dy * dy)

    def _call_eqa_if_available(self, final_obs: Observation, instruction: str):
        """Call EQA for UniNaVid models if available using final frame."""
        # Check if this is a UniNaVid policy
        policy_name = self.policy.__class__.__name__
        if not policy_name.lower().startswith('uninavid'):
            return  # Only UniNaVid has EQA capability

        # Check if EQA task is available (has qa info)
        # If qa field is null, None, or empty, skip EQA entirely
        has_eqa_task = False
        if hasattr(self, '_config') and self._config.extra:
            qa_info = self._config.extra.get('qa')
            if qa_info is not None and isinstance(qa_info, dict) and qa_info:
                # Check if qa has at least a question
                if 'question' in qa_info:
                    has_eqa_task = True

        if not has_eqa_task:
            # No EQA task, set result to None
            self._eqa_result = None
            print("[EpisodeRunner] Skipping EQA - no valid qa information found")
            return

        try:
            # Check if the policy has predict_text method
            if not hasattr(self.policy, 'predict_text'):
                print("[EpisodeRunner] UniNaVid policy doesn't have predict_text method, skipping EQA")
                self._eqa_result = None
                return

            # Get the EQA question from config
            qa_info = self._config.extra.get('qa')
            eqa_question = qa_info.get('question', instruction)

            # Use the final observation image
            if final_obs.rgb is None:
                print("[EpisodeRunner] No RGB image in final observation, skipping EQA")
                self._eqa_result = None
                return

            # Call EQA using policy's predict_text method
            print(f"[EpisodeRunner] Calling EQA with question: {eqa_question}")
            eqa_answer = self.policy.predict_text(eqa_question, final_obs.rgb)

            print(f"[EpisodeRunner] EQA Answer: {eqa_answer}")
            self._eqa_result = eqa_answer

            # Get ground truth answer and evaluate correctness
            ground_truth = qa_info.get('answer')
            if ground_truth:
                eqa_correct = compute_eqa(ground_truth, eqa_answer)
                print(f"[EpisodeRunner] EQA Correct: {eqa_correct} (GT: '{ground_truth}')")
                self._eqa_accuracy = eqa_correct
            else:
                print("[EpisodeRunner] No ground truth answer available for EQA evaluation")
                self._eqa_accuracy = None

        except Exception as e:
            print(f"[EpisodeRunner] EQA call failed: {e}")
            self._eqa_result = None
            self._eqa_accuracy = None

    def _resolve_mode(self) -> ExecutionMode:
        """Resolve execution mode for current policy."""
        return resolve_policy_mode(self.policy, self.execution_config.policy_mode_map)

    # Early-stop logic has been removed; termination is driven solely by
    # configured TerminationCondition instances and explicit stop actions.
