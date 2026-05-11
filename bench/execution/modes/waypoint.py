from __future__ import annotations

from typing import List, Tuple

import numpy as np

from bench.configs.execution import ExecutionMode
from bench.policy.base import Action, Observation

from .base import BaseActionExecutor, angle_diff, quat_to_yaw


class WaypointExecutor(BaseActionExecutor):
    """Execute controller-goal actions for waypoint-style policies."""

    def start(self, action: Action, obs: Observation, dt: float) -> None:
        """Start tracking a new waypoint action."""
        self.active_action = action
        self.finished = False

        if self.mode != ExecutionMode.WAYPOINT:
            raise ValueError(f"WaypointExecutor requires WAYPOINT mode, got {self.mode}")
        if action.stop:
            self.finished = True
            self.controller_name = None
            return

        self.start_pos = obs.position
        self.start_yaw = quat_to_yaw(obs.orientation)
        self.stuck_check_pos = None
        self.stuck_step_count = 0

        controller = action.extra.get("controller")
        if controller not in ("move_to_point", "rotate", "move_along_path", "go_toward_point"):
            raise ValueError(
                "WAYPOINT requires action.extra['controller'] in "
                "{'move_to_point','rotate','move_along_path','go_toward_point'}"
            )

        if controller == "move_to_point":
            self._start_move_to_point(action)
        elif controller == "rotate":
            self._start_rotate(action)
        elif controller == "move_along_path":
            self._start_move_along_path(action)
        elif controller == "go_toward_point":
            self._start_go_toward_point(action)

        self.target_lin = 0.0
        self.target_ang = 0.0

    def _start_move_to_point(self, action: Action) -> None:
        goal = action.extra.get("goal_position")
        if goal is None:
            raise ValueError("WAYPOINT/move_to_point requires action.extra['goal_position'] = (x, y, z) in meters")
        if not (isinstance(goal, (list, tuple)) and len(goal) == 3):
            raise ValueError("WAYPOINT/move_to_point requires action.extra['goal_position'] as a 3-tuple/list (x, y, z)")

        threshold_m = self._positive_float(
            action.extra.get("threshold_m"),
            "WAYPOINT/move_to_point requires action.extra['threshold_m'] (meters)",
            "WAYPOINT/move_to_point requires action.extra['threshold_m'] as float (meters)",
        )
        self.goal_position = (float(goal[0]), float(goal[1]), float(goal[2]))
        self.goal_threshold_m = threshold_m
        self.goal_threshold_rad = None
        self.goal_orientation = None
        self.controller_name = "move_to_point"

    def _start_rotate(self, action: Action) -> None:
        goal_q = action.extra.get("goal_orientation_wxyz")
        if goal_q is None:
            raise ValueError("WAYPOINT/rotate requires action.extra['goal_orientation_wxyz'] = (w, x, y, z)")
        if not (isinstance(goal_q, (list, tuple)) and len(goal_q) == 4):
            raise ValueError("WAYPOINT/rotate requires action.extra['goal_orientation_wxyz'] as a 4-tuple/list (w, x, y, z)")

        threshold_rad = self._positive_float(
            action.extra.get("threshold_rad"),
            "WAYPOINT/rotate requires action.extra['threshold_rad'] (radians)",
            "WAYPOINT/rotate requires action.extra['threshold_rad'] as float (radians)",
        )
        self.goal_orientation = tuple(float(v) for v in goal_q)
        self.goal_threshold_rad = threshold_rad
        self.goal_threshold_m = None
        self.goal_position = None
        self.controller_name = "rotate"

    def _start_move_along_path(self, action: Action) -> None:
        path_points_raw = action.extra.get("path_points")
        if path_points_raw is None:
            raise ValueError("WAYPOINT/move_along_path requires action.extra['path_points'] = list of (x, y, z) tuples")
        if not isinstance(path_points_raw, (list, tuple)) or len(path_points_raw) == 0:
            raise ValueError("WAYPOINT/move_along_path requires action.extra['path_points'] as a non-empty list")

        path_points: List[Tuple[float, float, float]] = []
        for index, point in enumerate(path_points_raw):
            if not (isinstance(point, (list, tuple)) and len(point) >= 2):
                raise ValueError(f"WAYPOINT/move_along_path: path_points[{index}] must be a tuple/list with at least 2 elements")
            x, y = float(point[0]), float(point[1])
            z = float(point[2]) if len(point) >= 3 else 0.0
            path_points.append((x, y, z))

        threshold = action.extra.get("threshold_m")
        threshold_m = 0.1 if threshold is None else self._positive_float(
            threshold,
            "WAYPOINT/move_along_path requires action.extra['threshold_m'] (meters)",
            "WAYPOINT/move_along_path requires action.extra['threshold_m'] as float (meters)",
            fallback=0.1,
        )
        self.path_points = path_points
        self.goal_position = path_points[-1]
        self.goal_threshold_m = threshold_m
        self.goal_threshold_rad = None
        self.goal_orientation = None
        self.controller_name = "move_along_path"

    def _start_go_toward_point(self, action: Action) -> None:
        theta = action.extra.get("theta_rad")
        r_m = action.extra.get("r_m")
        if theta is None or r_m is None:
            raise ValueError("WAYPOINT/go_toward_point requires action.extra['theta_rad'] and ['r_m']")
        try:
            theta_rad = float(theta)
            dist_m = float(r_m)
        except (TypeError, ValueError) as exc:
            raise ValueError("WAYPOINT/go_toward_point requires theta_rad/r_m as floats") from exc
        if not np.isfinite(theta_rad) or not np.isfinite(dist_m):
            raise ValueError("WAYPOINT/go_toward_point requires finite theta_rad/r_m")
        if dist_m < 0:
            raise ValueError("WAYPOINT/go_toward_point requires r_m >= 0")

        threshold = action.extra.get("threshold_m")
        if threshold is None:
            threshold_m = float(self.robot_profile.finish_pos_eps) if self.robot_profile is not None else 0.1
        else:
            threshold_m = self._positive_float(
                threshold,
                "WAYPOINT/go_toward_point requires threshold_m (meters)",
                "WAYPOINT/go_toward_point requires threshold_m as float (meters)",
            )

        if self.start_pos is None or self.start_yaw is None:
            raise RuntimeError("WAYPOINT/go_toward_point missing start pose")

        goal_yaw = self.start_yaw + theta_rad
        self.theta_rad = theta_rad
        self.r_m = dist_m
        self.command_id += 1
        self.goal_position = (
            self.start_pos[0] + float(np.cos(goal_yaw) * dist_m),
            self.start_pos[1] + float(np.sin(goal_yaw) * dist_m),
            self.start_pos[2],
        )
        self.goal_threshold_m = threshold_m
        self.goal_threshold_rad = None
        self.goal_orientation = None
        self.path_points = None
        self.controller_name = "go_toward_point"

    @staticmethod
    def _positive_float(value, missing_message: str, type_message: str, fallback: float | None = None) -> float:
        if value is None:
            raise ValueError(missing_message)
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(type_message) from exc
        if result <= 0:
            if fallback is not None:
                return fallback
            raise ValueError(missing_message.replace("requires", "requires positive"))
        return result

    def execute(self, runner, robot_name: str, action: Action) -> None:
        """Send the active waypoint action to the simulator runner."""
        if getattr(action, "action_type", None) == "stop" or action.stop:
            return

        if self.controller_name == "move_to_point":
            if self.goal_position is None:
                raise RuntimeError("WAYPOINT/move_to_point missing goal_position")
            controller_action = [np.array(self.goal_position, dtype=np.float32)]
            runner.step(actions=[{robot_name: {"move_to_point": controller_action}}], render=True)
            return

        if self.controller_name == "rotate":
            if self.goal_orientation is None:
                raise RuntimeError("WAYPOINT/rotate missing goal_orientation")
            controller_action = [np.array(self.goal_orientation, dtype=np.float32)]
            runner.step(actions=[{robot_name: {"rotate": controller_action}}], render=True)
            return

        if self.controller_name == "move_along_path":
            if self.path_points is None or len(self.path_points) == 0:
                raise RuntimeError("WAYPOINT/move_along_path missing path_points")
            path_points_np = [np.array(point, dtype=np.float32) for point in self.path_points]
            runner.step(actions=[{robot_name: {"move_along_path": [path_points_np]}}], render=True)
            return

        if self.controller_name == "go_toward_point":
            if self.theta_rad is None or self.r_m is None:
                raise RuntimeError("WAYPOINT/go_toward_point missing theta_rad/r_m")
            controller_action = [float(self.theta_rad), float(self.r_m), float(self.command_id)]
            runner.step(actions=[{robot_name: {"go_toward_point": controller_action}}], render=True)
            return

        raise RuntimeError(f"WAYPOINT has unsupported controller: {self.controller_name}")

    def _progress(self, obs_after: Observation) -> bool:
        """Check whether the active waypoint action has completed."""
        self.stuck_step_count += 1
        if self.stuck_check_pos is None:
            self.stuck_check_pos = obs_after.position

        if self.stuck_step_count >= self.stuck_check_interval:
            dx = obs_after.position[0] - self.stuck_check_pos[0]
            dy = obs_after.position[1] - self.stuck_check_pos[1]
            dist_moved = float(np.sqrt(dx * dx + dy * dy))
            if dist_moved < self.stuck_threshold_m:
                print(f"[Executor] WAYPOINT stuck: moved only {dist_moved:.3f}m in {self.stuck_step_count} steps, requesting new waypoints")
                return True
            self.stuck_check_pos = obs_after.position
            self.stuck_step_count = 0

        if self.controller_name in ("move_to_point", "move_along_path", "go_toward_point"):
            if self.goal_position is None or self.goal_threshold_m is None:
                raise RuntimeError(f"WAYPOINT/{self.controller_name} missing goal_position/threshold_m")
            dx = obs_after.position[0] - self.goal_position[0]
            dy = obs_after.position[1] - self.goal_position[1]
            dist = float(np.sqrt(dx * dx + dy * dy))
            return dist <= self.goal_threshold_m

        if self.controller_name == "rotate":
            if self.goal_orientation is None or self.goal_threshold_rad is None:
                raise RuntimeError("WAYPOINT/rotate missing goal_orientation/threshold_rad")
            yaw_now = quat_to_yaw(obs_after.orientation)
            yaw_goal = quat_to_yaw(self.goal_orientation)
            yaw_err = abs(angle_diff(yaw_now, yaw_goal))
            return yaw_err <= self.goal_threshold_rad

        raise RuntimeError(f"WAYPOINT has unsupported controller_name={self.controller_name}")
