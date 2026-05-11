from __future__ import annotations

import numpy as np

from bench.configs.execution import ExecutionMode
from bench.policy.base import Action, Observation

from .base import BaseActionExecutor, angle_diff, quat_to_yaw


class StepActionExecutor(BaseActionExecutor):
    """Execute discrete forward/backward/turn/wait/stop actions."""

    def start(self, action: Action, obs: Observation, dt: float) -> None:
        """Start tracking a new step action."""
        self.active_action = action
        self.finished = False

        if self.mode != ExecutionMode.STEP_ACTION:
            raise ValueError(f"StepActionExecutor requires STEP_ACTION mode, got {self.mode}")
        if self.robot_profile is None:
            raise ValueError("STEP_ACTION requires robot execution profile")

        self.start_pos = obs.position
        self.start_yaw = quat_to_yaw(obs.orientation)

        if getattr(action, "action_type", None):
            action_type = action.action_type
        else:
            if action.stop:
                action_type = "stop"
            elif abs(action.angular_velocity) > abs(action.linear_velocity):
                action_type = "left" if action.angular_velocity > 0 else "right"
            else:
                action_type = "forward" if action.linear_velocity > 0 else "stop"

        if action_type == "stop":
            self.finished = True
            self.controller_name = None
            self.target_lin = 0.0
            self.target_ang = 0.0
            return

        if action_type == "wait":
            wait_duration_s = 1.0
            self.remaining_steps = max(1, int(np.ceil(wait_duration_s / max(dt, 1e-6))))
            self.controller_name = None
            self.target_lin = 0.0
            self.target_ang = 0.0
            return

        if action_type in ("left", "right"):
            step_rad = np.deg2rad(self.robot_profile.step_ang_deg)
            delta = step_rad if action_type == "left" else -step_rad
            yaw_target = self.start_yaw + delta
            half = yaw_target * 0.5
            self.goal_orientation = (float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))
            self.goal_position = None
            self.controller_name = "rotate"
            self.target_ang = abs(step_rad)
            self.target_lin = 0.0
            return

        if action_type in ("forward", "backward"):
            sign = -1.0 if action_type == "backward" else 1.0
            dx = float(np.cos(self.start_yaw)) * self.robot_profile.step_lin_dist * sign
            dy = float(np.sin(self.start_yaw)) * self.robot_profile.step_lin_dist * sign
            x, y, z = self.start_pos
            self.goal_position = (x + dx, y + dy, z)
            self.goal_orientation = None
            self.controller_name = "move_by_speed"
            self.target_lin = self.robot_profile.step_lin_dist
            self.target_ang = 0.0
            return

        raise ValueError(f"Unknown action_type for STEP_ACTION: {action_type}")

    def execute(self, runner, robot_name: str, action: Action) -> None:
        """Send the active step action to the simulator runner."""
        if self.controller_name == "move_by_speed":
            if self.goal_position is None:
                raise RuntimeError("STEP_ACTION missing goal_position for move_by_speed")
            base_speed = self.robot_profile.max_lin_vel if self.robot_profile else 1.0
            forward_speed = -base_speed if action.action_type == "backward" else base_speed
            controller_action = (forward_speed, 0.0, 0.0)
            runner.step(actions=[{robot_name: {"move_by_speed": controller_action}}], render=True)
            return

        if self.controller_name == "rotate":
            if self.goal_orientation is None:
                raise RuntimeError("STEP_ACTION missing goal_orientation for rotate")
            controller_action = [np.array(self.goal_orientation, dtype=np.float32)]
            runner.step(actions=[{robot_name: {"rotate": controller_action}}], render=True)
            return

        if self.controller_name is None and action.action_type == "wait":
            runner.step(actions=[{robot_name: {"move_by_speed": (0.0, 0.0, 0.0)}}], render=True)
            return

        if action.action_type == "stop":
            return
        raise RuntimeError(f"STEP_ACTION has unsupported controller: {self.controller_name}")

    def _progress(self, obs_after: Observation) -> bool:
        """Check whether the active step action has completed."""
        if self.remaining_steps > 0:
            self.remaining_steps -= 1
            return self.remaining_steps <= 0

        if self.start_pos is None or self.start_yaw is None:
            return True

        dx = obs_after.position[0] - self.start_pos[0]
        dy = obs_after.position[1] - self.start_pos[1]
        dist = float(np.sqrt(dx * dx + dy * dy))
        yaw_now = quat_to_yaw(obs_after.orientation)
        dyaw = abs(angle_diff(yaw_now, self.start_yaw))

        pos_eps = float(self.robot_profile.finish_pos_eps) if self.robot_profile is not None else 0.0
        ang_eps = (
            float(np.deg2rad(self.robot_profile.finish_rot_eps_deg))
            if self.robot_profile is not None
            else 0.0
        )

        reached_lin = self.target_lin > 0 and dist >= max(0.0, self.target_lin - pos_eps)
        if self.controller_name == "rotate" and self.goal_orientation is not None:
            yaw_goal = quat_to_yaw(self.goal_orientation)
            yaw_err = abs(angle_diff(yaw_now, yaw_goal))
            reached_ang = yaw_err <= max(0.0, ang_eps)
        else:
            reached_ang = self.target_ang > 0 and dyaw >= max(0.0, self.target_ang - ang_eps)

        return bool(reached_lin or reached_ang)
