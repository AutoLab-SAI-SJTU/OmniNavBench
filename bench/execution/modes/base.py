from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from bench.configs.execution import ExecutionConfig, ExecutionMode, RobotExecutionProfile
from bench.policy.base import Action, Observation


def quat_to_yaw(quat: Tuple[float, float, float, float]) -> float:
    """Extract yaw (z-rotation) from quaternion (w, x, y, z)."""
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def angle_diff(a: float, b: float) -> float:
    """Return signed shortest angular difference a-b normalized to [-pi, pi]."""
    d = a - b
    return float((d + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class BaseActionExecutor:
    """Base state and interface for one high-level action executor."""

    mode: ExecutionMode
    robot_profile: Optional[RobotExecutionProfile]
    exec_config: ExecutionConfig

    def __post_init__(self) -> None:
        self.active_action: Optional[Action] = None
        self.finished: bool = True
        self.target_lin: float = 0.0
        self.target_ang: float = 0.0
        self.remaining_steps: int = 0
        self.start_pos: Optional[Tuple[float, float, float]] = None
        self.start_yaw: Optional[float] = None
        self.goal_position: Optional[Tuple[float, float, float]] = None
        self.goal_orientation: Optional[Tuple[float, float, float, float]] = None
        self.goal_threshold_m: Optional[float] = None
        self.goal_threshold_rad: Optional[float] = None
        self.controller_name: Optional[str] = None
        self.path_points: Optional[List[Tuple[float, float, float]]] = None
        self.theta_rad: Optional[float] = None
        self.r_m: Optional[float] = None
        self.command_id: int = 0
        self.stuck_check_pos: Optional[Tuple[float, float, float]] = None
        self.stuck_step_count: int = 0
        self.stuck_check_interval: int = 500
        self.stuck_threshold_m: float = 0.05

    def start(self, action: Action, obs: Observation, dt: float) -> None:
        raise NotImplementedError

    def execute(self, runner, robot_name: str, action: Action) -> None:
        raise NotImplementedError

    def _progress(self, obs_after: Observation) -> bool:
        raise NotImplementedError

    def step_finished(self, obs_after: Observation) -> bool:
        """Update internal state after one sim step."""
        if self.active_action is None:
            return True
        done = self._progress(obs_after)
        if done:
            self.finished = True
        return self.finished
