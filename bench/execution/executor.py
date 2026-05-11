from __future__ import annotations

from typing import Optional

from bench.configs.execution import ExecutionConfig, ExecutionMode, RobotExecutionProfile
from bench.execution.modes.base import BaseActionExecutor
from bench.execution.modes.step_action import StepActionExecutor
from bench.execution.modes.waypoint import WaypointExecutor


def create_executor(
    mode: ExecutionMode,
    robot_profile: Optional[RobotExecutionProfile],
    exec_config: ExecutionConfig,
) -> BaseActionExecutor:
    """Create an executor for the configured policy execution mode."""
    if mode == ExecutionMode.STEP_ACTION:
        return StepActionExecutor(mode=mode, robot_profile=robot_profile, exec_config=exec_config)
    if mode == ExecutionMode.WAYPOINT:
        return WaypointExecutor(mode=mode, robot_profile=robot_profile, exec_config=exec_config)
    raise ValueError(f"Unsupported execution mode: {mode}")
