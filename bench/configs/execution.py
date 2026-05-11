from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class ExecutionMode(str, Enum):
    """Execution modes for policy outputs."""

    STEP_ACTION = "step_action"
    WAYPOINT = "waypoint"


@dataclass
class EarlyStopConfig:
    """Thresholds for early-stop checks."""

    success_radius: float = 1.0
    max_steps: int | None = None

    # Rot/pos stagnation
    rot_stuck_deg: float = 1.0
    pos_stuck_m: float = 0.02
    rot_stuck_frames: int = 30

    dist_no_progress_m: float = 0.01
    dist_no_progress_frames: int = 20

    pos_stall_m: float = 0.02
    pos_stall_frames: int = 20


@dataclass
class ExecutionConfig:
    """Global execution defaults."""

    early_stop: EarlyStopConfig = field(default_factory=EarlyStopConfig)
    policy_mode_map: Dict[str, ExecutionMode] = field(default_factory=dict)


@dataclass
class RobotExecutionProfile:
    """Per-robot execution parameters."""

    max_lin_vel: float
    max_ang_vel: float
    step_lin_dist: float
    step_ang_deg: float
    finish_pos_eps: float
    finish_rot_eps_deg: float
