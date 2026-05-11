"""Termination conditions for episode evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import time

if TYPE_CHECKING:
    from ..policy.base import Observation


@dataclass
class TerminationResult:
    """Result of termination check.

    Attributes:
        terminated: Whether episode should terminate
        success: Whether goal was achieved (None if not applicable)
        reason: Human-readable termination reason
    """
    terminated: bool
    success: bool | None = None
    reason: str = ""


class TerminationCondition(ABC):
    """Abstract base class for termination conditions."""

    @abstractmethod
    def check(self, observation: "Observation") -> TerminationResult:
        """Check if episode should terminate.

        Args:
            observation: Current observation from environment

        Returns:
            TerminationResult with termination status
        """
        raise NotImplementedError

    def reset(self):
        """Reset condition state for new episode."""
        pass


class GoalReachedCondition(TerminationCondition):
    """Terminate when robot reaches goal position.

    Uses Euclidean distance in 2D (x, y) plane.

    For return tasks (where start position is near goal position),
    the robot must first leave the goal area before returning to it for success.
    """

    def __init__(
        self,
        goal_position: tuple[float, float, float],
        success_threshold: float = 1.0,
        require_leave_first: bool = False,
    ):
        """Initialize goal reached condition.

        Args:
            goal_position: Target (x, y, z) position
            success_threshold: Distance threshold in meters for success
            require_leave_first: If True, robot must leave goal area before success can be triggered
        """
        self.goal_position = goal_position
        self.success_threshold = success_threshold
        self.require_leave_first = require_leave_first
        # Leave threshold = success_threshold + 2m
        self.leave_threshold = success_threshold + 2.0
        self._ever_left_goal_area = False

    def check(self, observation: "Observation") -> TerminationResult:
        # 2D distance (ignore z)
        dx = observation.position[0] - self.goal_position[0]
        dy = observation.position[1] - self.goal_position[1]
        distance = np.sqrt(dx * dx + dy * dy)

        # Track if robot has ever left the goal area
        if distance > self.leave_threshold:
            if not self._ever_left_goal_area:
                self._ever_left_goal_area = True
                print(f"[GoalReached] Left goal area (distance={distance:.2f}m > {self.leave_threshold:.2f}m)")

        in_goal_area = distance <= self.success_threshold

        if in_goal_area:
            # Success only if: not require_leave_first OR already left once
            if not self.require_leave_first or self._ever_left_goal_area:
                return TerminationResult(
                    terminated=True,
                    success=True,
                    reason=f"Goal reached (distance={distance:.2f}m)"
                )

        return TerminationResult(terminated=False)

    def reset(self):
        """Reset condition state for new episode."""
        self._ever_left_goal_area = False


class TimeoutCondition(TerminationCondition):
    """Terminate when step limit exceeded."""

    def __init__(self, max_steps: int, max_wall_time_s: float | None = None):
        """Initialize timeout condition.

        Args:
            max_steps: Maximum number of steps
            max_wall_time_s: Optional wall-clock timeout in seconds
        """
        self.max_steps = max_steps
        self.max_wall_time_s = None if max_wall_time_s is None else float(max_wall_time_s)
        self._start_wall_time: float | None = None

        if self.max_wall_time_s is not None:
            if not np.isfinite(self.max_wall_time_s) or self.max_wall_time_s <= 0:
                raise ValueError(f"max_wall_time_s must be > 0, got {max_wall_time_s!r}")

    def check(self, observation: "Observation") -> TerminationResult:
        current_wall_time = time.time()
        if self._start_wall_time is None:
            self._start_wall_time = current_wall_time

        # Check step limit
        if observation.step >= self.max_steps:
            return TerminationResult(
                terminated=True,
                success=False,
                reason=f"Step limit reached ({observation.step} >= {self.max_steps})"
            )

        if self.max_wall_time_s is not None:
            elapsed_wall_time = current_wall_time - self._start_wall_time
            if elapsed_wall_time >= self.max_wall_time_s:
                return TerminationResult(
                    terminated=True,
                    success=False,
                    reason=(
                        f"Wall-clock timeout reached "
                        f"({elapsed_wall_time:.2f}s >= {self.max_wall_time_s:.2f}s)"
                    ),
                )

        return TerminationResult(terminated=False)

    def reset(self):
        self._start_wall_time = None

class StuckCondition(TerminationCondition):
    """Terminate when agent is stuck (no meaningful XY movement for a while).

    "Simulator time" is taken from Observation.time_s, which is computed as
    step * dt in EpisodeRunner. We consider the agent "not moving" if its
    XY displacement from the last anchor position stays within
    `move_threshold_m` for at least `duration_s` seconds of simulator time
    OR `wall_clock_duration_s` seconds of real world time.
    """

    def __init__(
        self,
        duration_s: float = 60.0,
        move_threshold_m: float = 0.1,
        wall_clock_duration_s: float = 120.0
    ):
        self.duration_s = float(duration_s)
        self.move_threshold_m = float(move_threshold_m)
        self.wall_clock_duration_s = float(wall_clock_duration_s)

        if not np.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError(f"duration_s must be > 0, got {duration_s!r}")
        if not np.isfinite(self.move_threshold_m) or self.move_threshold_m < 0:
            raise ValueError(f"move_threshold_m must be >= 0, got {move_threshold_m!r}")
        if not np.isfinite(self.wall_clock_duration_s) or self.wall_clock_duration_s <= 0:
            raise ValueError(f"wall_clock_duration_s must be > 0, got {wall_clock_duration_s!r}")

        self._anchor_time_s: float | None = None
        self._anchor_wall_time: float | None = None
        self._anchor_xy: tuple[float, float] | None = None

    def check(self, observation: "Observation") -> TerminationResult:
        current_wall_time = time.time()

        if self._anchor_time_s is None or self._anchor_xy is None or self._anchor_wall_time is None:
            self._anchor_time_s = float(observation.time_s)
            self._anchor_wall_time = current_wall_time
            self._anchor_xy = (float(observation.position[0]), float(observation.position[1]))
            return TerminationResult(terminated=False)

        x = float(observation.position[0])
        y = float(observation.position[1])
        dx = x - self._anchor_xy[0]
        dy = y - self._anchor_xy[1]
        dist_xy = float(np.sqrt(dx * dx + dy * dy))

        if dist_xy > self.move_threshold_m:
            self._anchor_time_s = float(observation.time_s)
            self._anchor_wall_time = current_wall_time
            self._anchor_xy = (x, y)
            return TerminationResult(terminated=False)

        # Check simulator time
        elapsed_s = float(observation.time_s) - self._anchor_time_s
        if elapsed_s >= self.duration_s:
            return TerminationResult(
                terminated=True,
                success=None,
                reason=(
                    f"Stuck (Sim Time): no XY movement > {self.move_threshold_m:.2f}m for "
                    f"{elapsed_s:.2f}s (threshold={self.duration_s:.2f}s)"
                ),
            )

        # Check wall clock time
        elapsed_wall = current_wall_time - self._anchor_wall_time
        if elapsed_wall >= self.wall_clock_duration_s:
            return TerminationResult(
                terminated=True,
                success=None,
                reason=(
                    f"Stuck (Wall Time): no XY movement > {self.move_threshold_m:.2f}m for "
                    f"{elapsed_wall:.2f}s (threshold={self.wall_clock_duration_s:.2f}s)"
                ),
            )

        return TerminationResult(terminated=False)

    def reset(self):
        self._anchor_time_s = None
        self._anchor_wall_time = None
        self._anchor_xy = None


class StopActionCondition(TerminationCondition):
    """Terminate when policy issues stop action."""

    def __init__(self):
        self._stop_requested = False

    def set_stop_requested(self, value: bool = True):
        """Called by runner when policy returns stop=True."""
        self._stop_requested = value

    def check(self, observation: "Observation") -> TerminationResult:
        if self._stop_requested:
            return TerminationResult(
                terminated=True,
                success=None,  # Success determined by other conditions
                reason="Policy requested stop"
            )
        return TerminationResult(terminated=False)

    def reset(self):
        self._stop_requested = False


class CompositeCondition(TerminationCondition):
    """Combines multiple termination conditions (OR logic)."""

    def __init__(self, conditions: list[TerminationCondition]):
        """Initialize with list of conditions.

        Args:
            conditions: List of termination conditions to check
        """
        self.conditions = conditions

    def check(self, observation: "Observation") -> TerminationResult:
        for condition in self.conditions:
            result = condition.check(observation)
            if result.terminated:
                return result
        return TerminationResult(terminated=False)

    def reset(self):
        for condition in self.conditions:
            condition.reset()
