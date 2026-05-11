"""Policy interface for VLM-style navigation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Observation:
    """Observation passed to policy at each step.

    Attributes:
        rgb: RGB image from camera (H, W, 3)
        depth: Depth map from camera (H, W)
        position: Robot position (x, y, z) in world coordinates
        orientation: Robot orientation as quaternion (w, x, y, z)
        instruction: Natural language task instruction
        step: Current step number in episode
        time_s: Elapsed time in seconds
        extra: Additional sensor data (extensible)
    """
    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    instruction: str = ""
    step: int = 0
    time_s: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """Action output from policy.

    Attributes:
        linear_velocity: Forward/backward speed (m/s) - for continuous control modes
        angular_velocity: Rotation speed (rad/s) - for continuous control modes
        lateral_velocity: Lateral speed (m/s), for omnidirectional robots
        stop: If True, request episode termination
        action_type: Discrete action type for STEP_ACTION mode ("forward", "left", "right", "stop")
                    When set, takes precedence over velocity-based inference
        extra: Additional action data (extensible)
    """
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0
    lateral_velocity: float = 0.0
    stop: bool = False
    action_type: Optional[str] = None  # "forward", "left", "right", "stop"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to (forward, lateral, angular) tuple for controller."""
        return (self.linear_velocity, self.lateral_velocity, self.angular_velocity)


class BasePolicy(ABC):
    """Abstract base class for navigation policies.

    VLM-style interface:
    - Input: Observation (images, robot state, instruction)
    - Output: Action (velocity command)

    Subclasses must implement `act()` method.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._history: List[Observation] = []
        self._instruction: str = ""
        self._max_history: int = self.config.get("max_history", 10)

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """Generate action given current observation.

        Args:
            observation: Current observation from environment

        Returns:
            Action to execute
        """
        raise NotImplementedError

    def observe(self, observation: Observation) -> None:
        """Consume a new observation frame (called every physics step by EpisodeRunner).

        Default behavior keeps a bounded history buffer. Policies that need continuous
        perception during long actions (e.g. waypoint execution) should override this.
        """
        self.update_history(observation)

    def reset(self, instruction: str = ""):
        """Reset policy state for new episode.

        Args:
            instruction: Task instruction for new episode
        """
        self._history.clear()
        self._instruction = instruction

    def update_history(self, observation: Observation):
        """Update observation history for context-aware policies."""
        self._history.append(observation)
        if len(self._history) > self._max_history:
            self._history.pop(0)

    @property
    def history(self) -> List[Observation]:
        """Get observation history."""
        return self._history

    @property
    def instruction(self) -> str:
        """Get current instruction."""
        return self._instruction
