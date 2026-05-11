from collections import OrderedDict
from typing import Any, List

import numpy as np

from OmniNav.core.robot.articulation_action import ArticulationAction
from OmniNav.core.robot.controller import BaseController
from OmniNav.core.robot.robot import BaseRobot
from OmniNav.core.scene.scene import IScene
from OmniNavExt.configs.controllers import GoTowardPointControllerCfg


def _quat_to_yaw(quat) -> float:
    """Return yaw (z-rotation) from quaternion assumed in (w, x, y, z)."""
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _angle_diff(a: float, b: float) -> float:
    """Return signed shortest angular difference a-b in [-pi, pi]."""
    d = a - b
    return float((d + np.pi) % (2 * np.pi) - np.pi)


@BaseController.register('GoTowardPointController')
class GoTowardPointController(BaseController):
    """Controller that executes a relative (theta, r) command in two phases:
    1. Move to the target position
    2. Rotate to the target orientation
    """

    def __init__(self, config: GoTowardPointControllerCfg, robot: BaseRobot, scene: IScene) -> None:
        self._last_command_id: int | None = None
        self._last_command: tuple[float, float] | None = None
        self._goal_yaw: float | None = None
        self._goal_position: np.ndarray | None = None
        self._phase: str = "move"  # "move" -> "rotate" -> "done"

        self.forward_speed = config.forward_speed if config.forward_speed is not None else 1.0
        self.rotation_speed = config.rotation_speed if config.rotation_speed is not None else 1.0
        self.yaw_threshold = config.yaw_threshold if config.yaw_threshold is not None else 0.02
        self.dist_threshold = config.dist_threshold if config.dist_threshold is not None else 0.02

        super().__init__(config=config, robot=robot, scene=scene)

    def _reset_goal(self, start_position: np.ndarray, start_yaw: float, theta: float, r: float) -> None:
        """Set goal position and orientation.

        Args:
            start_position: Current robot position
            start_yaw: Current robot yaw
            theta: Target orientation (relative to current yaw)
            r: Distance to target point

        The target position is computed as:
            - Direction to target = current_yaw + theta
            - Target position = current_position + (cos(direction) * r, sin(direction) * r)
        The target orientation is also current_yaw + theta.
        """
        self._goal_yaw = start_yaw + theta
        # Compute target position using absolute direction
        direction = start_yaw + theta
        dx = float(np.cos(direction) * r)
        dy = float(np.sin(direction) * r)
        self._goal_position = np.array([start_position[0] + dx, start_position[1] + dy, start_position[2]])
        self._phase = "move"

    @staticmethod
    def _heading_to_goal(start_position: np.ndarray, goal_position: np.ndarray) -> float:
        vec = np.array([goal_position[0] - start_position[0], goal_position[1] - start_position[1]], dtype=np.float32)
        if np.linalg.norm(vec) == 0:
            return 0.0
        return float(np.arctan2(vec[1], vec[0]))

    def forward(
        self,
        start_position: np.ndarray,
        start_orientation: np.ndarray,
        goal_position: np.ndarray,
        goal_yaw: float,
        forward_speed: float = 1.0,
        rotation_speed: float = 1.0,
        yaw_threshold: float = 0.02,
        dist_threshold: float = 0.02,
    ) -> ArticulationAction:
        # Distance in XY plane
        start_xy = np.array([start_position[0], start_position[1]], dtype=np.float32)
        goal_xy = np.array([goal_position[0], goal_position[1]], dtype=np.float32)
        dist_err = float(np.linalg.norm(start_xy - goal_xy))

        yaw_now = _quat_to_yaw(start_orientation)

        # Phase 1: Move to target position
        if self._phase == "move":
            if dist_err < dist_threshold:
                # Reached position, switch to rotate phase
                self._phase = "rotate"
            else:
                # Move toward goal position
                target_heading = self._heading_to_goal(start_position, goal_position)
                yaw_err = _angle_diff(target_heading, yaw_now)

                if abs(yaw_err) < yaw_threshold:
                    yaw_err = 0.0

                # Forward speed scaled by heading error
                if abs(yaw_err) > np.pi / 2:
                    forward_cmd = 0.0
                else:
                    forward_cmd = forward_speed
                    if dist_err < dist_threshold * 2:
                        forward_cmd *= (dist_err / (dist_threshold * 2)) ** 2
                    forward_cmd *= (1 - (abs(yaw_err) * 2 / np.pi)) ** 3

                # Rotation to face goal position
                if yaw_err == 0.0:
                    rotation_cmd = 0.0
                else:
                    rotation_cmd = rotation_speed
                    if abs(yaw_err) < 1.0:
                        rotation_cmd *= max(abs(yaw_err), 0.3) * np.sign(yaw_err)
                    else:
                        rotation_cmd *= np.sign(yaw_err)

                return self.sub_controllers[0].forward(
                    forward_speed=forward_cmd,
                    rotation_speed=rotation_cmd,
                    lateral_speed=0.0,
                )

        # Phase 2: Rotate to target orientation
        if self._phase == "rotate":
            yaw_err = _angle_diff(goal_yaw, yaw_now)

            if abs(yaw_err) < yaw_threshold:
                # Reached target orientation
                self._phase = "done"
                return self.sub_controllers[0].forward(
                    forward_speed=0.0,
                    rotation_speed=0.0,
                    lateral_speed=0.0,
                )

            # Pure rotation
            rotation_cmd = rotation_speed
            if abs(yaw_err) < 1.0:
                rotation_cmd *= max(abs(yaw_err), 0.3) * np.sign(yaw_err)
            else:
                rotation_cmd *= np.sign(yaw_err)

            return self.sub_controllers[0].forward(
                forward_speed=0.0,
                rotation_speed=rotation_cmd,
                lateral_speed=0.0,
            )

        # Phase 3: Done - stop
        return self.sub_controllers[0].forward(
            forward_speed=0.0,
            rotation_speed=0.0,
            lateral_speed=0.0,
        )

    def action_to_control(self, action: List | np.ndarray) -> ArticulationAction:
        """Convert (theta, r[, command_id]) to joint signals."""
        if len(action) not in (2, 3):
            raise ValueError('action must contain 2 or 3 elements')
        theta = float(action[0])
        r = float(action[1])
        command_id = int(action[2]) if len(action) == 3 else None

        start_position, start_orientation = self.robot.get_pose()
        start_yaw = _quat_to_yaw(start_orientation)

        if command_id is not None:
            if self._last_command_id != command_id:
                self._last_command_id = command_id
                self._last_command = (theta, r)
                self._reset_goal(start_position, start_yaw, theta, r)
        else:
            if self._last_command is None or not np.allclose(
                np.array([theta, r], dtype=np.float32),
                np.array(self._last_command, dtype=np.float32),
                atol=1e-6,
            ):
                self._last_command = (theta, r)
                self._reset_goal(start_position, start_yaw, theta, r)

        if self._goal_position is None:
            self._reset_goal(start_position, start_yaw, theta, r)

        return self.forward(
            start_position=start_position,
            start_orientation=start_orientation,
            goal_position=self._goal_position,
            goal_yaw=self._goal_yaw,
            forward_speed=self.forward_speed,
            rotation_speed=self.rotation_speed,
            yaw_threshold=self.yaw_threshold,
            dist_threshold=self.dist_threshold,
        )

    def get_obs(self) -> OrderedDict[str, Any]:
        obs = {
            "goal_position": self._goal_position,
            "goal_yaw": self._goal_yaw,
            "phase": self._phase,
        }
        return self._make_ordered(obs)
