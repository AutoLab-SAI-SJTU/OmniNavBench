from collections import OrderedDict
from typing import Any, List

import numpy as np

from OmniNav.core.robot.articulation_action import ArticulationAction
from OmniNav.core.robot.controller import BaseController
from OmniNav.core.robot.robot import BaseRobot
from OmniNav.core.scene.scene import IScene
from OmniNavExt.configs.controllers import RotateControllerCfg


def _quat_to_yaw(quat) -> float:
    """Return yaw (z-rotation) from quaternion assumed in (w, x, y, z)."""
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


@BaseController.register('RotateController')
class RotateController(BaseController):
    """Controller for turning to a certain orientation by utilizing a move-by-speed controller as sub-controller."""

    def __init__(self, config: RotateControllerCfg, robot: BaseRobot, scene: IScene) -> None:
        self._user_config = None
        self.goal_orientation: np.ndarray = None
        self.threshold: float = None

        self.rotation_speed = config.rotation_speed if config.rotation_speed is not None else 3.0
        self.threshold = config.threshold if config.threshold is not None else 0.02

        super().__init__(config=config, robot=robot, scene=scene)

    @staticmethod
    def get_delta_z_rot(
        start_orientation,
        goal_orientation,
    ) -> float:
        delta = _quat_to_yaw(goal_orientation) - _quat_to_yaw(start_orientation)
        delta = float((delta + np.pi) % (2 * np.pi) - np.pi)
        return delta

    def forward(
        self,
        start_orientation: np.ndarray,
        goal_orientation: np.ndarray,
        rotation_speed: float = 3,
        threshold: float = 0.02,
    ) -> ArticulationAction:
        self.goal_orientation = goal_orientation
        self.threshold = threshold

        delta_z_rot = RotateController.get_delta_z_rot(
            start_orientation=start_orientation, goal_orientation=goal_orientation
        )
        if abs(delta_z_rot) < threshold:
            delta_z_rot = 0

        # Never rotate faster than rotation_speed.
        # Use proportional control with minimum speed to ensure rotation completes.
        # For small angles, use a minimum speed threshold (e.g., 30% of max speed)
        # to prevent extremely slow rotation that may never reach the target.
        min_speed_factor = 0.3  # Minimum 30% of max rotation speed
        if abs(delta_z_rot) < 1.0:
            # Proportional control: scale speed by angle, but enforce minimum
            speed_factor = max(abs(delta_z_rot), min_speed_factor)
            rotation_speed *= speed_factor * np.sign(delta_z_rot)
        else:
            rotation_speed *= np.sign(delta_z_rot)

        return self.sub_controllers[0].forward(
            forward_speed=0.0,
            rotation_speed=rotation_speed,
        )

    def action_to_control(self, action: List | np.ndarray) -> ArticulationAction:
        """
        Args:
            action (List | np.ndarray): n-element 1d array containing:
              0. goal_orientation in quat (np.ndarray)
        """
        assert len(action) == 1, 'action must contain 1 elements'
        start_orientation = self.robot.get_pose()[1]
        return self.forward(
            start_orientation=start_orientation,
            goal_orientation=action[0],
            rotation_speed=self.rotation_speed,
            threshold=self.threshold,
        )

    def get_obs(self) -> OrderedDict[str, Any]:
        if self.goal_orientation is None or self.threshold is None:
            return {}
        start_orientation = self.robot.get_pose()[1]
        delta_z_rot = RotateController.get_delta_z_rot(
            start_orientation=start_orientation, goal_orientation=self.goal_orientation
        )
        finished = True if abs(delta_z_rot) < self.threshold else False
        obs = {
            'finished': finished,
        }
        return self._make_ordered(obs)


# Use class-var inject controllers types' class
BaseController.controllers['RotateController'] = RotateController
