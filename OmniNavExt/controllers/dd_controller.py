from typing import List

import numpy as np

from OmniNav.core.robot.articulation_action import ArticulationAction
from OmniNav.core.robot.controller import BaseController
from OmniNav.core.robot.robot import BaseRobot
from OmniNav.core.scene.scene import IScene
from OmniNavExt.configs.controllers import DifferentialDriveControllerCfg
from OmniNavExt.configs.controllers.dd_controller import DifferentialDriveMoveBySpeedControllerCfg


@BaseController.register('DifferentialDriveController')
class DifferentialDriveController(BaseController):
    def __init__(self, config: DifferentialDriveControllerCfg, robot: BaseRobot, scene: IScene) -> None:
        super().__init__(config=config, robot=robot, scene=scene)
        self._robot_scale = self.robot.get_robot_scale()[0]
        self._wheel_base = config.wheel_base * self._robot_scale
        self._wheel_radius = config.wheel_radius * self._robot_scale

    def forward(
        self,
        forward_speed: float = 0,
        rotation_speed: float = 0,
    ) -> ArticulationAction:
        left_wheel_vel = ((2 * forward_speed) - (rotation_speed * self._wheel_base)) / (2 * self._wheel_radius)
        right_wheel_vel = ((2 * forward_speed) + (rotation_speed * self._wheel_base)) / (2 * self._wheel_radius)
        # A controller has to return an ArticulationAction
        return ArticulationAction(joint_velocities=[left_wheel_vel, right_wheel_vel])

    def action_to_control(self, action: List | np.ndarray) -> ArticulationAction:
        """
        Args:
            action (List | np.ndarray): n-element 1d array containing:
              0. forward_speed (float)
              1. rotation_speed (float)
        """
        assert len(action) == 2, 'action must contain 2 elements'
        return self.forward(
            forward_speed=action[0],
            rotation_speed=action[1],
        )


@BaseController.register('DifferentialDriveMoveBySpeedController')
class DifferentialDriveMoveBySpeedController(BaseController):
    """Move-by-speed controller for differential drive robots.

    This controller matches the OmniNavBench velocity action convention:
    (forward_speed, lateral_speed, rotation_speed). The lateral component is ignored.
    """

    def __init__(self, config: DifferentialDriveMoveBySpeedControllerCfg, robot: BaseRobot, scene: IScene) -> None:
        super().__init__(config=config, robot=robot, scene=scene)
        self._robot_scale = self.robot.get_robot_scale()[0]
        self._wheel_base = config.wheel_base * self._robot_scale
        self._wheel_radius = config.wheel_radius * self._robot_scale
        self._forward_speed_scale = float(config.forward_speed) if config.forward_speed is not None else 1.0
        self._rotation_speed_scale = float(config.rotation_speed) if config.rotation_speed is not None else 1.0

    def forward(
        self,
        forward_speed: float = 0.0,
        rotation_speed: float = 0.0,
        lateral_speed: float = 0.0,
    ) -> ArticulationAction:
        left_wheel_vel = ((2 * forward_speed) - (rotation_speed * self._wheel_base)) / (2 * self._wheel_radius)
        right_wheel_vel = ((2 * forward_speed) + (rotation_speed * self._wheel_base)) / (2 * self._wheel_radius)
        return ArticulationAction(joint_velocities=[left_wheel_vel, right_wheel_vel])

    def action_to_control(self, action: List | np.ndarray) -> ArticulationAction:
        """
        Args:
            action (List | np.ndarray): 3-element 1d array containing:
              0. forward_speed (float)
              1. lateral_speed (float) (ignored)
              2. rotation_speed (float)
        """
        assert len(action) == 3, 'action must contain 3 elements'
        forward_speed = float(action[0]) * self._forward_speed_scale
        rotation_speed = float(action[2]) * self._rotation_speed_scale
        return self.forward(
            forward_speed=forward_speed,
            rotation_speed=rotation_speed,
        )

    @property
    def forward_speed(self) -> float:
        return float(self._forward_speed_scale)

    @property
    def rotation_speed(self) -> float:
        return float(self._rotation_speed_scale)
