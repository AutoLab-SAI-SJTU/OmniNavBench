from typing import Optional

from OmniNav.core.config.robot import ControllerCfg


class DifferentialDriveControllerCfg(ControllerCfg):

    type: Optional[str] = 'DifferentialDriveController'
    wheel_radius: float
    wheel_base: float


class DifferentialDriveMoveBySpeedControllerCfg(ControllerCfg):
    """Differential drive move-by-speed controller.

    Expects actions in (forward_speed, lateral_speed, rotation_speed) where lateral_speed is ignored.
    """

    type: Optional[str] = 'DifferentialDriveMoveBySpeedController'
    wheel_radius: float
    wheel_base: float
    forward_speed: Optional[float] = None
    rotation_speed: Optional[float] = None
