from typing import Optional

from OmniNav.core.config.robot import ControllerCfg


class GoTowardPointControllerCfg(ControllerCfg):
    type: Optional[str] = 'GoTowardPointController'
    forward_speed: Optional[float] = None
    rotation_speed: Optional[float] = None
    yaw_threshold: Optional[float] = None
    dist_threshold: Optional[float] = None
