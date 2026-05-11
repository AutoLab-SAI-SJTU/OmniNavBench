from typing import Optional

from OmniNav.core.config.robot import ControllerCfg


class RotateControllerCfg(ControllerCfg):
    type: Optional[str] = 'RotateController'
    rotation_speed: Optional[float] = None
    threshold: Optional[float] = None
