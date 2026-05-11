from typing import Optional

from OmniNav.core.config.robot import ControllerCfg


class GripperControllerCfg(ControllerCfg):
    type: Optional[str] = 'GripperController'
