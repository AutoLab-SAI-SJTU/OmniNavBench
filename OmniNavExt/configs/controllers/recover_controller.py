from typing import Optional

from OmniNav.core.config.robot import ControllerCfg


class RecoverControllerCfg(ControllerCfg):

    type: Optional[str] = 'RecoverController'
    recover_height: float
