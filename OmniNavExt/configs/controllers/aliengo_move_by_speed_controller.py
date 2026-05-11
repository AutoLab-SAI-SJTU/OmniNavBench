from typing import List, Optional

from OmniNav.core.config.robot import ControllerCfg


class AliengoMoveBySpeedControllerCfg(ControllerCfg):

    type: Optional[str] = 'AliengoMoveBySpeedController'
    joint_names: List[str]
    policy_weights_path: str
    # Speed scaling factors applied to input commands
    forward_speed: Optional[float] = None
    rotation_speed: Optional[float] = None
