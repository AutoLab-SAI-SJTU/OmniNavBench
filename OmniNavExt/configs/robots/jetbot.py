from typing import Optional

from OmniNav.core.config import RobotCfg
from OmniNav.macros import gm
from OmniNavExt.configs.controllers import (
    DifferentialDriveMoveBySpeedControllerCfg,
    MoveAlongPathPointsControllerCfg,
    MoveToPointBySpeedControllerCfg,
    RotateControllerCfg,
)
from OmniNavExt.configs.sensors import RepCameraCfg

move_by_speed_cfg = DifferentialDriveMoveBySpeedControllerCfg(name='move_by_speed', wheel_base=0.1125, wheel_radius=0.03)

move_to_point_cfg = MoveToPointBySpeedControllerCfg(
    name='move_to_point',
    forward_speed=1.0,
    rotation_speed=1.0,
    threshold=0.1,
    sub_controllers=[move_by_speed_cfg],
)

move_along_path_cfg = MoveAlongPathPointsControllerCfg(
    name='move_along_path',
    forward_speed=1.0,
    rotation_speed=1.0,
    threshold=0.1,
    sub_controllers=[move_to_point_cfg],
)

rotate_cfg = RotateControllerCfg(
    name='rotate',
    rotation_speed=2.0,
    threshold=0.02,
    sub_controllers=[move_by_speed_cfg],
)

camera_cfg = RepCameraCfg(
    name='camera',
    prim_path='chassis/rgb_camera/jetbot_camera',
    resolution=(640, 360),
    # Values are in meters.
    clipping_range_m=(0.01, 1000.0),
)


class JetbotRobotCfg(RobotCfg):
    # meta info
    name: Optional[str] = 'jetbot'
    type: Optional[str] = 'JetbotRobot'
    prim_path: Optional[str] = '/World/jetbot'
    usd_path: Optional[str] = gm.ASSET_PATH + '/robots/jetbot/jetbot.usd'
