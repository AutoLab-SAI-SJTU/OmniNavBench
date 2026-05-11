from typing import List, Optional, Tuple

from OmniNav.core.config import RobotCfg
from OmniNav.core.config.robot import ControllerCfg
from OmniNav.core.config.sensor import SensorCfg
from OmniNav.macros import gm
from OmniNavExt.configs.controllers import (
    DifferentialDriveMoveBySpeedControllerCfg,
    GoTowardPointControllerCfg,
    MoveAlongPathPointsControllerCfg,
    MoveToPointBySpeedControllerCfg,
    RotateControllerCfg,
)
from OmniNavExt.configs.sensors import RepCameraCfg

move_by_speed_cfg = DifferentialDriveMoveBySpeedControllerCfg(name='move_by_speed', wheel_base=0.54, wheel_radius=0.24)

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

go_toward_point_cfg = GoTowardPointControllerCfg(
    name='go_toward_point',
    forward_speed=0.8,
    rotation_speed=1.2,
    yaw_threshold=0.02,
    dist_threshold=0.02,
    sub_controllers=[move_by_speed_cfg],
)

rotate_cfg = RotateControllerCfg(
    name='rotate',
    rotation_speed=2.0,
    threshold=0.02,
    sub_controllers=[move_by_speed_cfg],
)

camera_cfg = RepCameraCfg(
    name='camera',
    prim_path='chassis_link/camera_mount/carter_camera_first_person',
    depth = True,
    resolution=(640, 360),
    translation=None,
    orientation=None,
    # Values are in meters.
    clipping_range_m=(0.01, 1000.0),
)


class CarterV1RobotCfg(RobotCfg):
    # meta info
    name: Optional[str] = 'carter_v1'
    type: Optional[str] = 'CarterV1Robot'
    prim_path: Optional[str] = '/World/carter_v1'
    usd_path: Optional[str] = gm.ASSET_PATH + '/robots/carter/carter_v1.usd'

    # common config
    position: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.25)
    
    # default controllers and sensors
    controllers: Optional[List[ControllerCfg]] = [
        move_by_speed_cfg,
        move_to_point_cfg,
        move_along_path_cfg,
        go_toward_point_cfg,
        rotate_cfg,
    ]
    sensors: Optional[List[SensorCfg]] = [camera_cfg]
