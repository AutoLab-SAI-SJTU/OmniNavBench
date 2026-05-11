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

# KITT15 wheel parameters (from URDF geometry analysis)
KITT15_WHEEL_BASE = 0.442    # track width in meters
KITT15_WHEEL_RADIUS = 0.058  # wheel radius in meters

move_by_speed_cfg = DifferentialDriveMoveBySpeedControllerCfg(
    name='move_by_speed',
    wheel_base=KITT15_WHEEL_BASE,
    wheel_radius=KITT15_WHEEL_RADIUS,
)

move_to_point_cfg = MoveToPointBySpeedControllerCfg(
    name='move_to_point',
    forward_speed=0.5,
    rotation_speed=1.0,
    threshold=0.1,
    sub_controllers=[move_by_speed_cfg],
)

move_along_path_cfg = MoveAlongPathPointsControllerCfg(
    name='move_along_path',
    forward_speed=0.5,
    rotation_speed=1.0,
    threshold=0.1,
    sub_controllers=[move_to_point_cfg],
)

go_toward_point_cfg = GoTowardPointControllerCfg(
    name='go_toward_point',
    forward_speed=0.5,
    rotation_speed=1.0,
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

# Head camera (mounted on cam_install_head)
kitt15_camera_cfg = RepCameraCfg(
    name='camera',
    prim_path='cam_install_head/camera_link/head_camera',
    resolution=(640, 480),
    depth=True,
    clipping_range_m=(0.01, 1000.0),
    orientation=(0.0, 1.0, 0.0, 0.0),  # 180 deg around X (flip up + flip forward)
)


class KITT15RobotCfg(RobotCfg):
    name: Optional[str] = 'kitt15'
    type: Optional[str] = 'KITT15Robot'
    prim_path: Optional[str] = '/kitt15'
    usd_path: Optional[str] = gm.ASSET_PATH + '/robots/kitt15/kitt15.usd'

    position: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.5)

    controllers: Optional[List[ControllerCfg]] = [
        move_by_speed_cfg,
        move_to_point_cfg,
        move_along_path_cfg,
        go_toward_point_cfg,
        rotate_cfg,
    ]
    sensors: Optional[List[SensorCfg]] = [kitt15_camera_cfg]
