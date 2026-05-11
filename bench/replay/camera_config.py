from __future__ import annotations

from typing import List

from OmniNav.core.config import RobotCfg
from OmniNavExt.configs.sensors import RepCameraCfg


def configure_replay_robot_sensors(robot_cfg: RobotCfg, *, enable_depth: bool = True) -> None:
    """Configure replay-specific camera sensors by robot type.

    Replay uses its own camera config instead of borrowing policy-local configs.
    """
    robot_type = str(robot_cfg.type or "")
    if robot_type == "AliengoRobot":
        robot_cfg.sensors = _aliengo_sensors(enable_depth=enable_depth)
        return
    if robot_type == "CarterV1Robot":
        robot_cfg.sensors = _carter_v1_sensors(enable_depth=enable_depth)
        return
    if robot_type == "H1Robot":
        robot_cfg.sensors = _h1_sensors(enable_depth=enable_depth)
        return
    if robot_type == "KITT15Robot":
        robot_cfg.sensors = _kitt15_sensors(enable_depth=enable_depth)
        return

    raise RuntimeError(
        f"[replay.camera_config] Unknown robot_type='{robot_type}', cannot configure replay cameras"
    )


def _aliengo_sensors(*, enable_depth: bool) -> List[RepCameraCfg]:
    front_camera = RepCameraCfg(
        name="front",
        prim_path="trunk/Camera_Front",
        resolution=(720, 640),
        depth=enable_depth,
        camera_params=True,
        translation=(0.35, 0.0, 0.0),
        orientation=(0.454519, 0.541675, -0.541675, -0.454519),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=110.0,
    )
    left_camera = RepCameraCfg(
        name="left",
        prim_path="trunk/Camera_Left",
        resolution=(720, 640),
        depth=enable_depth,
        camera_params=True,
        translation=(0.35, 0.30, 0.0),
        orientation=(0.642788, 0.766044, 0.0, 0.0),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=110.0,
    )
    right_camera = RepCameraCfg(
        name="right",
        prim_path="trunk/Camera_Right",
        resolution=(720, 640),
        depth=enable_depth,
        camera_params=True,
        translation=(0.35, -0.30, 0.0),
        orientation=(-0.0, 0.0, 0.766044, 0.642788),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=110.0,
    )
    return [front_camera, left_camera, right_camera]


# def _carter_v1_sensors(*, enable_depth: bool) -> List[RepCameraCfg]:
#     front_camera = RepCameraCfg(
#         name="front",
#         prim_path="chassis_link/camera_mount/Camera_Front",
#         resolution=(720, 640),
#         depth=enable_depth,
#         camera_params=True,
#         translation=(0.089172, 0.0, 0.326497),
#         orientation=(-0.5, -0.5, 0.5, 0.5),
#         clipping_range_m=(0.01, 1000.0),
#         fov_degrees=110.0,
#     )
#     left_camera = RepCameraCfg(
#         name="left",
#         prim_path="chassis_link/camera_mount/Camera_Left",
#         resolution=(720, 640),
#         depth=enable_depth,
#         camera_params=True,
#         translation=(0.103, 0.30, 0.309),
#         orientation=(0.707, 0.707, 0.0, 0.0),
#         clipping_range_m=(0.01, 1000.0),
#         fov_degrees=110.0,
#     )
#     right_camera = RepCameraCfg(
#         name="right",
#         prim_path="chassis_link/camera_mount/Camera_Right",
#         resolution=(720, 640),
#         depth=enable_depth,
#         camera_params=True,
#         translation=(0.103, -0.3, 0.309),
#         orientation=(0.0, 0.0, 0.707, 0.707),
#         clipping_range_m=(0.01, 1000.0),
#         fov_degrees=110.0,
#     )
#     return [front_camera, left_camera, right_camera]

def _carter_v1_sensors(*, enable_depth: bool) -> List[RepCameraCfg]:
    """Camera configuration for CarterV1 wheeled robot."""
    camera_cfg = RepCameraCfg(
        name="camera",
        prim_path="chassis_link/camera_mount/carter_camera_first_person",
        depth=True,
        camera_params=True,
        translation=(0.0,0.0,0.7),
        orientation=None,
        resolution=(640, 480),  # VLN-CE: 640x480
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=58,  # VLN-CE: HFOV=120
    )
    return [camera_cfg]


def _kitt15_sensors(*, enable_depth: bool) -> List[RepCameraCfg]:
    """Camera configuration for KITT15 wheeled mobile manipulator."""
    camera_cfg = RepCameraCfg(
        name="camera",
        prim_path="cam_install_head/camera_link/head_camera",
        depth=enable_depth,
        camera_params=True,
        resolution=(640, 480),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=58,
        # Rotate 180 deg around forward axis to flip the upside-down image
        orientation=(0.0, 1.0, 0.0, 0.0),  # (w,x,y,z) = 180 deg around X (flip up + flip forward)
    )
    return [camera_cfg]


def _h1_sensors(*, enable_depth: bool) -> List[RepCameraCfg]:
    front_camera = RepCameraCfg(
        name="front",
        prim_path="logo_link/Camera_Front",
        resolution=(720, 640),
        depth=enable_depth,
        camera_params=True,
        translation=(0.083626, 0.0, 0.390786),
        orientation=(0.5, 0.5, -0.5, -0.5),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=110.0,
    )
    left_camera = RepCameraCfg(
        name="left",
        prim_path="logo_link/Camera_Left",
        resolution=(720, 640),
        depth=enable_depth,
        camera_params=True,
        translation=(0.083630, 0.300000, 0.390790),
        orientation=(0.707107, 0.707107, 0.000000, 0.000000),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=110.0,
    )
    right_camera = RepCameraCfg(
        name="right",
        prim_path="logo_link/Camera_Right",
        resolution=(720, 640),
        depth=enable_depth,
        camera_params=True,
        translation=(0.083630, -0.300000, 0.390790),
        orientation=(-0.000000, 0.000000, 0.707107, 0.707107),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=110.0,
    )
    return [front_camera, left_camera, right_camera]
