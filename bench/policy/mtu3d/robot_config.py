from __future__ import annotations

from typing import List

from OmniNav.core.config import RobotCfg
from OmniNavExt.configs.sensors import RepCameraCfg


# How many of the ring-mounted cameras MTU3D should consume.
#
#   1  -> original MTU3D paper pattern: single forward-facing `camera_0` view,
#         the robot rotates in place at episode start to collect 12 yaw steps
#         (the policy's spin phase is enabled).
#   12 -> use the full 12-camera ring at once, skip the spin phase.
#
# Other values (2..11) are accepted but untested; the policy treats anything
# >1 as multi-camera mode and skips the spin.
NUM_CAMERAS: int = 1


def get_required_cameras(robot_cfg: RobotCfg) -> List[str]:
    """Return the list of camera names EpisodeRunner must capture for MTU3D.

    Selects `camera_0` ... `camera_<NUM_CAMERAS-1>` from the 12 RepCameraCfg
    entries each robot defines.
    """
    sensors = list(robot_cfg.sensors or [])
    names: List[str] = []
    for s in sensors:
        if getattr(s, "type", None) != "RepCamera":
            continue
        name = getattr(s, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("[mtu3d.robot_config] Camera SensorCfg.name must be a non-empty string")
        name = name.strip()
        if name not in names:
            names.append(name)

    required_names = [f"camera_{i}" for i in range(NUM_CAMERAS)]
    missing_names = [name for name in required_names if name not in names]
    if missing_names:
        raise RuntimeError(f"[mtu3d.robot_config] Missing required cameras: {missing_names}. "
                          f"MTU3D requires cameras named: {required_names}")

    return required_names


def configure_robot_sensors(robot_cfg: RobotCfg) -> None:
    """Per-policy sensor configuration for MTU3D (branch by robot_cfg.type)."""
    robot_type = str(robot_cfg.type or "")
    if robot_type == "AliengoRobot":
        robot_cfg.sensors = _aliengo_sensors()
        return
    if robot_type == "CarterV1Robot":
        robot_cfg.sensors = _carter_v1_sensors()
        return
    if robot_type == "H1Robot":
        robot_cfg.sensors = _h1_sensors()
        return


def _aliengo_sensors() -> List[RepCameraCfg]:
    camera_0 = RepCameraCfg(
        name="camera_0",
        prim_path="trunk/Camera_0",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(0.350000, 0.000000, 0.000000),
        orientation=(0.454519, 0.541675, -0.541675, -0.454519),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_1 = RepCameraCfg(
        name="camera_1",
        prim_path="trunk/Camera_1",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(0.303109, 0.175000, 0.000000),
        orientation=(0.556670, 0.663414, -0.383022, -0.321394),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_2 = RepCameraCfg(
        name="camera_2",
        prim_path="trunk/Camera_2",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(0.175000, 0.303109, 0.000000),
        orientation=(0.620885, 0.739942, -0.198267, -0.166366),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_3 = RepCameraCfg(
        name="camera_3",
        prim_path="trunk/Camera_3",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(0.000000, 0.350000, 0.000000),
        orientation=(0.642788, 0.766044, -0.000000, 0.000000),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_4 = RepCameraCfg(
        name="camera_4",
        prim_path="trunk/Camera_4",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(-0.175000, 0.303109, 0.000000),
        orientation=(0.620885, 0.739942, 0.198267, 0.166366),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_5 = RepCameraCfg(
        name="camera_5",
        prim_path="trunk/Camera_5",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(-0.303109, 0.175000, 0.000000),
        orientation=(0.556670, 0.663414, 0.383022, 0.321394),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_6 = RepCameraCfg(
        name="camera_6",
        prim_path="trunk/Camera_6",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(-0.350000, 0.000000, 0.000000),
        orientation=(0.454519, 0.541675, 0.541675, 0.454519),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_7 = RepCameraCfg(
        name="camera_7",
        prim_path="trunk/Camera_7",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(-0.303109, -0.175000, 0.000000),
        orientation=(0.321394, 0.383022, 0.663414, 0.556670),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_8 = RepCameraCfg(
        name="camera_8",
        prim_path="trunk/Camera_8",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(-0.175000, -0.303109, 0.000000),
        orientation=(0.166366, 0.198267, 0.739942, 0.620885),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_9 = RepCameraCfg(
        name="camera_9",
        prim_path="trunk/Camera_9",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(-0.000000, -0.350000, 0.000000),
        orientation=(-0.000000, 0.000000, 0.766044, 0.642788),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_10 = RepCameraCfg(
        name="camera_10",
        prim_path="trunk/Camera_10",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(0.175000, -0.303109, 0.000000),
        orientation=(-0.166366, -0.198267, 0.739942, 0.620885),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_11 = RepCameraCfg(
        name="camera_11",
        prim_path="trunk/Camera_11",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(0.303109, -0.175000, 0.000000),
        orientation= (-0.321394, -0.383022, 0.663414, 0.556670),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    return [camera_0, camera_1, camera_2, camera_3, camera_4, camera_5,
            camera_6, camera_7, camera_8, camera_9, camera_10, camera_11]


def _carter_v1_sensors() -> List[RepCameraCfg]:
    camera_0 = RepCameraCfg(
        name="camera_0",
        prim_path="chassis_link/camera_mount/carter_camera_0",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(0.186047, 0.000000, 0.326918),
        orientation=(-0.500000, -0.500000, 0.500000, 0.500000),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_1 = RepCameraCfg(
        name="camera_1",
        prim_path="chassis_link/camera_mount/carter_camera_1",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(0.161122, 0.093024, 0.326918),
        orientation=(-0.612372, -0.612372, 0.353553, 0.353553),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_2 = RepCameraCfg(
        name="camera_2",
        prim_path="chassis_link/camera_mount/carter_camera_2",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(0.093024, 0.161122, 0.326918),
        orientation=(-0.683013, -0.683013, 0.183013, 0.183013),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_3 = RepCameraCfg(
        name="camera_3",
        prim_path="chassis_link/camera_mount/carter_camera_3",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(0.000000, 0.186047, 0.326918),
        orientation=(-0.707107, -0.707107, 0.000000, -0.000000),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_4 = RepCameraCfg(
        name="camera_4",
        prim_path="chassis_link/camera_mount/carter_camera_4",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(-0.093024, 0.161122, 0.326918),
        orientation=(-0.683013, -0.683013, -0.183013, -0.183013),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_5 = RepCameraCfg(
        name="camera_5",
        prim_path="chassis_link/camera_mount/carter_camera_5",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(-0.161122, 0.093024, 0.326918),
        orientation=(-0.612372, -0.612372, -0.353553, -0.353553),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_6 = RepCameraCfg(
        name="camera_6",
        prim_path="chassis_link/camera_mount/carter_camera_6",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(-0.186047, 0.000000, 0.326918),
        orientation=(-0.500000, -0.500000, -0.500000, -0.500000),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_7 = RepCameraCfg(
        name="camera_7",
        prim_path="chassis_link/camera_mount/carter_camera_7",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(-0.161122, -0.093024, 0.326918),
        orientation=(-0.353553, -0.353553, -0.612372, -0.612372),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_8 = RepCameraCfg(
        name="camera_8",
        prim_path="chassis_link/camera_mount/carter_camera_8",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(-0.093024, -0.161122, 0.326918),
        orientation=(-0.183013, -0.183013, -0.683013, -0.683013),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_9 = RepCameraCfg(
        name="camera_9",
        prim_path="chassis_link/camera_mount/carter_camera_9",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(-0.000000, -0.186047, 0.326918),
        orientation=(0.000000, -0.000000, -0.707107, -0.707107),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_10 = RepCameraCfg(
        name="camera_10",
        prim_path="chassis_link/camera_mount/carter_camera_10",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(0.093024, -0.161122, 0.326918),
        orientation=(0.183013, 0.183013, -0.683013, -0.683013),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_11 = RepCameraCfg(
        name="camera_11",
        prim_path="chassis_link/camera_mount/carter_camera_11",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(0.161122, -0.093024, 0.326918),
        orientation=(0.353553, 0.353553, -0.612372, -0.612372),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    return [camera_0, camera_1, camera_2, camera_3, camera_4, camera_5,
            camera_6, camera_7, camera_8, camera_9, camera_10, camera_11]


def _h1_sensors() -> List[RepCameraCfg]:
    camera_0 = RepCameraCfg(
        name="camera_0",
        prim_path="logo_link/Camera_0",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(0.300000, 0.000000, 0.390786),
        orientation=(0.500000, 0.500000, -0.500000, -0.500000),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_1 = RepCameraCfg(
        name="camera_1",
        prim_path="logo_link/Camera_1",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(0.259808, 0.150000, 0.390786),
        orientation=(0.612372, 0.612372, -0.353553, -0.353553),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_2 = RepCameraCfg(
        name="camera_2",
        prim_path="logo_link/Camera_2",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(0.150000, 0.259808, 0.390786),
        orientation=(0.683013, 0.683013, -0.183013, -0.183013),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_3 = RepCameraCfg(
        name="camera_3",
        prim_path="logo_link/Camera_3",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(0.000000, 0.300000, 0.390786),
        orientation=(0.707107, 0.707107, -0.000000, -0.000000),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_4 = RepCameraCfg(
        name="camera_4",
        prim_path="logo_link/Camera_4",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(-0.150000, 0.259808, 0.390786),
        orientation=(0.683013, 0.683013, 0.183013, 0.183013),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_5 = RepCameraCfg(
        name="camera_5",
        prim_path="logo_link/Camera_5",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(-0.259808, 0.150000, 0.390786),
        orientation=(0.612372, 0.612372, 0.353553, 0.353553),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_6 = RepCameraCfg(
        name="camera_6",
        prim_path="logo_link/Camera_6",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(-0.300000, 0.000000, 0.390786),
        orientation=(0.500000, 0.500000, 0.500000, 0.500000),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_7 = RepCameraCfg(
        name="camera_7",
        prim_path="logo_link/Camera_7",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(-0.259808, -0.150000, 0.390786),
        orientation=(0.353553, 0.353553, 0.612372, 0.612372),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_8 = RepCameraCfg(
        name="camera_8",
        prim_path="logo_link/Camera_8",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(-0.150000, -0.259808, 0.390786),
        orientation=(0.183013, 0.183013, 0.683013, 0.683013),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_9 = RepCameraCfg(
        name="camera_9",
        prim_path="logo_link/Camera_9",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(-0.000000, -0.300000, 0.390786),
        orientation=(0.000000, 0.000000, 0.707107, 0.707107),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_10 = RepCameraCfg(
        name="camera_10",
        prim_path="logo_link/Camera_10",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(0.150000, -0.259808, 0.390786),
        orientation=(-0.183013, -0.183013, 0.683013, 0.683013),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    camera_11 = RepCameraCfg(
        name="camera_11",
        prim_path="logo_link/Camera_11",
        resolution=(640, 360),
        depth=True,
        camera_params=True,
        translation=(0.259808, -0.150000, 0.390786),
        orientation=(-0.353553, -0.353553, 0.612372, 0.612372),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60.0,
    )
    return [camera_0, camera_1, camera_2, camera_3, camera_4, camera_5,
            camera_6, camera_7, camera_8, camera_9, camera_10, camera_11]
