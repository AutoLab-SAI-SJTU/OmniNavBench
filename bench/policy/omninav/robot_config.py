from __future__ import annotations

from typing import List

from OmniNav.core.config import RobotCfg
from OmniNavExt.configs.sensors import RepCameraCfg


def get_required_cameras(robot_cfg: RobotCfg) -> List[str]:
    """Required camera names for OmniNav: ``left``, ``front``, ``right`` (panoramic view)."""
    sensors = list(robot_cfg.sensors or [])
    names: List[str] = []
    for s in sensors:
        if getattr(s, "type", None) != "RepCamera":
            continue
        name = getattr(s, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("[omninav.robot_config] Camera SensorCfg.name must be a non-empty string")
        name = name.strip()
        if name not in names:
            names.append(name)

    required_names = ["left", "front", "right"]
    missing_names = [name for name in required_names if name not in names]
    if missing_names:
        raise RuntimeError(f"[omninav.robot_config] Missing required cameras: {missing_names}. "
                          f"OmniNav requires cameras named: {required_names}")

    return required_names


def configure_robot_sensors(robot_cfg: RobotCfg) -> None:
    """Per-policy sensor configuration for OmniNav (branch by robot_cfg.type).
    
    This function is called by BenchRunner._apply_policy_robot_config() before
    sensor creation, allowing OmniNav to configure the three cameras it needs.
    """
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
    
    # Unknown robot type - warn and use generic fallback
    raise RuntimeError(f"[omninav.robot_config] WARNING: Unknown robot_type='{robot_type}', "
          f"using generic three-camera configuration")


def _aliengo_sensors() -> List[RepCameraCfg]:
    """Three-camera OmniNav setup for the Aliengo robot."""
    front_camera = RepCameraCfg(
        name="front",
        prim_path="trunk/Camera_Front",
        resolution=(720, 640),  # OmniNav input size.
        depth=False,  # OmniNav uses RGB only.
        camera_params=True,
        translation=(0.35, 0.0, 0.0),
        orientation=(0.454519, 0.541675, -0.541675, -0.454519),  # Forward, with a 15-degree pitch up.
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=110.0,  # OmniNav default FOV.
    )
    left_camera = RepCameraCfg(
        name="left",
        prim_path="trunk/Camera_Left",
        resolution=(720, 640),
        depth=False,
        camera_params=True,
        translation=(0.35, 0.30, 0),
        orientation=(0.642788, 0.766044, 0, 0),  # +90 deg about Z (w, x, y, z).
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=110.0,
    )
    right_camera = RepCameraCfg(
        name="right",
        prim_path="trunk/Camera_Right",
        resolution=(720, 640),
        depth=False,
        camera_params=True,
        translation=(0.35, -0.30, 0),
        orientation=(-0.0, 0.0, 0.766044, 0.642788),  # -90 deg about Z (w, x, y, z).
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=110.0,
    )
    return [front_camera, left_camera, right_camera]


def _carter_v1_sensors() -> List[RepCameraCfg]:
    """Three-camera OmniNav setup for the Carter V1 robot."""
    front_camera = RepCameraCfg(
        name="front",
        prim_path="chassis_link/camera_mount/Camera_Front",
        resolution=(720, 640),
        depth=False,
        camera_params=True,
        translation=(0.089172, 0.0, 0.326497),
        orientation=(-0.5, -0.5, 0.5, 0.5),  # Forward, with a 15-degree pitch up.
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=110.0,
    )
    left_camera = RepCameraCfg(
        name="left",
        prim_path="chassis_link/camera_mount/Camera_Left",
        resolution=(720, 640),
        depth=False,
        camera_params=True,
        translation=(0.103, 0.30, 0.309),
        orientation=(0.707, 0.707, 0.0, 0.0),  # +90 deg about Z (w, x, y, z).
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=110.0,
    )
    right_camera = RepCameraCfg(
        name="right",
        prim_path="chassis_link/camera_mount/Camera_Right",
        resolution=(720, 640),
        depth=False,
        camera_params=True,
        translation=(0.103, -0.3, 0.309),
        orientation=(0.0, 0.0, 0.707, 0.707),  # -90 deg about Z (w, x, y, z).
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=110.0,
    )
    return [front_camera, left_camera, right_camera]


def _h1_sensors() -> List[RepCameraCfg]:
    """Three-camera OmniNav setup for the H1 robot."""
    front_camera = RepCameraCfg(
        name="front",
        prim_path="logo_link/Camera_Front",
        resolution=(720, 640),
        depth=False,
        camera_params=True,
        translation=(0.083626, 0.0, 0.390786),
        orientation=(0.5, 0.5, -0.5, -0.5),  # Forward, with a 15-degree pitch up.
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=110.0,
    )
    left_camera = RepCameraCfg(
        name="left",
        prim_path="logo_link/Camera_Left",
        resolution=(720, 640),
        depth=False,
        camera_params=True,
        translation=(0.083630, 0.300000, 0.390790),
        orientation=(0.707107, 0.707107, 0.000000, 0.000000),  # +90 deg about Z (w, x, y, z).
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=110.0,
    )
    right_camera = RepCameraCfg(
        name="right",
        prim_path="logo_link/Camera_Right",
        resolution=(720, 640),
        depth=False,
        camera_params=True,
        translation=(0.083630, -0.300000, 0.390790),
        orientation=(-0.000000, 0.000000, 0.707107, 0.707107),  # -90 deg about Z (w, x, y, z).
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=110.0,
    )
    return [front_camera, left_camera, right_camera]
