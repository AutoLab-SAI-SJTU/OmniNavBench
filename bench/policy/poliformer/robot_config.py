from __future__ import annotations

from typing import List

from OmniNav.core.config import RobotCfg
from OmniNavExt.configs.sensors import RepCameraCfg


def get_required_cameras(robot_cfg: RobotCfg) -> List[str]:
    """Required camera names for this policy, auto-derived from ``robot_cfg.sensors``.

    Only sensors with ``type == 'RepCamera'`` are treated as cameras.
    """
    sensors = list(robot_cfg.sensors or [])
    names: List[str] = []
    for s in sensors:
        if getattr(s, "type", None) != "RepCamera":
            continue
        name = getattr(s, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("[uninavid.robot_config] Camera SensorCfg.name must be a non-empty string")
        name = name.strip()
        if name not in names:
            names.append(name)
    if not names:
        raise RuntimeError(f"[uninavid.robot_config] No RepCamera sensors configured for robot={robot_cfg.type}")
    return names


def configure_robot_sensors(robot_cfg: RobotCfg) -> None:
    """Per-policy sensor configuration for UniNaVid (branch by robot_cfg.type)."""
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
    camera_cfg = RepCameraCfg(
        name="camera",
        prim_path="trunk/Camera",
        resolution=(396, 224),
        depth=True,
        camera_params=True,
        translation=None,
        orientation=None,
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=59.0,  # FOV required by the UniNaVid model.
    )
    return [camera_cfg]


def _carter_v1_sensors() -> List[RepCameraCfg]:
    camera_cfg = RepCameraCfg(
        name="camera",
        prim_path="chassis_link/camera_mount/carter_camera_first_person",
        depth=True,
        camera_params=True,
        resolution=(396, 224),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=59.0,  # FOV required by the UniNaVid model.
    )
    return [camera_cfg]


def _h1_sensors() -> List[RepCameraCfg]:
    camera_cfg = RepCameraCfg(
        name="camera",
        prim_path="logo_link/Camera",
        resolution=(396, 224),
        depth=True,
        camera_params=True,
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=59.0,  # FOV required by the UniNaVid model.
    )
    return [camera_cfg]
