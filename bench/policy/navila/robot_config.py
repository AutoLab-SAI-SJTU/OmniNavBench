from __future__ import annotations

from typing import List

from bench.configs.execution import RobotExecutionProfile
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
            raise ValueError("[navila.robot_config] Camera SensorCfg.name must be a non-empty string")
        name = name.strip()
        if name not in names:
            names.append(name)
    if not names:
        raise RuntimeError(f"[navila.robot_config] No RepCamera sensors configured for robot={robot_cfg.type}")
    return names


def configure_robot_sensors(robot_cfg: RobotCfg) -> None:
    """Per-policy sensor configuration for NaVILA (branch by robot_cfg.type)."""
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
    raise RuntimeError(f"[navila.robot_config] Unknown robot_type='{robot_type}', cannot configure sensors.")


def get_execution_profile_override(profile: RobotExecutionProfile) -> RobotExecutionProfile:
    """
    Align STEP_ACTION semantics with NaVILA evaluation defaults:
    - FORWARD_STEP_SIZE=0.25m
    - TURN_ANGLE=15deg

    Source: NaVILA/evaluation/habitat_extensions/config/vlnce_task.yaml
    """
    return RobotExecutionProfile(
        max_lin_vel=profile.max_lin_vel,
        max_ang_vel=profile.max_ang_vel,
        step_lin_dist=0.25,
        step_ang_deg=15.0,
        finish_pos_eps=profile.finish_pos_eps,
        finish_rot_eps_deg=profile.finish_rot_eps_deg,
    )


def _aliengo_sensors() -> List[RepCameraCfg]:
    camera_cfg = RepCameraCfg(
        name="camera",
        prim_path="trunk/Camera",
        resolution=(512, 512),
        depth=True,
        camera_params=True,
        translation=None,
        orientation=None,
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=90.0,
    )
    return [camera_cfg]


def _carter_v1_sensors() -> List[RepCameraCfg]:
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


def _h1_sensors() -> List[RepCameraCfg]:
    camera_cfg = RepCameraCfg(
        name="camera",
        prim_path="logo_link/Camera",
        resolution=(512, 512),
        depth=True,
        camera_params=True,
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=90.0,
    )
    return [camera_cfg]
