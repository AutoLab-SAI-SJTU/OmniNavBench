"""Robot configuration for Uni-NaVid policy.

This module configures camera sensors and execution profiles for different
robot types to match the original VLN-CE configuration.

Camera Configuration (from vlnce_task_uninavid_r2r.yaml):
- Resolution: 640x480
- HFOV: 120 degrees

Execution Profile (from VLN-CE):
- Forward step: 0.25m (FORWARD_STEP_SIZE)
- Turn angle: 15 degrees (TURN_ANGLE / 2)
"""

from __future__ import annotations

from typing import List

from bench.configs.execution import RobotExecutionProfile
from OmniNav.core.config import RobotCfg
from OmniNavExt.configs.sensors import RepCameraCfg


# Global variable to store current robot type for execution profile override
_current_robot_type: str = ""
_current_robot_scale: float = 1.0


def get_required_cameras(robot_cfg: RobotCfg) -> List[str]:
    """Return list of camera names required by this policy.

    The camera names are derived from robot_cfg.sensors to avoid
    maintaining duplicate configuration.

    Args:
        robot_cfg: Robot configuration

    Returns:
        List of camera names that EpisodeRunner should collect
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
    """Configure camera sensors for Uni-NaVid policy.

    Configures cameras based on robot type to match original VLN-CE settings:
    - Resolution: 640x480
    - HFOV: 120 degrees

    Args:
        robot_cfg: Robot configuration to modify
    """
    global _current_robot_type, _current_robot_scale
    robot_type = str(robot_cfg.type or "")
    _current_robot_type = robot_type
    scale = getattr(robot_cfg, "scale", None)
    scale_factor = 1.0
    if isinstance(scale, (list, tuple)) and len(scale) >= 1:
        try:
            scale_factor = float(scale[0])
        except (TypeError, ValueError):
            scale_factor = 1.0
    _current_robot_scale = scale_factor
    print(f"[uninavid.robot_config] Configuring sensors for robot_type='{robot_type}'")

    if robot_type == "AliengoRobot":
        robot_cfg.sensors = _aliengo_sensors()
        print(f"[uninavid.robot_config] Configured AliengoRobot sensors: {[s.name for s in robot_cfg.sensors]}")
        return
    if robot_type == "CarterV1Robot":
        robot_cfg.sensors = _carter_v1_sensors()
        print(f"[uninavid.robot_config] Configured CarterV1Robot sensors: {[s.name for s in robot_cfg.sensors]}")
        return
    if robot_type == "H1Robot":
        robot_cfg.sensors = _h1_sensors()
        print(f"[uninavid.robot_config] Configured H1Robot sensors: {[s.name for s in robot_cfg.sensors]}")
        return

    # Unknown robot type
    raise RuntimeError(f"[uninavid.robot_config] Unknown robot_type='{robot_type}'")


def get_execution_profile_override(profile: RobotExecutionProfile) -> RobotExecutionProfile:
    """Return Uni-NaVid specific execution parameters.

    Overrides default execution profile to match VLN-CE settings:
    - step_lin_dist: 0.25m (VLN-CE FORWARD_STEP_SIZE)
    - step_ang_deg: 15.0 degrees (VLN-CE TURN_ANGLE / 2)

    Args:
        profile: Default execution profile

    Returns:
        Modified execution profile for Uni-NaVid
    """
    global _current_robot_type, _current_robot_scale

    linear_scale = max(float(_current_robot_scale), 1e-6)
    if linear_scale < 1.0:
        linear_scale *= 0.5

    # Keep the discrete turn angle definition, but scale linear motion-related
    # thresholds and speeds with the robot size so shrunken Carter variants do
    # not appear unnaturally fast.
    max_lin_vel = 0.25 * linear_scale
    step_lin_dist = 0.5 * linear_scale
    finish_pos_eps = 0.05 * linear_scale

    if _current_robot_type == "CarterV1Robot":
        # CarterV1 specific parameters
        return RobotExecutionProfile(
            max_lin_vel=max_lin_vel,
            max_ang_vel=1.0,
            step_lin_dist=step_lin_dist,  # Scale linear step with robot size.
            step_ang_deg=15.0,   # VLN-CE: TURN_ANGLE=30, each turn is 15 degrees
            finish_pos_eps=finish_pos_eps,
            finish_rot_eps_deg=3.0,
        )
    else:
        # AliengoRobot and H1Robot (default)
        return RobotExecutionProfile(
            max_lin_vel=max_lin_vel,
            max_ang_vel=2.0,
            step_lin_dist=step_lin_dist,  # Scale linear step with robot size.
            step_ang_deg=15.0,   # VLN-CE: TURN_ANGLE=30, each turn is 15 degrees
            finish_pos_eps=finish_pos_eps,
            finish_rot_eps_deg=3.0,
        )


def _aliengo_sensors() -> List[RepCameraCfg]:
    """Camera configuration for Aliengo quadruped robot."""
    camera_cfg = RepCameraCfg(
        name="camera",
        prim_path="trunk/Camera",
        resolution=(640, 480),  # VLN-CE: 640x480
        depth=True,
        camera_params=True,
        translation=None,
        orientation=None,
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=120.0,  # VLN-CE: HFOV=120
    )
    rgb_cfg = RepCameraCfg(
        name="rgb",
        prim_path="trunk/rgb",
        resolution=(640, 480),  # VLN-CE: 640x480
        depth=False,
        camera_params=True,
        translation=(-0.6,0,0.3),
        orientation=(0.454519, 0.541675, -0.541675, -0.454519),
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=120.0,  # VLN-CE: HFOV=120
    )
    return [camera_cfg, rgb_cfg]


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
    """Camera configuration for H1 humanoid robot."""
    camera_cfg = RepCameraCfg(
        name='camera',
        prim_path='logo_link/Camera',
        resolution=(640, 480),  # VLN-CE: 640x480
        depth=True,
        camera_params=True,
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=120.0,  # VLN-CE: HFOV=120
    )
    return [camera_cfg]
