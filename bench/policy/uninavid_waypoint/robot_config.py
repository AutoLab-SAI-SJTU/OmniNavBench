"""Robot configuration for Uni-NaVid Waypoint policy.

This module configures camera sensors and execution profiles for different
robot types. The waypoint model uses the same visual input as the original
Uni-NaVid (single camera, 640x480, HFOV 120).

Camera Configuration:
- Resolution: 640x480
- HFOV: 120 degrees

Execution Profile:
- Uses go_toward_point controller for waypoint following
- No discrete step parameters needed (continuous waypoint execution)
"""

from __future__ import annotations

from typing import List, Optional

from bench.configs.execution import RobotExecutionProfile
from OmniNav.core.config import RobotCfg
from OmniNavExt.configs.sensors import RepCameraCfg


# Global variable to store current robot type for execution profile override
_current_robot_type: str = ""


def get_required_cameras(robot_cfg: RobotCfg) -> List[str]:
    """Return list of camera names required by this policy.

    Uni-NaVid Waypoint uses a single front-facing camera.

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
            raise ValueError("[uninavid_waypoint.robot_config] Camera SensorCfg.name must be a non-empty string")
        name = name.strip()
        if name not in names:
            names.append(name)
    if not names:
        raise RuntimeError(f"[uninavid_waypoint.robot_config] No RepCamera sensors configured for robot={robot_cfg.type}")
    return names


def configure_robot_sensors(robot_cfg: RobotCfg) -> None:
    """Configure camera sensors for Uni-NaVid Waypoint policy.

    Configures cameras based on robot type to match original VLN-CE settings:
    - Resolution: 640x480
    - HFOV: 120 degrees

    Args:
        robot_cfg: Robot configuration to modify
    """
    global _current_robot_type
    robot_type = str(robot_cfg.type or "")
    _current_robot_type = robot_type
    print(f"[uninavid_waypoint.robot_config] Configuring sensors for robot_type='{robot_type}'")

    if robot_type == "AliengoRobot":
        robot_cfg.sensors = _aliengo_sensors()
        print(f"[uninavid_waypoint.robot_config] Configured AliengoRobot sensors: {[s.name for s in robot_cfg.sensors]}")
        return
    if robot_type == "CarterV1Robot":
        robot_cfg.sensors = _carter_v1_sensors()
        print(f"[uninavid_waypoint.robot_config] Configured CarterV1Robot sensors: {[s.name for s in robot_cfg.sensors]}")
        return
    if robot_type == "H1Robot":
        robot_cfg.sensors = _h1_sensors()
        print(f"[uninavid_waypoint.robot_config] Configured H1Robot sensors: {[s.name for s in robot_cfg.sensors]}")
        return

    # Unknown robot type
    raise RuntimeError(f"[uninavid_waypoint.robot_config] Unknown robot_type='{robot_type}'")


def get_execution_profile_override(profile: RobotExecutionProfile) -> RobotExecutionProfile:
    """Return Uni-NaVid Waypoint specific execution parameters.

    For waypoint-based navigation, we use go_toward_point controller
    which doesn't rely on discrete step parameters. However, we still
    provide reasonable defaults for any fallback scenarios.

    Args:
        profile: Default execution profile

    Returns:
        Modified execution profile for Uni-NaVid Waypoint
    """
    global _current_robot_type

    # Waypoint policy uses controller-based local goal following, so these parameters
    # are mainly for fallback or compatibility
    return RobotExecutionProfile(
        max_lin_vel=0.25,
        max_ang_vel=0.5,
        step_lin_dist=0.5,
        step_ang_deg=15.0,
        finish_pos_eps=0.1,  # Waypoint arrival threshold
        finish_rot_eps_deg=5.0,
    )


def _aliengo_sensors() -> List[RepCameraCfg]:
    """Camera configuration for Aliengo quadruped robot."""
    camera_cfg = RepCameraCfg(
        name="camera",
        prim_path="trunk/Camera",
        resolution=(640, 480),  # VLN-CE: 640x480
        depth=False,
        camera_params=True,
        translation=None,
        orientation=None,
        hfov_deg=120.0,  # VLN-CE: 120 degrees
        clipping_range_m=(0.01, 1000.0)
    )
    return [camera_cfg]


def _carter_v1_sensors() -> List[RepCameraCfg]:
    """Camera configuration for CarterV1 wheeled robot."""
    camera_cfg = RepCameraCfg(
        name="camera",
        prim_path="chassis_link/camera_mount/carter_camera_first_person",
        depth=True,
        camera_params=True,
        translation=(0.0,0.0,0.6),
        orientation=None,
        resolution=(640, 480),  # VLN-CE: 640x480
        clipping_range_m=(0.01, 1000.0),
        fov_degrees=60,  # VLN-CE: HFOV=120
    )
    return [camera_cfg]

def _h1_sensors() -> List[RepCameraCfg]:
    """Camera configuration for H1 humanoid robot."""
    camera_cfg = RepCameraCfg(
        name='camera',
        prim_path='logo_link/Camera',
        resolution=(640, 480),  # VLN-CE: 640x480
        depth=False, 
        camera_params=True,
        translation=None,
        orientation=None,
        hfov_deg=120.0,  # VLN-CE: 120 degrees
        clipping_range_m=(0.01, 1000.0)
    )
    return [camera_cfg]
