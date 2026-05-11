from typing import Optional, Tuple

from OmniNav.core.config.robot import SensorCfg


class RepCameraCfg(SensorCfg):
    # Fields from params.
    type: Optional[str] = 'RepCamera'
    resolution: Optional[Tuple[int, int]] = None  # Camera only
    rgba: Optional[bool] = True
    landmarks: Optional[bool] = False
    depth: Optional[bool] = False
    pointcloud: Optional[bool] = False
    camera_params: Optional[bool] = False
    # Camera settings (meters, converted to stage units at runtime)
    clipping_range_m: Optional[Tuple[float, float]] = None  # (near_m, far_m)
    # Local pose relative to robot or mount
    translation: Optional[Tuple[float, float, float]] = None
    orientation: Optional[Tuple[float, float, float, float]] = None  # Quaternion (w, x, y, z)

    # Focal length for camera field of view control (mm), smaller values mean wider field of view
    focal_length: Optional[float] = None
    # Field of view in degrees, alternative to focal_length for direct FOV control
    fov_degrees: Optional[float] = None


class MocapControlledCameraCfg(SensorCfg):
    # Fields from params.
    type: Optional[str] = 'MocapControlledCamera'
    resolution: Optional[Tuple[int, int]] = None  # Camera only
    translation: Optional[Tuple[float, float, float]] = None
    orientation: Optional[Tuple[float, float, float, float]] = None  # Quaternion in local frame


class LayoutEditMocapControlledCameraCfg(SensorCfg):
    type: Optional[str] = 'LayoutEditMocapControlledCamera'
    resolution: Optional[Tuple[int, int]] = None  # Camera only
    translation: Optional[Tuple[float, float, float]] = None
    orientation: Optional[Tuple[float, float, float, float]] = None  # Quaternion in local frame
