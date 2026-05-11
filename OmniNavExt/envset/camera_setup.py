"""Helpers to create a robot-mounted camera prim with a fixed local pose."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

def _to_vec3(translation: Sequence[float]):
    from pxr import Gf

    return Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2]))


def _to_quat(orientation: Sequence[float]):
    from pxr import Gf

    return Gf.Quatd(float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3]))


def ensure_robot_camera_prim(
    robot_prim_path: str,
    camera_rel_path: str,
    translation_m: Optional[Sequence[float]] = None,
    orientation_wxyz: Optional[Sequence[float]] = None,
) -> str:
    """
    Ensure a Camera prim exists under the given robot with the desired local pose.

    Args:
        robot_prim_path: Absolute prim path of the robot root.
        camera_rel_path: Relative path from robot root to camera (e.g., 'camera_mount/rgb').
        translation_m: Optional local translation in meters.
        orientation_wxyz: Optional local orientation quaternion (w, x, y, z).

    Returns:
        The absolute camera prim path.
    """
    if not robot_prim_path or not camera_rel_path:
        raise ValueError("robot_prim_path and camera_rel_path are required")

    # Lazily import omni/usd deps after SimulationApp is ready
    from OmniNavExt.envset.stage_util import UnitScaleService
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available when creating camera prim")

    camera_path = Sdf.Path(robot_prim_path).AppendPath(camera_rel_path)
    mount_path = camera_path.GetParentPath()

    # Create mount as Xform if missing
    mount_prim = stage.GetPrimAtPath(mount_path)
    if not mount_prim or not mount_prim.IsValid():
        mount_prim = UsdGeom.Xform.Define(stage, mount_path).GetPrim()

    # Create camera prim if missing
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim or not camera_prim.IsValid():
        camera_prim = UsdGeom.Camera.Define(stage, camera_path).GetPrim()

    # Apply local transform if provided
    xformable = UsdGeom.Xformable(camera_prim)
    meters_to_stage = UnitScaleService.get_stage_units_per_meter()

    if translation_m is not None:
        local_t = _to_vec3(translation_m) * meters_to_stage
        _set_or_update_op(xformable, UsdGeom.XformOp.TypeTranslate, local_t)

    if orientation_wxyz is not None:
        local_q = _to_quat(orientation_wxyz)
        _set_or_update_op(xformable, UsdGeom.XformOp.TypeOrient, local_q)

    return str(camera_path)


def _set_or_update_op(xformable: UsdGeom.Xformable, op_type: UsdGeom.XformOp.Type, value):
    """Set an existing op of given type or create one."""
    existing = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == op_type]
    if existing:
        op = existing[0]
    else:
        op = xformable.AddXformOp(op_type)
    op.Set(value)
