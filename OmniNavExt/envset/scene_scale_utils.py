"""Scene unit-scaling helpers."""
from __future__ import annotations

import logging
from typing import Optional

from pxr import Usd, UsdGeom, Sdf

log = logging.getLogger(__name__)


def _should_skip_scaling(
    scene_units_in_meters: Optional[float],
    stage_meters_per_unit: float,
    epsilon: float = 1e-6
) -> bool:
    """Return True when no scaling is required."""
    if scene_units_in_meters is None:
        return True
    return abs(scene_units_in_meters - stage_meters_per_unit) < epsilon


def _validate_scale(scale: float, scene_prim_path: str) -> None:
    """Validate that the computed scale factor is in a sane range."""
    if scale < 1e-4 or scale > 100.0:
        raise RuntimeError(
            f"Invalid scale {scale} for {scene_prim_path}. "
            f"Scale must be in range [1e-4, 100]"
        )


def _has_existing_scale(prim: Usd.Prim, scene_units_in_meters: float) -> bool:
    """Return True if the scale has already been applied to this prim."""
    if not prim.IsA(UsdGeom.Xformable):
        return False

    xform = UsdGeom.Xformable(prim)
    scale_ops = [op for op in xform.GetOrderedXformOps()
                 if op.GetOpType() == UsdGeom.XformOp.TypeScale]

    if not scale_ops:
        return False

    # Check the marker attribute.
    marker = prim.GetAttribute("uniscene:scaledFromUnits")
    if marker and abs(marker.Get() - scene_units_in_meters) < 1e-6:
        return True

    # An identity (1, 1, 1) scale is the default and may be safely overwritten.
    scale_value = scale_ops[0].Get()
    is_identity = all(abs(v - 1.0) < 1e-6 for v in scale_value)
    if is_identity:
        return False

    # Non-default scale without our marker likely means a previous double-scale.
    raise RuntimeError(
        f"{prim.GetPath()} already has non-default scale={scale_value}, "
        f"but no uniscene:scaledFromUnits marker. Refusing to double-scale."
    )


def _apply_scale_transform(prim: Usd.Prim, scale: float, scene_units_in_meters: float) -> None:
    """Apply the scale transform and write the tracking marker."""
    if not prim.IsA(UsdGeom.Xformable):
        raise RuntimeError(f"{prim.GetPath()} is not Xformable")

    xform = UsdGeom.Xformable(prim)

    existing_scale_ops = [op for op in xform.GetOrderedXformOps()
                         if op.GetOpType() == UsdGeom.XformOp.TypeScale]

    if existing_scale_ops:
        # Reuse the existing scale op even when it is the identity.
        scale_op = existing_scale_ops[0]
        scale_op.Set((scale, scale, scale))
    else:
        scale_op = xform.AddScaleOp()
        scale_op.Set((scale, scale, scale))

    marker = prim.CreateAttribute("uniscene:scaledFromUnits", Sdf.ValueTypeNames.Double)
    marker.Set(scene_units_in_meters)


def apply_scene_unit_scale(
    stage: Usd.Stage,
    scene_prim_path: str,
    scene_units_in_meters: Optional[float],
    stage_meters_per_unit: float = 1.0,
) -> None:
    """Apply a unit scale to the scene prim, converting its internal units into stage units.

    Args:
        stage: USD Stage.
        scene_prim_path: Scene prim path (e.g. /World/env_0/scene).
        scene_units_in_meters: 1 unit = X meters, or ``None`` when no scaling is required.
        stage_meters_per_unit: Stage unit size (defaults to 1.0 m).

    Raises:
        RuntimeError: If the scale is out of range or has already been applied.
    """
    if _should_skip_scaling(scene_units_in_meters, stage_meters_per_unit):
        log.info(f"[SceneScale] {scene_prim_path}: NO scaling needed (units_in_meters={scene_units_in_meters})")
        return

    scale = scene_units_in_meters / stage_meters_per_unit

    log.info("=" * 70)
    log.info(f"[SceneScale] Processing: {scene_prim_path}")
    log.info(f"[SceneScale] BEFORE:")
    log.info(f"[SceneScale]   - Scene units: 1 unit = {scene_units_in_meters} meters")
    log.info(f"[SceneScale]   - Stage units: 1 unit = {stage_meters_per_unit} meters")
    log.info(f"[SceneScale]   - Calculated scale factor: {scale}")

    _validate_scale(scale, scene_prim_path)

    prim = stage.GetPrimAtPath(scene_prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Scene prim {scene_prim_path} not found")

    if _has_existing_scale(prim, scene_units_in_meters):
        log.info(f"[SceneScale] {scene_prim_path}: already scaled, skip")
        log.info("=" * 70)
        return

    _apply_scale_transform(prim, scale, scene_units_in_meters)

    log.info(f"[SceneScale] AFTER:")
    log.info(f"[SceneScale]   - Applied scale transform: ({scale}, {scale}, {scale})")
    log.info(f"[SceneScale]   - Marker set: uniscene:scaledFromUnits = {scene_units_in_meters}")
    log.info(f"[SceneScale] ✓ Scale successfully applied to {scene_prim_path}")
    log.info("=" * 70)
