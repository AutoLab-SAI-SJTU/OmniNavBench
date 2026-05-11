from __future__ import annotations

import carb

_DEFAULT_WORLD_KWARGS = {
    "physics_dt": 1.0 / 30.0,
    "rendering_dt": 1.0 / 30.0,
    "stage_units_in_meters": 1.0,  # Default; overridden at runtime from the scene.
}


def _detect_stage_units_in_meters() -> float:
    """Detect the unit scale of the current USD stage.

    Returns:
        float: metersPerUnit value (defaults to 1.0 m).
    """
    try:
        import omni.usd
        from pxr import UsdGeom

        ctx = omni.usd.get_context()
        if not ctx:
            carb.log_warn("[World] USD context not available, using default stage units (1.0)")
            return 1.0

        stage = ctx.get_stage()
        if not stage:
            carb.log_warn("[World] USD stage not available, using default stage units (1.0)")
            return 1.0

        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))

        if meters_per_unit <= 0:
            carb.log_warn(f"[World] Invalid stage units ({meters_per_unit}), using default (1.0)")
            return 1.0

        # Log a hint when the stage uses non-meter units.
        if abs(meters_per_unit - 1.0) > 1e-3:
            unit_name = "centimeters" if abs(meters_per_unit - 0.01) < 1e-3 else f"{meters_per_unit}m"
            carb.log_info(f"[World] Detected stage units: {meters_per_unit} meters per unit ({unit_name})")

        return meters_per_unit

    except Exception as exc:
        carb.log_warn(f"[World] Failed to detect stage units: {exc}, using default (1.0)")
        return 1.0


def bootstrap_world_if_needed(**overrides):
    """Create the Isaac World singleton if it doesn't exist yet.

    Auto-detects the scene's metersPerUnit and configures the physics engine accordingly.
    This is critical for scenes that use non-meter units, such as GRScenes (centimeters).
    """

    try:
        from omni.isaac.core import World
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Isaac Core World API is unavailable in the current environment.") from exc

    try:
        world = World.instance()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to query Isaac World instance: {exc}") from exc

    if world is not None:
        return world

    world_kwargs = dict(_DEFAULT_WORLD_KWARGS)

    # If the caller did not provide stage_units_in_meters explicitly, auto-detect from the stage.
    if "stage_units_in_meters" not in overrides or overrides.get("stage_units_in_meters") is None:
        detected_units = _detect_stage_units_in_meters()
        world_kwargs["stage_units_in_meters"] = detected_units
    else:
        world_kwargs["stage_units_in_meters"] = overrides["stage_units_in_meters"]

    # Apply any other caller-supplied overrides.
    world_kwargs.update({k: v for k, v in overrides.items() if v is not None})

    try:
        world = World(**world_kwargs)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Unable to create Isaac World with arguments {world_kwargs}: {exc}") from exc

    carb.log_info(
        "[World] Created Isaac World (physics_dt=%s, rendering_dt=%s, stage_units_in_meters=%s)."
        % (
            world_kwargs.get("physics_dt"),
            world_kwargs.get("rendering_dt"),
            world_kwargs.get("stage_units_in_meters"),
        )
    )
    return world


def ensure_world(*_args, **_kwargs):
    """Return a valid omni.isaac.core.World, creating it if necessary."""

    return bootstrap_world_if_needed(**_kwargs)
