"""NavMesh parameter unit conversion."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class NavmeshConfigRaw:
    """Raw NavMesh config in scene units."""
    bake_root_prim_path: str
    include_volume_parent: Optional[str]
    z_padding: float
    agent_radius: float
    min_include_xy: Optional[float]
    min_include_z: Optional[float]


@dataclass
class NavmeshConfigWorld:
    """NavMesh config in world coordinates (meters)."""
    bake_root_prim_path: str
    include_volume_parent: Optional[str]
    z_padding_m: float
    agent_radius_m: float
    min_include_xy_m: Optional[float]
    min_include_z_m: Optional[float]


def _calculate_scale_factor(
    scene_units_in_meters: Optional[float],
    stage_meters_per_unit: float,
    epsilon: float = 1e-6
) -> float:
    """Compute the unit-conversion scale factor."""
    if scene_units_in_meters is None:
        return 1.0
    if abs(scene_units_in_meters - stage_meters_per_unit) < epsilon:
        return 1.0
    return scene_units_in_meters / stage_meters_per_unit


def to_world_navmesh_config(
    raw_cfg: NavmeshConfigRaw,
    scene_units_in_meters: Optional[float],
    stage_meters_per_unit: float = 1.0,
) -> NavmeshConfigWorld:
    """Convert a NavMesh config from scene units to world meters.

    ``scene_units_in_meters`` is 1 unit = X meters; ``None`` means the raw config
    is already in meters. ``stage_meters_per_unit`` defaults to 1.0 m.
    """
    scale = _calculate_scale_factor(scene_units_in_meters, stage_meters_per_unit)

    log.info("=" * 70)
    log.info("[NavMeshUnits] Converting NavMesh parameters")
    log.info(f"[NavMeshUnits] Scene units: {scene_units_in_meters} meters/unit")
    log.info(f"[NavMeshUnits] Scale factor: {scale}")
    log.info(f"[NavMeshUnits] BEFORE (scene units):")
    log.info(f"[NavMeshUnits]   - agent_radius: {raw_cfg.agent_radius}")
    log.info(f"[NavMeshUnits]   - z_padding: {raw_cfg.z_padding}")
    log.info(f"[NavMeshUnits]   - min_include_xy: {raw_cfg.min_include_xy}")
    log.info(f"[NavMeshUnits]   - min_include_z: {raw_cfg.min_include_z}")

    world_config = NavmeshConfigWorld(
        bake_root_prim_path=raw_cfg.bake_root_prim_path,
        include_volume_parent=raw_cfg.include_volume_parent,
        z_padding_m=raw_cfg.z_padding * scale,
        agent_radius_m=raw_cfg.agent_radius * scale,
        min_include_xy_m=raw_cfg.min_include_xy * scale if raw_cfg.min_include_xy else None,
        min_include_z_m=raw_cfg.min_include_z * scale if raw_cfg.min_include_z else None,
    )

    log.info(f"[NavMeshUnits] AFTER (world units in meters):")
    log.info(f"[NavMeshUnits]   - agent_radius_m: {world_config.agent_radius_m}")
    log.info(f"[NavMeshUnits]   - z_padding_m: {world_config.z_padding_m}")
    log.info(f"[NavMeshUnits]   - min_include_xy_m: {world_config.min_include_xy_m}")
    log.info(f"[NavMeshUnits]   - min_include_z_m: {world_config.min_include_z_m}")
    log.info(f"[NavMeshUnits] ✓ Conversion completed")
    log.info("=" * 70)

    return world_config
