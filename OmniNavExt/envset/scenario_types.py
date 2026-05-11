from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class SceneSpec:
    usd_path: Optional[str]
    scene_type: Optional[str]
    category: Optional[str]
    root_prim_path: Optional[str]
    navmesh_root_prim_path: Optional[str]
    units_in_meters: Optional[float]  # Scene unit size: 1 unit = X meters (e.g. 0.01 means centimeters).
    notes: Optional[str]
    raw: Dict[str, Any]


@dataclass
class NavmeshSpec:
    bake_root_prim_path: Optional[str]
    include_volume_parent: Optional[str]
    z_padding: Optional[float]
    agent_radius: Optional[float]
    max_step_height: Optional[float]
    min_include_xy: Optional[float]
    min_include_z: Optional[float]
    spawn_min_separation_m: Optional[float]
    raw: Dict[str, Any]


@dataclass
class SpawnPointSpec:
    name: Optional[str]
    position: Tuple[float, float, float]
    orientation_deg: Optional[float]
    raw: Dict[str, Any]


@dataclass
class RouteSpec:
    name: Optional[str]
    commands: Tuple[str, ...]
    raw: Dict[str, any]


@dataclass
class VirtualHumanSpec:
    category: Optional[str]
    count: Optional[int]
    name_sequence: Tuple[str, ...]
    assets: Dict[str, str]
    asset_root: Dict[str, Any]
    spawn_points: Tuple[SpawnPointSpec, ...]
    routes: Tuple[RouteSpec, ...]
    options: Dict[str, Any]


@dataclass
class RobotControlSpec:
    mode: Optional[str]
    module: Optional[str]
    entry: Optional[str]
    params: Dict[str, Any]


@dataclass
class RobotSpec:
    label: Optional[str]
    type: Optional[str]
    spawn_path: Optional[str]
    usd_path: Optional[str]
    scale: Optional[Tuple[float, float, float]]
    initial_position: Tuple[float, float, float]
    initial_orientation_deg: Optional[float]
    control: Optional[RobotControlSpec]
    raw: Dict[str, Any]


@dataclass
class EnvsetScenarioData:
    scene: SceneSpec
    navmesh: Optional[NavmeshSpec]
    virtual_humans: Optional[VirtualHumanSpec]
    robots: Tuple[RobotSpec, ...]
    logging: Dict[str, Any]
    raw: Dict[str, Any]
