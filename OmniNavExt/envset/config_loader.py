"""Utilities to merge OmniNav config with envset scenario data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from .scenario_types import (
    EnvsetScenarioData,
    NavmeshSpec,
    RobotControlSpec,
    RobotSpec,
    RouteSpec,
    SceneSpec,
    SpawnPointSpec,
    VirtualHumanSpec,
)

@dataclass
class EnvsetConfigBundle:
    """Result of merging the base config with an envset scenario."""

    config: Dict[str, Any]
    envset: Dict[str, Any]
    scenario_id: str
    scenario: Dict[str, Any]
    scenario_data: EnvsetScenarioData


class EnvsetConfigLoader:
    def __init__(
        self,
        config_path: Path,
        envset_path: Path,
        scenario_id: str | None = None,
        scene_root: Path | None = None,
    ):
        self._config_path = config_path
        self._envset_path = envset_path
        self._scenario_id = scenario_id
        self._scene_root = scene_root

    def load(self) -> EnvsetConfigBundle:
        config = self._load_config_yaml()
        envset = self._load_envset_json()
        scenario_id, scenario = self._select_scenario(envset)
        # Normalize envset file paths using explicit scene_root to avoid cwd fallback.
        self.normalize_scenario_paths(scenario, self._scene_root)
        scenario_data = self._build_scenario_data(scenario)
        # Store scenario_id for use in _apply_envset
        self._selected_scenario_id = scenario_id
        merged_config = self._apply_envset(config, scenario_data)
        return EnvsetConfigBundle(
            config=merged_config,
            envset=envset,
            scenario_id=scenario_id,
            scenario=scenario,
            scenario_data=scenario_data,
        )

    def _load_config_yaml(self) -> Dict[str, Any]:
        if not self._config_path.exists():
            raise FileNotFoundError(f"Config YAML not found: {self._config_path}")
        with self._config_path.open("r", encoding="utf-8") as fp:
            return yaml.safe_load(fp) or {}

    def _load_envset_json(self) -> Dict[str, Any]:
        if not self._envset_path.exists():
            raise FileNotFoundError(f"Envset JSON not found: {self._envset_path}")
        with self._envset_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def _select_scenario(self, envset: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        scenarios = envset.get("scenarios") or []
        if not scenarios:
            raise ValueError("Envset file does not define any scenarios")
        if self._scenario_id:
            for item in scenarios:
                if item.get("id") == self._scenario_id:
                    return self._scenario_id, item
            raise ValueError(f"Scenario '{self._scenario_id}' not found in envset")
        first = scenarios[0]
        scenario_id = first.get("id") or "scenario_0"
        return scenario_id, first

    @staticmethod
    def normalize_scenario_paths(scenario: Dict[str, Any], scene_root: Path | None) -> None:
        """Normalize envset file paths in-place using the provided scene_root."""
        if not isinstance(scenario, dict):
            return
        if scene_root is None:
            if EnvsetConfigLoader._scenario_has_asset_paths(scenario):
                raise ValueError("[Envset] --scene-root is required to resolve envset file paths.")
            return

        base = Path(scene_root).expanduser()
        if not base.is_absolute():
            raise ValueError(f"[Envset] --scene-root must be an absolute path: {base}")
        if not base.exists():
            raise FileNotFoundError(f"[Envset] scene_root not found: {base}")
        EnvsetConfigLoader._apply_scene_root_asset_path(base)

        def join_and_check(value: Any, label: str) -> Any:
            if value is None:
                return value
            value_str = str(value)
            if not value_str:
                return value
            resolved = (base / value_str).expanduser()
            if not resolved.exists():
                raise FileNotFoundError(f"[Envset] {label} not found: {resolved}")
            return str(resolved)

        scene_cfg = scenario.get("scene")
        if isinstance(scene_cfg, dict):
            if "usd_path" in scene_cfg:
                scene_cfg["usd_path"] = join_and_check(scene_cfg.get("usd_path"), "scene.usd_path")
            if "obj_path" in scene_cfg:
                scene_cfg["obj_path"] = join_and_check(scene_cfg.get("obj_path"), "scene.obj_path")
            if "asset_path" in scene_cfg:
                scene_cfg["asset_path"] = join_and_check(scene_cfg.get("asset_path"), "scene.asset_path")
            mp_cfg = scene_cfg.get("matterport")
            if isinstance(mp_cfg, dict):
                if "usd_path" in mp_cfg:
                    mp_cfg["usd_path"] = join_and_check(mp_cfg.get("usd_path"), "scene.matterport.usd_path")
                if "obj_path" in mp_cfg:
                    mp_cfg["obj_path"] = join_and_check(mp_cfg.get("obj_path"), "scene.matterport.obj_path")

        robots_cfg = scenario.get("robots")
        if isinstance(robots_cfg, dict):
            entries = robots_cfg.get("entries") or []
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict) and "usd_path" in entry:
                        entry["usd_path"] = join_and_check(entry.get("usd_path"), "robots.entries[].usd_path")

        vh_cfg = scenario.get("virtual_humans")
        if isinstance(vh_cfg, dict):
            asset_root = vh_cfg.get("asset_root")
            if isinstance(asset_root, dict):
                if "path" in asset_root:
                    asset_root["path"] = join_and_check(asset_root.get("path"), "virtual_humans.asset_root.path")
                if "fallback" in asset_root:
                    asset_root["fallback"] = join_and_check(asset_root.get("fallback"), "virtual_humans.asset_root.fallback")

    @staticmethod
    def _apply_scene_root_asset_path(scene_root: Path) -> None:
        """Bind OmniNav asset root to scene_root so controller weights resolve under dataset."""
        import os

        from OmniNav.macros import gm

        # MacroDict locks attributes after first access; allow idempotent reuse of the same scene_root.
        try:
            if "ASSET_PATH" in gm.get("_read", set()):
                current = os.path.abspath(str(gm.ASSET_PATH))
                desired = os.path.abspath(str(scene_root))
                if current == desired:
                    os.environ["OmniNav_ASSETS_PATH"] = str(scene_root)
                    return
                raise AttributeError(
                    "[Envset] OmniNav_ASSETS_PATH is locked to a different root; "
                    f"current={current} desired={desired}"
                )
        except Exception:
            pass

        os.environ["OmniNav_ASSETS_PATH"] = str(scene_root)
        try:
            gm.ASSET_PATH = str(scene_root)
        except AttributeError as exc:
            current = None
            try:
                if isinstance(gm, dict):
                    current = gm.get("ASSET_PATH")
            except Exception:
                current = None
            if current and Path(current).resolve() == scene_root.resolve():
                return
            raise AttributeError(
                "[Envset] OmniNav_ASSETS_PATH is locked; restart with --scene-root or set it before imports."
            ) from exc

    @staticmethod
    def _scenario_has_asset_paths(scenario: Dict[str, Any]) -> bool:
        scene_cfg = scenario.get("scene") if isinstance(scenario.get("scene"), dict) else {}
        if scene_cfg.get("usd_path") or scene_cfg.get("obj_path") or scene_cfg.get("asset_path"):
            return True
        mp_cfg = scene_cfg.get("matterport") if isinstance(scene_cfg.get("matterport"), dict) else {}
        if mp_cfg.get("usd_path") or mp_cfg.get("obj_path"):
            return True
        robots_cfg = scenario.get("robots") if isinstance(scenario.get("robots"), dict) else {}
        for entry in robots_cfg.get("entries") or []:
            if isinstance(entry, dict) and entry.get("usd_path"):
                return True
        vh_cfg = scenario.get("virtual_humans") if isinstance(scenario.get("virtual_humans"), dict) else {}
        asset_root = vh_cfg.get("asset_root") if isinstance(vh_cfg.get("asset_root"), dict) else {}
        return bool(asset_root.get("path") or asset_root.get("fallback"))

    def _apply_envset(self, base: Dict[str, Any], scenario_data: EnvsetScenarioData) -> Dict[str, Any]:
        from .task_adapter import EnvsetTaskAugmentor

        merged = dict(base or {})
        scene_section = dict(merged.get("scene") or {})
        scene_cfg = scenario_data.scene.raw

        usd_path = scene_cfg.get("usd_path") or scene_cfg.get("asset_path")
        if usd_path:
            scene_section["asset_path"] = usd_path

        use_matterport = scene_cfg.get("use_matterport")
        if use_matterport is None:
            category = str(scene_cfg.get("category") or "").lower()
            use_matterport = category == "mp3d"
        scene_section["use_matterport"] = bool(use_matterport)

        merged["scene"] = scene_section
        # Use the selected scenario_id (from _select_scenario) instead of self._scenario_id
        scenario_id = getattr(self, '_selected_scenario_id', self._scenario_id)
        merged = EnvsetTaskAugmentor.apply(merged, scenario_data, scenario_id=scenario_id)
        return merged

    def _build_scenario_data(self, scenario: Dict[str, Any]) -> EnvsetScenarioData:
        scene = self._build_scene_spec(scenario.get("scene") or {})
        navmesh = self._build_navmesh_spec(scenario.get("navmesh"))
        vh = self._build_virtual_humans_spec(scenario.get("virtual_humans"))
        robots = tuple(self._build_robot_spec(entry) for entry in scenario.get("robots", {}).get("entries", []))
        logging = scenario.get("logging") or {}
        return EnvsetScenarioData(
            scene=scene,
            navmesh=navmesh,
            virtual_humans=vh,
            robots=robots,
            logging=logging,
            raw=scenario,
        )

    def _build_scene_spec(self, data: Dict[str, Any]) -> SceneSpec:
        # Normalize use_matterport flag for MP3D category when missing.
        try:
            category = str(data.get("category") or "").lower()
            if "mp3d" in category and "use_matterport" not in data:
                data = dict(data)
                data["use_matterport"] = True
        except Exception:
            pass
        return SceneSpec(
            usd_path=data.get("usd_path"),
            scene_type=data.get("type"),
            category=data.get("category"),
            root_prim_path=data.get("root_prim_path"),
            navmesh_root_prim_path=data.get("navmesh_root_prim_path"),
            units_in_meters=self._safe_float(data.get("units_in_meters")),
            notes=data.get("notes"),
            raw=data,
        )

    def _build_navmesh_spec(self, data: Optional[Dict[str, Any]]) -> Optional[NavmeshSpec]:
        if not data:
            return None
        min_size = data.get("min_include_volume_size") or {}
        return NavmeshSpec(
            bake_root_prim_path=data.get("bake_root_prim_path"),
            include_volume_parent=data.get("include_volume_parent"),
            z_padding=data.get("z_padding"),
            agent_radius=data.get("agent_radius"),
            max_step_height=data.get("max_step_height"),
            min_include_xy=min_size.get("xy"),
            min_include_z=min_size.get("z"),
            spawn_min_separation_m=data.get("spawn_min_separation_m"),
            raw=data,
        )

    def _build_virtual_humans_spec(self, data: Optional[Dict[str, Any]]) -> Optional[VirtualHumanSpec]:
        if not data:
            return None
        name_sequence = tuple(str(name) for name in data.get("name_sequence", []) if name is not None)
        spawn_points = tuple(self._build_spawn_point_spec(item) for item in data.get("spawn_points", []))
        routes_raw = data.get("move_routes") if data.get("move_routes") is not None else data.get("routes", [])
        routes = tuple(self._build_route_spec(item) for item in (routes_raw or []))
        assets = {}
        raw_assets = data.get("assets") or {}
        for key, value in raw_assets.items():
            assets[str(key)] = str(value)
        return VirtualHumanSpec(
            category=data.get("category"),
            count=data.get("count"),
            name_sequence=name_sequence,
            assets=assets,
            asset_root=data.get("asset_root") or {},
            spawn_points=spawn_points,
            routes=routes,
            options=data,
        )

    def _build_spawn_point_spec(self, data: Dict[str, Any]) -> SpawnPointSpec:
        pos = tuple(float(v) for v in data.get("position", (0.0, 0.0, 0.0)))
        return SpawnPointSpec(
            name=data.get("name"),
            position=(pos + (0.0, 0.0, 0.0))[:3],
            orientation_deg=self._safe_float(data.get("orientation_deg")),
            raw=data,
        )

    def _build_route_spec(self, data: Dict[str, Any]) -> RouteSpec:
        commands = tuple(str(cmd) for cmd in data.get("commands", []) if cmd)
        return RouteSpec(name=data.get("name"), commands=commands, raw=data)

    def _build_robot_spec(self, data: Dict[str, Any]) -> RobotSpec:
        initial_pose = data.get("initial_pose") or {}
        position = tuple(float(v) for v in initial_pose.get("position", (0.0, 0.0, 0.0)))
        orientation_deg = self._safe_float(initial_pose.get("orientation_deg"))
        scale = self._parse_scale(data.get("scale"))
        # Camera configuration is not supported in envset; use the robot USD-authored camera prim.
        if "camera" in data:
            raise ValueError("[Envset] robots.entries[].camera is not supported; use the robot's built-in camera (e.g., trunk/Camera).")
        control_cfg = data.get("control") or None
        control = None
        if control_cfg:
            control = RobotControlSpec(
                mode=control_cfg.get("mode"),
                module=control_cfg.get("module"),
                entry=control_cfg.get("entry"),
                params=control_cfg.get("params") or {},
            )
        return RobotSpec(
            label=data.get("label"),
            type=data.get("type"),
            spawn_path=data.get("spawn_path"),
            usd_path=data.get("usd_path"),
            scale=scale,
            initial_position=(position + (0.0, 0.0, 0.0))[:3],
            initial_orientation_deg=orientation_deg,
            control=control,
            raw=data,
        )

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _parse_scale(cls, value: Any) -> Optional[Tuple[float, float, float]]:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            scalar = float(value)
            return (scalar, scalar, scalar)

        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                scalar = cls._safe_float(value[0])
                if scalar is None:
                    return None
                return (scalar, scalar, scalar)
            if len(value) >= 3:
                sx = cls._safe_float(value[0])
                sy = cls._safe_float(value[1])
                sz = cls._safe_float(value[2])
                if sx is None or sy is None or sz is None:
                    return None
                return (sx, sy, sz)

        return None
