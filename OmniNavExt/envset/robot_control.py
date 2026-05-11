from __future__ import annotations

import asyncio
import copy
import importlib
from typing import Any, Dict, List, Optional

import carb

class RobotControlHandle:
    """Wrapper that standardizes the stop lifecycle for arbitrary control objects."""

    def __init__(self, handle: Any):
        self._handle = handle

    async def stop(self):
        target = self._handle
        if target is None:
            return
        stop_fn = getattr(target, "stop", None)
        if stop_fn is None:
            return
        if asyncio.iscoroutinefunction(stop_fn):
            await stop_fn()
        else:
            result = stop_fn()
            if asyncio.iscoroutine(result):
                await result


class RobotControlManager:
    """Launches and tracks robot control scripts defined via envset configuration."""

    def __init__(self):
        self._scene_cfg: Optional[dict] = None
        self._handles: List[RobotControlHandle] = []

    def set_scene_config(self, scene_cfg: Optional[dict]):
        self._scene_cfg = scene_cfg

    async def start_controls(self):
        await self.stop_all()
        if not self._scene_cfg:
            return

        robots = self._scene_cfg.get("robots", {}).get("entries", [])
        for robot in robots:
            control_cfg = robot.get("control")
            if not control_cfg:
                continue
            module_name = control_cfg.get("module")
            entry_name = control_cfg.get("entry", "launch")
            params = dict(control_cfg.get("params", {}))
            scene_logging = self._resolve_scene_logging(robot)
            param_logging = params.get("logging")
            merged_logging = self._merge_logging(scene_logging, param_logging)
            if merged_logging:
                scene_objects = None
                if self._scene_cfg and isinstance(self._scene_cfg, dict):
                    scene_objects = self._scene_cfg.get("objects")
                if scene_objects is not None and "objects" not in merged_logging:
                    merged_logging["objects"] = copy.deepcopy(scene_objects)
                params["logging"] = merged_logging
            if "robot_path" not in params:
                spawn_path = robot.get("spawn_path")
                if spawn_path:
                    params.setdefault("robot_path", spawn_path)
            if not module_name:
                carb.log_warn("[RobotControl] Skipping control without module path.")
                continue
            try:
                module = importlib.import_module(module_name)
                entry = getattr(module, entry_name)
            except Exception as exc:  # noqa: BLE001
                carb.log_error(f"[RobotControl] Failed to import {module_name}.{entry_name}: {exc}")
                continue
            try:
                result = entry(**params)
                if asyncio.iscoroutine(result):
                    result = await result
                if result is None:
                    continue
                self._handles.append(RobotControlHandle(result))
                carb.log_info(f"[RobotControl] Started control via {module_name}.{entry_name}")
            except Exception as exc:  # noqa: BLE001
                carb.log_error(f"[RobotControl] Control launch failed for {module_name}.{entry_name}: {exc}")

    async def stop_all(self):
        if not self._handles:
            return
        for handle in self._handles:
            try:
                await handle.stop()
            except Exception as exc:  # noqa: BLE001
                carb.log_warn(f"[RobotControl] Control stop raised: {exc}")
        self._handles.clear()

    def _resolve_scene_logging(self, robot_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._scene_cfg:
            return None
        scene_logging = self._scene_cfg.get("logging")
        if not scene_logging:
            return None
        label = robot_cfg.get("label")
        robot_type = robot_cfg.get("type")

        def _clone(data):
            return dict(data) if isinstance(data, dict) else None

        if isinstance(scene_logging, dict):
            selected: Optional[Dict[str, Any]] = None
            robots_map = scene_logging.get("robots")
            if isinstance(robots_map, dict):
                if label and isinstance(robots_map.get(label), dict):
                    selected = _clone(robots_map[label])
                elif robot_type and isinstance(robots_map.get(robot_type), dict):
                    selected = _clone(robots_map[robot_type])
            if selected is None:
                default_cfg = scene_logging.get("default")
                if isinstance(default_cfg, dict):
                    selected = _clone(default_cfg)
                else:
                    # Use the top-level dict minus nested maps as fallback
                    selected = {
                        k: v
                        for k, v in scene_logging.items()
                        if k not in {"robots", "default"}
                    }
                    if not selected:
                        selected = None
            return selected
        return None

    @staticmethod
    def _merge_logging(scene_logging: Optional[Dict[str, Any]], param_logging: Any) -> Optional[Dict[str, Any]]:
        if param_logging is None and scene_logging is None:
            return None
        merged: Dict[str, Any] = {}
        if isinstance(scene_logging, dict):
            merged.update(scene_logging)
        if isinstance(param_logging, dict):
            merged.update(param_logging)
        return merged or None
