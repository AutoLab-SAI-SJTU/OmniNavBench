from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from pxr import Semantics, Usd, UsdGeom

from datagen.config import RobotConfig


def _log_info(msg: str) -> None:
    try:
        import carb  # type: ignore

        carb.log_info(msg)
    except Exception:
        print(msg)


@dataclass
class ObjectEntry:
    object_id: str
    prim_path: str
    category: str
    position: np.ndarray
    room_id: Optional[str] = None

class ObjectRegistry:
    """
    Scans the stage and maintains a database of valid interaction targets.
    Applies strict height filtering during construction.
    """

    def __init__(self, stage: Usd.Stage, robot_cfg: RobotConfig):
        self._stage = stage
        self._robot_cfg = robot_cfg
        self._objects: Dict[str, ObjectEntry] = {}

    def build(self, external_objects: Optional[Sequence[Dict[str, object]]] = None):
        """
        Traverses the stage, filters objects, and builds the registry.
        """
        _log_info("[Registry] Building object registry...")

        self._objects.clear()

        if external_objects is not None:
            self._build_from_external(external_objects)
            _log_info(f"[Registry] Built with {len(self._objects)} external objects.")
            return
        
        # Traverse all prims
        # Use Usd.PrimRange for efficient traversal
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
        for prim in Usd.PrimRange(self._stage.GetPseudoRoot()):
            if not prim.IsA(UsdGeom.Imageable):
                continue
                
            # Check Semantics
            category = self._get_semantic_label(prim)
            if not category:
                continue
                
            # Compute an approximate world-space center.
            bound = bbox_cache.ComputeWorldBound(prim)
            range3d = bound.ComputeAlignedBox()
            
            min_pt = np.array(range3d.GetMin())
            max_pt = np.array(range3d.GetMax())
            center = (min_pt + max_pt) / 2.0
            
            entry = ObjectEntry(
                object_id=str(prim.GetPath()),
                prim_path=str(prim.GetPath()),
                category=category,
                position=center,
            )
            
            if self._filter_height(entry):
                self._objects[entry.object_id] = entry
                
        _log_info(f"[Registry] Built with {len(self._objects)} objects.")

    def _build_from_external(self, external_objects: Sequence[Dict[str, object]]) -> None:
        for idx, obj in enumerate(external_objects):
            object_id = obj.get("object_id")
            category = obj.get("category")
            position = obj.get("position")
            prim_path = obj.get("prim_path")
            room_id = obj.get("room_id")

            if not object_id:
                raise ValueError(f"External object #{idx} is missing object_id")
            if not category:
                raise ValueError(f"External object '{object_id}' is missing category")
            if not isinstance(position, (list, tuple)) or len(position) < 3:
                raise ValueError(f"External object '{object_id}' has invalid position")

            entry = ObjectEntry(
                object_id=str(object_id),
                prim_path=str(prim_path or f"/ExternalObjects/{object_id}"),
                category=str(category),
                position=np.asarray([float(position[0]), float(position[1]), float(position[2])], dtype=np.float32),
                room_id=str(room_id) if room_id else None,
            )

            if self._filter_height(entry):
                self._objects[entry.object_id] = entry

    def _get_semantic_label(self, prim) -> Optional[str]:
        """Retrieves semantic class label if present using SemanticsAPI."""
        # Isaac Sim convention: SemanticsAPI instance name "Semantics" with:
        # - semanticType="class"
        # - semanticData="Chair"
        for api_name in ("Semantics", "semantics", "Semantics_0", "Semantics_1"):
            try:
                sem_api = Semantics.SemanticsAPI.Get(prim, api_name)
            except Exception:
                sem_api = None
            if not sem_api:
                continue
            try:
                sem_type = sem_api.GetSemanticTypeAttr().Get()
                sem_data = sem_api.GetSemanticDataAttr().Get()
            except Exception:
                continue
            if sem_type and str(sem_type).lower() == "class" and sem_data:
                return str(sem_data)

        # Fallbacks: common attribute names
        for attr_name in (
            "semantics:semantictag",
            "semantics:semanticData",
            "semantics:data",
            "semantics:class",
            "semantic:class",
        ):
            try:
                attr = prim.GetAttribute(attr_name)
            except Exception:
                attr = None
            if not attr:
                continue
            try:
                if attr.HasValue():
                    val = attr.Get()
            except Exception:
                continue
            if isinstance(val, str) and val.strip():
                return val.strip()

        return None

    def _filter_height(self, entry: ObjectEntry) -> bool:
        """
        Returns False if object is too high/low for the robot's camera.
        """
        z_center = entry.position[2]
        # Check against Robot Config
        if z_center < self._robot_cfg.min_interaction_height:
            return False
        if z_center > self._robot_cfg.max_interaction_height:
            return False
        return True

    def query(self, category: str = None, room_id: str = None) -> List[ObjectEntry]:
        """
        Returns a list of objects matching criteria.
        """
        results = []
        for obj in self._objects.values():
            if category and obj.category != category:
                continue
            if room_id and obj.room_id != room_id:
                continue
            results.append(obj)
        return results
