from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence

class GeometryInterface(ABC):
    @abstractmethod
    def is_reachable(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Checks if a valid path exists between start and end on the NavMesh."""
        pass
    
    @abstractmethod
    def query_path(self, start: np.ndarray, end: np.ndarray) -> Optional[List[np.ndarray]]:
        """Returns the raw shortest path points from NavMesh."""
        pass

    @abstractmethod
    def get_clearance(self, position: np.ndarray, num_rays: int = 16, max_dist: float = 5.0) -> float:
        """
        Calculates the distance to the nearest obstacle at 'position'.
        Uses raycasting to approximate clearance.
        """
        pass

    @abstractmethod
    def set_random_seed(self, seed: int, name: str = "datagen") -> None:
        """Seed the NavMesh random sampler for reproducibility."""
        pass
    
    @abstractmethod
    def sample_random_point(self) -> Optional[np.ndarray]:
        """Returns a random point on the NavMesh."""
        pass

    @abstractmethod
    def snap_point(self, position: np.ndarray) -> np.ndarray:
        """Projects a point onto the NavMesh."""
        pass

class NavMeshGeometry(GeometryInterface):
    """
    Implementation of GeometryInterface using Omni NavMesh and PhysX Raycasts.
    """
    
    def __init__(self, navmesh_interface: Any):
        self._nav = navmesh_interface
        self._navmesh = self._nav.get_navmesh()

        self._carb = None
        self._physx_query = None
        try:
            import carb  # type: ignore

            self._carb = carb
        except Exception:
            self._carb = None

        # We need physx interface for raycasts (optional; used for clearance)
        try:
            import omni.physx  # type: ignore

            self._physx_query = omni.physx.get_physx_scene_query_interface()
        except Exception:
            self._physx_query = None
        
    def is_reachable(self, start: np.ndarray, end: np.ndarray) -> bool:
        if not self._navmesh:
            return False

        if self._carb is None:
            return False

        start_carb = self._carb.Float3(float(start[0]), float(start[1]), float(start[2]))
        end_carb = self._carb.Float3(float(end[0]), float(end[1]), float(end[2]))
        
        # Snap points first to ensure validity
        start_snapped = self._navmesh.query_closest_point(start_carb)
        end_snapped = self._navmesh.query_closest_point(end_carb)
        
        # Check path existence
        # query_shortest_path returns a path object or None/False if failed
        path = self._navmesh.query_shortest_path(start_snapped, end_snapped)
        
        # Depending on API version, path might be a list or object. 
        # OmniNavExt uses `if not navmesh.query_shortest_path(...)`
        return bool(path)

    def query_path(self, start: np.ndarray, end: np.ndarray) -> Optional[List[np.ndarray]]:
        if not self._navmesh:
            return None

        if self._carb is None:
            return None

        start_carb = self._carb.Float3(float(start[0]), float(start[1]), float(start[2]))
        end_carb = self._carb.Float3(float(end[0]), float(end[1]), float(end[2]))
        
        start_snapped = self._navmesh.query_closest_point(start_carb)
        end_snapped = self._navmesh.query_closest_point(end_carb)
        
        path_points = self._navmesh.query_shortest_path(start_snapped, end_snapped)
        
        if not path_points:
            return None

        points = _coerce_nav_path_points(path_points)
        if not points:
            return None
        return [np.asarray(p, dtype=np.float32) for p in points]

    def snap_point(self, position: np.ndarray) -> np.ndarray:
        if not self._navmesh:
            return position
        if self._carb is None:
            return position
        p = self._carb.Float3(float(position[0]), float(position[1]), float(position[2]))
        snapped = self._navmesh.query_closest_point(p)
        return np.array([snapped.x, snapped.y, snapped.z])

    def get_clearance(self, position: np.ndarray, num_rays: int = 16, max_dist: float = 5.0) -> float:
        """
        Approximates clearance by casting rays radially.
        """
        if self._physx_query is None or self._carb is None:
            return float(max_dist)

        min_dist = max_dist
        
        # Raycast height offset (e.g., 0.2m above ground)
        origin = self._carb.Float3(float(position[0]), float(position[1]), float(position[2]) + 0.2)
        
        for i in range(num_rays):
            angle = (2 * np.pi * i) / num_rays
            direction = self._carb.Float3(float(np.cos(angle)), float(np.sin(angle)), 0.0)

            hit = self._physx_query.raycast_closest(origin, direction, float(max_dist))
            hit_ok, dist = _parse_raycast_result(hit)
            if hit_ok and dist is not None and dist < min_dist:
                min_dist = dist
                    
        return float(min_dist)

    def set_random_seed(self, seed: int, name: str = "datagen") -> None:
        if not self._nav:
            raise RuntimeError("Nav interface unavailable for random seed")
        try:
            self._nav.set_random_seed(str(name), int(seed))
        except Exception as exc:
            raise RuntimeError(f"Failed to set NavMesh random seed: {exc}") from exc

    def sample_random_point(self) -> Optional[np.ndarray]:
        if not self._navmesh:
            return None
        try:
            area_count = int(self._navmesh.get_area_count())
        except Exception:
            area_count = 0
        area_probabilities = None
        if area_count > 0:
            area_probabilities = np.ones(area_count, dtype=np.float32)
        try:
            if area_probabilities is not None:
                sampled = self._navmesh.query_random_point("datagen", area_probabilities)
            else:
                sampled = self._navmesh.query_random_point("datagen")
        except TypeError:
            try:
                sampled = self._navmesh.query_random_point(area_probabilities)
            except Exception:
                return None
        except Exception:
            return None
        try:
            return np.asarray([float(sampled.x), float(sampled.y), float(sampled.z)], dtype=np.float32)
        except Exception:
            try:
                return np.asarray([float(sampled[0]), float(sampled[1]), float(sampled[2])], dtype=np.float32)
            except Exception:
                return None


def _coerce_nav_path_points(path: Any) -> List[Sequence[float]]:
    """Handle omni.anim.navigation return type differences across Isaac Sim versions."""
    if path is None:
        return []
    # Common case: list/tuple of carb.Float3-like objects
    if isinstance(path, (list, tuple)):
        out: List[Sequence[float]] = []
        for p in path:
            try:
                out.append((float(p.x), float(p.y), float(p.z)))
            except Exception:
                try:
                    out.append((float(p[0]), float(p[1]), float(p[2])))
                except Exception:
                    continue
        return out

    # Some bindings return a path object with .points or .get_points()
    for attr in ("points", "get_points"):
        if hasattr(path, attr):
            try:
                pts = getattr(path, attr)
                pts = pts() if callable(pts) else pts
                return _coerce_nav_path_points(pts)
            except Exception:
                continue
    return []


def _parse_raycast_result(hit: Any) -> tuple[bool, Optional[float]]:
    """Best-effort parsing for PhysX scene query raycast results."""
    if hit is None:
        return False, None
    if isinstance(hit, dict):
        ok = bool(hit.get("hit") or hit.get("hasHit") or hit.get("is_hit"))
        dist = hit.get("distance")
        return ok, None if dist is None else float(dist)
    # Some APIs return tuples; commonly (hit, distance, position, normal, collider)
    if isinstance(hit, (list, tuple)) and hit:
        ok = bool(hit[0])
        dist = None
        if len(hit) >= 2:
            try:
                dist = float(hit[1])
            except Exception:
                dist = None
        return ok, dist
    return False, None
