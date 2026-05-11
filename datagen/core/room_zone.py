from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from datagen.core.geometry import GeometryInterface
from datagen.core.registry import ObjectEntry, ObjectRegistry


def _log_info(msg: str) -> None:
    try:
        import carb  # type: ignore

        carb.log_info(msg)
    except Exception:
        print(msg)


def _log_warn(msg: str) -> None:
    try:
        import carb  # type: ignore

        carb.log_warn(msg)
    except Exception:
        print(f"[WARN] {msg}")


@dataclass(frozen=True)
class RoomRegion:
    room_id: str
    bbox_min: Tuple[float, float, float]
    bbox_max: Tuple[float, float, float]
    centroid: Tuple[float, float, float]

class RoomZoning:
    """
    Handles geometric room segmentation and semantic naming.
    Uses object clustering to infer zones in the absence of GT room labels.
    """
    
    def __init__(self, geometry: GeometryInterface, registry: ObjectRegistry):
        self._geo = geometry
        self._registry = registry
        
        self._room_centroids: Dict[str, np.ndarray] = {}
        self._room_names: Dict[str, str] = {}
        self._room_evidence: Dict[str, List[str]] = {}
        self._room_regions: Dict[str, RoomRegion] = {}
        self._object_room_map: Dict[str, str] = {}
        self._adjacency: List[Tuple[str, str]] = []
        
        # Heuristic rules for evidence-driven naming
        self._rules = {
            "bathroom": {"toilet", "sink", "bathtub", "shower"},
            "bedroom": {"bed", "wardrobe"},
            "kitchen": {"refrigerator", "oven", "stove", "kitchen_counter"},
            "living_room": {"sofa", "television", "coffee_table"},
            "dining_room": {"dining_table", "chair"}
        }

    def compute_zones(self, eps: float = 2.5, min_samples: int = 3) -> None:
        """Infer room-like zones from object clusters (no GT room labels).

        This is a deterministic, dependency-free approximation of DBSCAN-style clustering.
        """
        _log_info("[RoomZoning] Computing zones from object registry...")

        objects = sorted(self._registry.query(), key=lambda o: o.prim_path)
        if not objects:
            _log_warn("[RoomZoning] No objects found. Cannot compute zones.")
            return

        points = np.array([obj.position for obj in objects], dtype=np.float32)
        labels = _cluster_dbscan_like(points, eps=float(eps), min_samples=int(min_samples))

        cluster_map: Dict[int, List[ObjectEntry]] = {}
        for obj, label in zip(objects, labels):
            if label < 0:
                continue
            cluster_map.setdefault(label, []).append(obj)

        # Fallback: if everything is noise, treat all objects as one zone.
        if not cluster_map:
            cluster_map = {0: objects}

        for label, cluster_objs in sorted(cluster_map.items(), key=lambda kv: kv[0]):
            room_id = f"room_{label}"
            positions = np.array([o.position for o in cluster_objs], dtype=np.float32)
            centroid = np.mean(positions, axis=0)
            bbox_min = np.min(positions, axis=0)
            bbox_max = np.max(positions, axis=0)

            self._room_centroids[room_id] = centroid
            self._room_regions[room_id] = RoomRegion(
                room_id=room_id,
                bbox_min=tuple(float(v) for v in bbox_min),
                bbox_max=tuple(float(v) for v in bbox_max),
                centroid=tuple(float(v) for v in centroid),
            )

            name, evidence = self._resolve_name(cluster_objs)
            self._room_names[room_id] = name
            self._room_evidence[room_id] = evidence

            for obj in cluster_objs:
                obj.room_id = room_id
                self._object_room_map[obj.prim_path] = room_id

        self._adjacency = self._compute_adjacency(max_path_m=25.0)
        self._apply_corridor_heuristic()

        _log_info(f"[RoomZoning] Identified {len(self._room_centroids)} zones.")

    def _resolve_name(self, objects: List[ObjectEntry]) -> Tuple[str, List[str]]:
        """Determines room name based on object categories."""
        categories = [obj.category.lower() for obj in objects]
        counts = Counter(categories)
        
        best_match = "unknown"
        max_score = 0
        evidence_found = []

        for room_type, keywords in self._rules.items():
            score = 0
            current_evidence = []
            for k in keywords:
                if counts[k] > 0:
                    score += counts[k]
                    current_evidence.append(k)
            
            if score > max_score:
                max_score = score
                best_match = room_type
                evidence_found = current_evidence
        
        return best_match, evidence_found

    def get_room_id(self, position: np.ndarray) -> str:
        """
        Returns the ID of the nearest room cluster.
        """
        best_id = "unknown"
        min_dist = float('inf')
        
        for r_id, centroid in self._room_centroids.items():
            dist = np.linalg.norm(position - centroid)
            if dist < min_dist:
                min_dist = dist
                best_id = r_id
                
        # Threshold (e.g. if > 5m away from any cluster, it's corridor/unknown)
        if min_dist > 5.0:
            return "unknown"
            
        return best_id
        
    def get_room_name(self, room_id: str) -> str:
        return self._room_names.get(room_id, "unknown")

    def as_room_zone_payload(self) -> Dict[str, Any]:
        """Serialize room zoning results for scenario JSON."""
        regions = {
            room_id: {
                "bbox_min": list(region.bbox_min),
                "bbox_max": list(region.bbox_max),
                "centroid": list(region.centroid),
            }
            for room_id, region in self._room_regions.items()
        }
        room_name_map = {
            room_id: {"name": self.get_room_name(room_id), "evidence_objects": self._room_evidence.get(room_id, [])}
            for room_id in self._room_regions.keys()
        }
        return {
            "regions": regions,
            "adjacency": [list(edge) for edge in self._adjacency],
            "object_room_map": dict(self._object_room_map),
            "room_name_map": room_name_map,
        }

    def _compute_adjacency(self, max_path_m: float) -> List[Tuple[str, str]]:
        room_ids = sorted(self._room_centroids.keys())
        edges: List[Tuple[str, str]] = []
        for i, a in enumerate(room_ids):
            for b in room_ids[i + 1 :]:
                path = self._geo.query_path(self._room_centroids[a], self._room_centroids[b])
                if not path:
                    continue
                length = _polyline_length(path)
                if length <= float(max_path_m):
                    edges.append((a, b))
        return edges

    def _apply_corridor_heuristic(self) -> None:
        """Post-process unknown rooms into corridors based on geometry + topology."""
        degree: Dict[str, int] = {room_id: 0 for room_id in self._room_regions.keys()}
        for a, b in self._adjacency:
            degree[a] += 1
            degree[b] += 1

        for room_id, region in self._room_regions.items():
            if self._room_names.get(room_id, "unknown") != "unknown":
                continue
            bbox_min = np.array(region.bbox_min, dtype=np.float32)
            bbox_max = np.array(region.bbox_max, dtype=np.float32)
            size_xy = bbox_max[:2] - bbox_min[:2]
            lo = float(min(size_xy[0], size_xy[1]) + 1e-6)
            hi = float(max(size_xy[0], size_xy[1]) + 1e-6)
            aspect = hi / lo
            if degree.get(room_id, 0) >= 3 and aspect >= 4.0:
                self._room_names[room_id] = "corridor"


def _polyline_length(points: List[np.ndarray]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(points[:-1], points[1:]):
        total += float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
    return total


def _cluster_dbscan_like(points: np.ndarray, eps: float, min_samples: int) -> List[int]:
    """A tiny, deterministic DBSCAN-like clustering (no external deps).

    Notes:
      - Points with < min_samples neighbors within eps are labeled as noise (-1).
      - This is O(n^2); acceptable for moderate object counts.
    """
    n = int(points.shape[0])
    if n == 0:
        return []
    labels = [-1] * n
    visited = [False] * n
    cluster_id = 0

    # Precompute squared distances for speed.
    eps2 = float(eps) * float(eps)

    def neighbors(idx: int) -> List[int]:
        p = points[idx]
        out: List[int] = []
        for j in range(n):
            d = p - points[j]
            if float(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) <= eps2:
                out.append(j)
        return out

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neigh = neighbors(i)
        if len(neigh) < int(min_samples):
            labels[i] = -1
            continue

        # Expand cluster
        labels[i] = cluster_id
        queue = list(neigh)
        while queue:
            j = queue.pop()
            if not visited[j]:
                visited[j] = True
                neigh_j = neighbors(j)
                if len(neigh_j) >= int(min_samples):
                    for k in neigh_j:
                        if k not in queue:
                            queue.append(k)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1

    return labels
