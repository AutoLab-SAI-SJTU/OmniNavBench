"""Reusable start-position pool builder.

Extracted from ChainSampler so it can be used independently (e.g. by a
standalone script that only needs to sample positions without running the
full datagen pipeline).
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

import numpy as np

from datagen.core.geometry import GeometryInterface
from datagen.core.registry import ObjectRegistry

_OBJECT_TASK_TYPES = {"vln", "objectnav", "objnav", "eqa"}


def _log_info(msg: str) -> None:
    try:
        import carb  # type: ignore

        carb.log_info(msg)
    except Exception:
        print(msg)


class StartPoolBuilder:
    """Samples well-spread start positions on a NavMesh.

    Algorithm:
      1. Sample many random candidate points; keep those with sufficient
         clearance from walls (and optionally far enough from object targets).
      2. Greedy farthest-point selection: iteratively pick the candidate whose
         minimum distance to the already-selected set is largest.
    """

    def __init__(
        self,
        geometry: GeometryInterface,
        registry: ObjectRegistry,
        rng: random.Random,
    ):
        self._geo = geometry
        self._registry = registry
        self._rng = rng

    def build(
        self,
        num_episodes: int,
        min_clearance: float,
        min_dist: float,
        *,
        task_types: Optional[Sequence[str]] = None,
        object_start_min_dist: float = 0.0,
    ) -> List[np.ndarray]:
        """Return *num_episodes* start positions that are (a) away from walls
        and (b) spread out across the navmesh.

        Raises ``RuntimeError`` if there are not enough valid candidates or if
        the scene is too small to fit *num_episodes* points at the requested
        spacing.
        """
        n_candidates = max(200, num_episodes * 10)

        task_types_norm = {str(t).strip().lower() for t in (task_types or []) if t}
        object_targets = (
            list(self._registry.query())
            if any(t in _OBJECT_TASK_TYPES for t in task_types_norm)
            else []
        )
        object_start_min_dist = max(0.0, float(object_start_min_dist))

        candidates: List[np.ndarray] = []
        for _ in range(n_candidates):
            p = self._geo.sample_random_point()
            if p is None:
                continue
            if self._geo.get_clearance(p, num_rays=8) < float(min_clearance):
                continue
            p_arr = np.asarray(p, dtype=np.float32)
            if object_targets and object_start_min_dist > 0.0:
                nearest_target_xy = min(
                    float(np.linalg.norm((p_arr - np.asarray(obj.position, dtype=np.float32))[:2]))
                    for obj in object_targets
                )
                if nearest_target_xy < object_start_min_dist:
                    continue
            candidates.append(p_arr)

        if len(candidates) < num_episodes:
            raise RuntimeError(
                f"build_start_pool: only {len(candidates)} candidates passed "
                f"clearance={min_clearance:.3f} (need {num_episodes}). "
                "Lower NavMeshConfig.min_clearance_m or check scene navmesh coverage."
            )

        # Greedy farthest-point: guarantees maximum spread across candidates.
        selected: List[np.ndarray] = [self._rng.choice(candidates)]
        min_dists = np.array(
            [float(np.linalg.norm((c - selected[0])[:2])) for c in candidates],
            dtype=np.float64,
        )

        while len(selected) < num_episodes:
            best_idx = int(np.argmax(min_dists))
            if float(min_dists[best_idx]) < float(min_dist):
                raise RuntimeError(
                    f"build_start_pool: cannot place {num_episodes} starts with "
                    f"min_dist={min_dist:.3f} — best remaining gap is "
                    f"{float(min_dists[best_idx]):.3f}. "
                    "Reduce TaskConfig.start_min_dist_m or use a larger scene."
                )
            new_pt = candidates[best_idx]
            selected.append(new_pt)
            for i, c in enumerate(candidates):
                d = float(np.linalg.norm((c - new_pt)[:2]))
                if d < min_dists[i]:
                    min_dists[i] = d

        _log_info(
            f"[StartPoolBuilder] selected {len(selected)} starts "
            f"min_clearance={min_clearance:.3f} min_dist={min_dist:.3f} "
            f"candidates={len(candidates)}"
        )
        return selected
