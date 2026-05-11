"""Top-down oracle map + frontier sampling for MTU3D (Isaac Sim).

MTU3D's standard loop uses:
  - an oracle top-down occupancy map (whole house / whole floor),
  - a fog-of-war mask (revealed by the agent's camera FOV),
  - frontier candidates extracted from explored/unexplored boundaries.

This module re-implements that logic for OmniNavBench without Habitat-Lab.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class Bounds3D:
    """Axis-aligned bounds in world coordinates (stage units)."""

    min_xyz: Tuple[float, float, float]
    max_xyz: Tuple[float, float, float]


@dataclass(frozen=True)
class TopdownMap:
    """Binary free-space map and its world mapping."""

    free_mask: np.ndarray  # uint8 {0,1}, shape (H, W); 1=free, 0=occupied
    resolution: float  # meters per pixel (assumes stage units are meters)
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """World (x,y) -> (row, col) into free_mask."""
        col = int((x - self.min_x) / self.resolution)
        row = int((self.max_y - y) / self.resolution)
        return row, col

    def grid_to_world(self, row: int, col: int, z: float) -> Tuple[float, float, float]:
        """(row, col) -> world (x,y,z) at cell center."""
        x = self.min_x + (col + 0.5) * self.resolution
        y = self.max_y - (row + 0.5) * self.resolution
        return float(x), float(y), float(z)


def resolve_navmesh_bake_bounds() -> Bounds3D:
    """Resolve bounds from existing NavMesh include volumes (baked region)."""
    try:
        import omni.usd  # type: ignore
        from pxr import Usd, UsdGeom  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError("resolve_navmesh_bake_bounds() must be called inside Isaac Sim (omni.usd required)") from e

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage not available")

    # Match Isaac Sim schemas used by navmesh_utils.
    valid_types = {"NavMeshVolume", "NavMeshIncludeVolume"}
    volumes = [prim for prim in stage.Traverse() if prim.GetTypeName() in valid_types]
    if not volumes:
        raise RuntimeError("No NavMesh include volumes found; ensure NavMesh was baked before MTU3D init")

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], useExtentsHint=True)
    mn_all: Optional[Tuple[float, float, float]] = None
    mx_all: Optional[Tuple[float, float, float]] = None
    for prim in volumes:
        try:
            aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        except Exception as e:
            raise RuntimeError(f"Failed to compute NavMesh volume bbox for prim {prim.GetPath()}: {e}") from e
        mn = aligned.GetMin()
        mx = aligned.GetMax()
        cur_mn = (float(mn[0]), float(mn[1]), float(mn[2]))
        cur_mx = (float(mx[0]), float(mx[1]), float(mx[2]))
        if mn_all is None or mx_all is None:
            mn_all, mx_all = cur_mn, cur_mx
        else:
            mn_all = (min(mn_all[0], cur_mn[0]), min(mn_all[1], cur_mn[1]), min(mn_all[2], cur_mn[2]))
            mx_all = (max(mx_all[0], cur_mx[0]), max(mx_all[1], cur_mx[1]), max(mx_all[2], cur_mx[2]))

    if mn_all is None or mx_all is None:
        raise RuntimeError("Failed to compute NavMesh bake bounds from volumes")
    return Bounds3D(
        min_xyz=mn_all,
        max_xyz=mx_all,
    )


def generate_oracle_topdown_map(bounds: Bounds3D, resolution_m: float = 0.05) -> TopdownMap:
    """Generate an oracle free-space top-down map using Isaac's occupancy map generator."""
    if resolution_m <= 0:
        raise ValueError("resolution_m must be > 0")

    try:
        import omni
        from isaacsim.asset.gen.omap.bindings import _omap
    except ModuleNotFoundError as e:
        raise RuntimeError("generate_oracle_topdown_map() must be called inside Isaac Sim (isaacsim.asset.gen.omap required)") from e

    physx = omni.physx.acquire_physx_interface()
    stage_id = omni.usd.get_context().get_stage_id()
    generator = _omap.Generator(physx, stage_id)

    # Encode values: occupied/free/unknown (values don't matter as long as consistent).
    occ_val, free_val, unk_val = 4, 5, 6
    generator.update_settings(float(resolution_m), occ_val, free_val, unk_val)

    min_x, min_y, min_z = bounds.min_xyz
    max_x, max_y, max_z = bounds.max_xyz
    
    scan_min_z = min_z + 0.1  # Avoid carpet/floor unevenness.
    scan_max_z = min(min_z + 2.0, max_z)  # Avoid the ceiling.
    
    origin = (0.0, 0.0, 0.0)
    generator.set_transform(origin, (min_x, min_y, scan_min_z), (max_x, max_y, scan_max_z))

    generator.generate2d()
    buf = generator.get_buffer()
    if buf is None:
        raise RuntimeError("Occupancy generator returned empty buffer")

    # Buffer comes as a 2D numpy array in most Isaac builds; normalize to ndarray.
    if len(buf) > 0 and isinstance(buf[0], str):
        # Convert char list to bytes, then to a uint8 numpy array.
        data_bytes = "".join(buf).encode('latin1')
        grid = np.frombuffer(data_bytes, dtype=np.uint8)
    else:
        grid = np.asarray(buf, dtype=np.uint8)
        
    dims = generator.get_dimensions()
    w, h = dims[0], dims[1]  # Note: the API may return (W, H).
    if grid.shape[0] != w * h:
        raise RuntimeError(f"Unexpected occupancy buffer shape: {grid.shape}")
    grid = grid.reshape((h, w))

    free_mask = (grid == free_val).astype(np.uint8)
    h, w = free_mask.shape
    if h <= 1 or w <= 1:
        raise RuntimeError(f"Occupancy buffer too small: shape={free_mask.shape}")

    return TopdownMap(
        free_mask=free_mask,
        resolution=float(resolution_m),
        min_x=float(min_x),
        min_y=float(min_y),
        max_x=float(max_x),
        max_y=float(max_y),
    )


class FogOfWarFrontier:
    """Maintain fog-of-war + compute frontier candidates on a fixed oracle map."""

    def __init__(
        self,
        topdown: TopdownMap,
        hfov_deg: float = 42.0,
        max_range_m: float = 8.0,
        rays: int = 121,
    ) -> None:
        if hfov_deg <= 0:
            raise ValueError("hfov_deg must be > 0")
        if max_range_m <= 0:
            raise ValueError("max_range_m must be > 0")
        if rays <= 1:
            raise ValueError("rays must be > 1")

        self._map = topdown
        self._hfov_rad = float(np.deg2rad(hfov_deg))
        self._max_steps = int(np.ceil(float(max_range_m) / self._map.resolution))
        self._rays = int(rays)

        self._explored = np.zeros_like(self._map.free_mask, dtype=np.uint8)
        self._last_z: float = 0.0

    def reset(self) -> None:
        self._explored.fill(0)
        self._last_z = 0.0

    @property
    def explored_mask(self) -> np.ndarray:
        return self._explored

    def update(self, position_xyz: Tuple[float, float, float], yaw_rad: float) -> None:
        """Reveal visible free cells from current pose (ray-march on oracle occupancy).
        Fast version: inline world_to_grid + incremental stepping.
        """
        x, y, z = position_xyz
        self._last_z = float(z)

        free = self._map.free_mask
        explored = self._explored
        H, W = free.shape

        # Pull constants into locals (faster than attribute lookups)
        res = float(self._map.resolution)
        inv_res = 1.0 / res
        min_x = float(self._map.min_x)
        max_y = float(self._map.max_y)

        fx = float(x)
        fy = float(y)

        # Inline world_to_grid for current pose
        col0 = int((fx - min_x) * inv_res)
        row0 = int((max_y - fy) * inv_res)

        if row0 < 0 or row0 >= H or col0 < 0 or col0 >= W:
            raise RuntimeError("Robot position is outside oracle map bounds; check navmesh include volume bounds")

        # Reveal current cell if it is free.
        if free[row0, col0] == 1:
            explored[row0, col0] = 1

        # Cast rays within HFOV centered at yaw.
        half = self._hfov_rad * 0.5
        angles = np.linspace(yaw_rad - half, yaw_rad + half, int(self._rays))

        # Optional micro-opt: precompute trig arrays once
        cos_vals = np.cos(angles)
        sin_vals = np.sin(angles)

        max_steps = int(self._max_steps)

        for i in range(cos_vals.shape[0]):
            dx = float(cos_vals[i])
            dy = float(sin_vals[i])

            # Incremental world stepping: avoid (step * res) each iteration
            wx = fx
            wy = fy

            # Per-step delta in world units
            step_dx = dx * res
            step_dy = dy * res

            for _ in range(max_steps):
                wx += step_dx
                wy += step_dy

                c = int((wx - min_x) * inv_res)
                r = int((max_y - wy) * inv_res)

                if r < 0 or r >= H or c < 0 or c >= W:
                    break
                if free[r, c] == 0:
                    break
                explored[r, c] = 1

    def sample_frontiers(
        self,
        max_candidates: int = 64,
        min_separation_m: float = 0.5,
    ) -> List[Tuple[float, float, float]]:
        """Extract frontier cells and return sampled world waypoints."""
        if max_candidates <= 0:
            raise ValueError("max_candidates must be > 0")
        if min_separation_m <= 0:
            raise ValueError("min_separation_m must be > 0")

        free = self._map.free_mask
        explored = self._explored

        # A frontier cell is explored+free and adjacent to at least one unexplored+free cell.
        unexplored_free = (free == 1) & (explored == 0)
        explored_free = (free == 1) & (explored == 1)

        # 4-neighborhood adjacency (cheap and stable).
        up = np.zeros_like(unexplored_free); up[1:] = unexplored_free[:-1]
        dn = np.zeros_like(unexplored_free); dn[:-1] = unexplored_free[1:]
        lf = np.zeros_like(unexplored_free); lf[:, 1:] = unexplored_free[:, :-1]
        rt = np.zeros_like(unexplored_free); rt[:, :-1] = unexplored_free[:, 1:]
        neighbor_unexplored = up | dn | lf | rt

        frontier = explored_free & neighbor_unexplored
        rows, cols = np.where(frontier)
        if rows.size == 0:
            return []

        # Greedy thinning using a grid bucket based on min_separation.
        stride = max(1, int(np.ceil(min_separation_m / self._map.resolution)))
        picked = {}
        for r, c in zip(rows.tolist(), cols.tolist()):
            br = r // stride
            bc = c // stride
            key = (br, bc)
            if key in picked:
                continue
            picked[key] = (r, c)
            if len(picked) >= max_candidates:
                break

        out: List[Tuple[float, float, float]] = []
        for r, c in picked.values():
            out.append(self._map.grid_to_world(int(r), int(c), z=self._last_z))
        return out
