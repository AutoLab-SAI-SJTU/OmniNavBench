"""
Scene probe helper for Isaac Sim Script Editor.

Usage in Script Editor:
    import sys
    sys.path.append(r"/home/sun/PythonProjects/actorsgd/isaacsim.replicator.agent.core-0.7.19-107.3.1/docs")
    import scene_probe
    scene_probe.run(asset_root="/home/sun/PythonProjects/actorsgd/isaacsim.replicator.agent.core-0.7.19-107.3.1")

This prints:
- folders & (up to) 5 files in the given filesystem directory
- key stage metadata (metersPerUnit, timeline FPS)
- character prim positions
- environment layout under /Root and /World (with bounding boxes and root-scale info)
- NavMesh volumes and the NavMesh origin (closest point to each volume center)
"""

import os
import pathlib
from typing import Iterable, List, Optional, Tuple

import carb
import omni.timeline
import omni.usd
from pxr import Gf, Usd, UsdGeom

try:
    import omni.anim.navigation.core as nav
except ImportError:  # Script Editor should always have this, guard for completeness
    nav = None


RUNTIME_CONTAINERS = {
    "/World/Characters",
    "/World/Robots",
    "/World/Cameras",
    "/World/Lidars",
    "/World/NavMesh",
    "/World/Debug",
}


def _list_assets(root: pathlib.Path, file_limit: int = 5):
    folders: List[str] = []
    files: List[str] = []
    try:
        for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if entry.is_dir():
                folders.append(entry.name)
            else:
                files.append(entry.name)
    except Exception as exc:
        carb.log_error(f"[SceneProbe] Failed to read directory '{root}': {exc}")
        return

    carb.log_info(f"[SceneProbe] Directory listing for {root}:")
    if folders:
        for name in folders:
            carb.log_info(f"  [Dir]  {name}")
    else:
        carb.log_info("  (no sub-directories)")

    if files:
        for name in files[:file_limit]:
            carb.log_info(f"  [File] {name}")
        if len(files) > file_limit:
            carb.log_info(f"  ... (total files: {len(files)})")
    else:
        carb.log_info("  (no files)")


def _meters_per_unit(stage) -> float:
    value = 1.0
    try:
        value = float(UsdGeom.GetStageMetersPerUnit(stage))
    except Exception:
        pass
    if not value or value <= 0:
        try:
            raw = stage.GetMetadata("metersPerUnit")
            if raw:
                value = float(raw)
        except Exception:
            value = 1.0
    return value


def _describe_characters(stage):
    parent = stage.GetPrimAtPath("/World/Characters")
    if not parent or not parent.IsValid():
        carb.log_warn("[SceneProbe] /World/Characters not present.")
        return

    count = 0
    for prim in Usd.PrimRange(parent):
        if prim.GetTypeName() != "SkelRoot":
            continue
        count += 1
        matrix = omni.usd.get_world_transform_matrix(prim)
        pos = matrix.ExtractTranslation()
        carb.log_info(
            f"[SceneProbe] Character {prim.GetPath()}: position=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})"
        )
    if count == 0:
        carb.log_warn("[SceneProbe] No SkelRoot characters found under /World/Characters.")


def _bbox_for_prim(stage, prim) -> Optional[Tuple[Gf.Vec3d, Gf.Vec3d]]:
    try:
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], useExtentsHint=True)
        bound = bbox_cache.ComputeWorldBound(prim)
        aligned = bound.ComputeAlignedBox()
        return aligned.GetMin(), aligned.GetMax()
    except Exception:
        return None


def _describe_root(stage, root_path: str, skip_prefixes: Optional[Iterable[str]] = None):
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        carb.log_warn(f"[SceneProbe] {root_path} prim not found.")
        return

    xform = UsdGeom.Xformable(root)
    scales = [op.Get() for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
    if scales:
        carb.log_info(f"[SceneProbe] {root_path} applied scale(s): {scales}")

    skip_prefixes = set(skip_prefixes or [])
    carb.log_info(f"[SceneProbe] Children of {root_path}:")
    found_any = False
    for child in root.GetChildren():
        path = str(child.GetPath())
        if any(path.startswith(prefix) for prefix in skip_prefixes):
            continue
        found_any = True
        bbox = _bbox_for_prim(stage, child)
        if bbox:
            mn, mx = bbox
            center = ((mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5)
            size = (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])
            carb.log_info(f"  {path} ({child.GetTypeName()}): center={center}, size={size}")
        else:
            carb.log_info(f"  {path} ({child.GetTypeName()}): bbox unavailable")
    if not found_any:
        carb.log_warn(f"[SceneProbe] {root_path} has no qualifying child prims (all skipped or none present).")


def _describe_navmesh(stage):
    if nav is None:
        carb.log_error("[SceneProbe] omni.anim.navigation.core not available, cannot query NavMesh.")
        return

    interface = nav.acquire_interface()
    navmesh = interface.get_navmesh()
    if navmesh is None:
        carb.log_warn("[SceneProbe] NavMesh interface does not currently hold a mesh (baking incomplete?).")
    else:
        try:
            area_count = navmesh.get_area_count()
        except Exception:
            area_count = None
        carb.log_info(f"[SceneProbe] NavMesh ready: True, areas={area_count}")

    valid_types = {"NavMeshVolume", "NavMeshIncludeVolume", "NavMeshExcludeVolume"}
    for prim in stage.Traverse():
        if prim.GetTypeName() not in valid_types:
            continue
        xform = UsdGeom.Xformable(prim)
        translate = None
        scale = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and translate is None:
                translate = op.Get()
            elif op.GetOpType() == UsdGeom.XformOp.TypeScale and scale is None:
                scale = op.Get()
        carb.log_info(
            f"[SceneProbe] NavMesh volume {prim.GetPath()} (type={prim.GetTypeName()}): center={translate}, size={scale}"
        )
        if navmesh is not None and translate is not None:
            try:
                center_vec = carb.Float3(float(translate[0]), float(translate[1]), float(translate[2]))
                closest = navmesh.query_closest_point(center_vec)
                carb.log_info(
                    f"              closest walkable point=({closest.x:.4f}, {closest.y:.4f}, {closest.z:.4f})"
                )
            except Exception:
                carb.log_warn("              (failed to project volume center onto NavMesh)")


def run(asset_root: str = ".", file_limit: int = 5):
    """Entry point."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        carb.log_error("[SceneProbe] No USD stage is open.")
        return

    try:
        root_path = pathlib.Path(asset_root).resolve()
    except Exception:
        carb.log_error(f"[SceneProbe] Invalid asset_root: {asset_root}")
        root_path = pathlib.Path('.')

    _list_assets(root_path, file_limit=file_limit)

    meters = _meters_per_unit(stage)
    carb.log_info(f"[SceneProbe] Stage metersPerUnit = {meters}")

    # Timeline FPS metadata (optional)
    try:
        timeline = omni.timeline.get_timeline_interface()
        fps = timeline.get_time_codes_per_second()
        carb.log_info(f"[SceneProbe] Timeline FPS = {fps}")
    except Exception:
        pass

    _describe_characters(stage)
    _describe_root(stage, "/Root")
    _describe_root(stage, "/World", skip_prefixes=RUNTIME_CONTAINERS)
    _describe_navmesh(stage)


if __name__ == "__main__":
    run()
