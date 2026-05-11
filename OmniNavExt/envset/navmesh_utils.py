try:
    import carb
    import omni.anim.navigation.core as nav
    import omni.kit.commands
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom
    ISAAC_AVAILABLE = True
except ImportError:
    carb = None
    nav = None
    ISAAC_AVAILABLE = False

import asyncio
from typing import Optional


def _list_navmesh_volumes(stage):
    # Support both legacy and new schema names across Isaac Sim versions
    valid_types = {"NavMeshVolume", "NavMeshIncludeVolume"}
    return [prim for prim in stage.Traverse() if prim.GetTypeName() in valid_types]


def _range3d_union(a: Gf.Range3d, b: Gf.Range3d) -> Gf.Range3d:
    """Return the union of two Gf.Range3d in a way compatible with Isaac Sim 5.0 bindings."""
    if a is None:
        return b
    # Try API union first (present in some builds)
    try:
        return a.Union(b)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Manual union via min/max
    try:
        a_min = a.GetMin(); a_max = a.GetMax()
        b_min = b.GetMin(); b_max = b.GetMax()
    except Exception:
        # Fallback to attribute access
        a_min = getattr(a, "min", Gf.Vec3d(0))
        a_max = getattr(a, "max", Gf.Vec3d(0))
        b_min = getattr(b, "min", Gf.Vec3d(0))
        b_max = getattr(b, "max", Gf.Vec3d(0))

    mn = Gf.Vec3d(min(a_min[0], b_min[0]), min(a_min[1], b_min[1]), min(a_min[2], b_min[2]))
    mx = Gf.Vec3d(max(a_max[0], b_max[0]), max(a_max[1], b_max[1]), max(a_max[2], b_max[2]))
    return Gf.Range3d(mn, mx)


def _range3d_min_max(r: Gf.Range3d):
    """Return (min, max) as Gf.Vec3d for a Range3d in a 5.0‑compatible way."""
    try:
        return r.GetMin(), r.GetMax()
    except Exception:
        # Older bindings may expose attributes directly
        return getattr(r, "min", Gf.Vec3d(0)), getattr(r, "max", Gf.Vec3d(0))


def ensure_navmesh_volume(
    root_prim_path: str,
    z_padding: float = 2.0,
    include_volume_parent: Optional[str] = None,
    min_xy: Optional[float] = None,
    min_z: Optional[float] = None,
):
    """Ensure that at least one NavMesh include volume exists around the root prim."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        carb.log_error("MP3D import failed. See navmesh_utils.py.")
        return []

    existing = _list_navmesh_volumes(stage)
    if existing:
        return existing

    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim or not root_prim.IsValid():
        carb.log_warn(f"Root prim '{root_prim_path}' not found; skip NavMesh volume creation.")
        return []

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], useExtentsHint=True)
    world_aabox = None
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Imageable):
            try:
                aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            except Exception as e:
                carb.log_warn(f"[MP] Failed to compute bbox for prim {prim.GetPath()}: {e}")
                continue
            world_aabox = _range3d_union(world_aabox, aligned)

    # If a hidden ground plane exists, include it into the union so the volume surely encloses it
    try:
        from pxr import UsdGeom as _UsdGeom
        gp_prim = stage.GetPrimAtPath("/World/GroundPlane/Plane")
        if gp_prim and gp_prim.IsValid():
            bbox_cache2 = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], useExtentsHint=True)
            try:
                gp_aligned = bbox_cache2.ComputeWorldBound(gp_prim).ComputeAlignedBox()
                world_aabox = gp_aligned if world_aabox is None else _range3d_union(world_aabox, gp_aligned)
                carb.log_info("[MP] Included GroundPlane bbox into union for NavMeshVolume")
            except Exception:
                pass
    except Exception:
        pass

    if world_aabox is None:
        carb.log_warn("Failed to compute world bounding box for NavMesh volume generation.")
        return []

    mn, mx = _range3d_min_max(world_aabox)
    center = Gf.Vec3d((mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5)
    size = Gf.Vec3d(abs(mx[0] - mn[0]), abs(mx[1] - mn[1]), abs(mx[2] - mn[2]))
    # Apply minimal size to avoid degenerate volumes
    if min_xy is None:
        min_xy = 10.0
    if min_z is None:
        min_z = 3.0
    size = Gf.Vec3d(max(size[0], float(min_xy)), max(size[1], float(min_xy)), max(size[2], float(min_z)))
    carb.log_info(f"[MP] NavMeshVolume bbox center={center} size={size}")
    pad_z = max(float(z_padding), float(size[2]) * 0.1)
    adjusted_size = Gf.Vec3d(float(size[0]), float(size[1]), float(size[2]) + pad_z)

    # Determine parent path for navmesh volumes and ensure it exists
    if not include_volume_parent:
        include_volume_parent = "/World/NavMesh"
    if not include_volume_parent.startswith("/"):
        include_volume_parent = "/" + include_volume_parent
    try:
        stage.DefinePrim(include_volume_parent, "Xform")
    except Exception:
        pass

    # Create an include volume via the official command (ensures correct schema)
    omni.kit.commands.execute(
        "CreateNavMeshVolumeCommand",
        parent_prim_path=include_volume_parent,
        volume_type=0,  # include
        layer=stage.GetRootLayer(),
    )

    volumes = _list_navmesh_volumes(stage)
    if not volumes:
        carb.log_error("NavMesh volume creation command did not produce a volume.")
        return []
    else:
        try:
            found = ", ".join([f"{p.GetPath()}[{p.GetTypeName()}]" for p in volumes])
            carb.log_info(f"[MP] Found NavMesh volumes after create: {found}")
        except Exception:
            pass

    vol_prim = volumes[-1]

    # If volume wasn't parented under expected path, move it
    expected_prefix = include_volume_parent.rstrip("/") + "/"
    vol_path = str(vol_prim.GetPath())
    if not vol_path.startswith(expected_prefix):
        navmesh_parent = include_volume_parent
        if not stage.GetPrimAtPath(navmesh_parent).IsValid():
            stage.DefinePrim(navmesh_parent, "Xform")
        # Choose a destination path that doesn't collide
        base_name = vol_prim.GetName() or "NavMeshVolume"
        dst_path = Sdf.Path(navmesh_parent).AppendChild(base_name)
        if stage.GetPrimAtPath(str(dst_path)).IsValid():
            i = 1
            while stage.GetPrimAtPath(f"{navmesh_parent}/{base_name}_{i}").IsValid():
                i += 1
            dst_path = Sdf.Path(f"{navmesh_parent}/{base_name}_{i}")
        moved = False
        # Try a couple of known commands to move the prim across USD/Kit versions
        executed_cmd = None
        for cmd_name, kwargs in [
            ("MovePrim", {"path_from": vol_prim.GetPath(), "path_to": str(dst_path)}),
            ("MovePrimCommand", {"path_from": vol_prim.GetPath(), "path_to": str(dst_path)}),
            ("MovePrims", {"paths": [str(vol_prim.GetPath())], "new_parent_path": navmesh_parent}),
            ("MovePrimsCommand", {"paths": [str(vol_prim.GetPath())], "new_parent_path": navmesh_parent}),
            ("RenamePrim", {"old_path": vol_prim.GetPath(), "new_path": str(dst_path)}),
            ("RenamePrimCommand", {"old_path": vol_prim.GetPath(), "new_path": str(dst_path)}),
        ]:
            try:
                omni.kit.commands.execute(cmd_name, **kwargs)
                moved = True
                executed_cmd = cmd_name
                break
            except Exception:
                continue
        if not moved:
            # Fallback A: copy on the same layer and remove the original.
            try:
                Sdf.CopySpec(stage.GetRootLayer(), vol_prim.GetPath(), dst_path)
                stage.RemovePrim(vol_prim.GetPath())
                moved = True
            except Exception:
                moved = False
        if not moved:
            # Fallback B: define a NavMeshIncludeVolume directly under /World/NavMesh,
            # reusing the center/size computed above.
            try:
                # ``center`` and ``adjusted_size`` were computed earlier in this function.
                tmp_name = "IncludeVolume"
                dst2 = Sdf.Path(f"{navmesh_parent}/{tmp_name}")
                idx = 1
                while stage.GetPrimAtPath(str(dst2)).IsValid():
                    dst2 = Sdf.Path(f"{navmesh_parent}/{tmp_name}_{idx}"); 
                    idx += 1
                
                include_prim = stage.DefinePrim(str(dst2), "NavMeshIncludeVolume")
                
                if not include_prim or not include_prim.IsValid():
                    carb.log_error(f"[MP] Failed to create NavMeshIncludeVolume at {dst2}")
                    moved = False
                else:
                    xform = UsdGeom.Xformable(include_prim)
                    xform.ClearXformOpOrder()
                    xform.AddTranslateOp().Set(center)
                    xform.AddScaleOp().Set(adjusted_size)
                    # Conservatively remove the volume that was created at the wrong path.
                    try:
                        stage.RemovePrim(vol_prim.GetPath())
                    except Exception:
                        pass
                    moved = True
                    carb.log_info(f"[MP] Created include volume at {vol_prim.GetPath()} (fallback B).")
            except Exception:
                pass
        # Re-resolve the moved prim (MovePrim/MovePrims/Rename may yield different paths).
        if moved:
            new_candidates = []
            # Try the explicit destination path first.
            try:
                new_candidates.append(str(dst_path))
            except Exception:
                pass
            # Then try the original name appended to the new parent (covers the MovePrims family).
            try:
                base_name = base_name or "NavMeshVolume"
                new_candidates.append(f"{navmesh_parent}/{base_name}")
            except Exception:
                pass
            # As a final fallback, rescan the stage and prefer volumes under /World/NavMesh.
            for path in new_candidates:
                p = stage.GetPrimAtPath(path)
                if p and p.IsValid():
                    vol_prim = p
                    break
            else:
                vols_after = _list_navmesh_volumes(stage)
                for p in vols_after[::-1]:
                    if str(p.GetPath()).startswith(f"{navmesh_parent}/"):
                        vol_prim = p
                        break
    # Defensive: if the prim is still invalid, bail out before touching the schema.
    if not vol_prim or not vol_prim.IsValid():
        carb.log_error("NavMesh volume prim is invalid after creation/move; skip transform setup.")
        return volumes
    xformable = UsdGeom.Xformable(vol_prim)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp()
    scale_op = xformable.AddScaleOp()
    translate_op.Set(Gf.Vec3d(center[0], center[1], center[2]))
    scale_op.Set(adjusted_size)

    carb.log_info(
        f"NavMesh include volume created at {vol_prim.GetPath()} with center={center} size={adjusted_size}."
    )
    return volumes


async def ensure_navmesh_async(
    root_prim_path: str,
    z_padding: float = 2.0,
    status_callback=None,
    include_volume_parent: Optional[str] = None,
    min_xy: Optional[float] = None,
    min_z: Optional[float] = None,
    agent_radius: Optional[float] = None,
    max_step_height: Optional[float] = None,
):
    """Async variant: create the NavMesh volume and trigger baking on a background thread,
    avoiding event-loop re-entrancy issues on Isaac Sim 5.0.

    Steps:
    - compute / create / move the volume on the main thread;
    - wait one or two update frames so USD and the navigation system register the volume;
    - call the synchronous ``baking_and_wait`` via ``run_in_executor`` so the caller's
      asyncio task is not blocked;
    - if baking fails, enlarge the volume once and retry.

    Args:
        root_prim_path: Root prim path for NavMesh volume creation.
        z_padding: Z-axis padding for the NavMesh volume.
        status_callback: Optional callback that receives status updates.
        agent_radius: Agent radius in meters.
        max_step_height: Maximum step height the agent can climb, in meters.
    """

    # Report baking start
    if status_callback:
        status_callback("baking")

    volumes = ensure_navmesh_volume(
        root_prim_path,
        z_padding,
        include_volume_parent=include_volume_parent,
        min_xy=min_xy,
        min_z=min_z,
    )
    if not volumes:
        carb.log_warn("NavMesh baking skipped because no NavMeshVolume is available.")
        if status_callback:
            status_callback("failed")
        return None

    # Pump a few frames so the volume changes are consumed.
    try:
        import omni.kit.app as _kit_app
        _app = _kit_app.get_app()
        await _app.next_update_async()
        await _app.next_update_async()
    except Exception:
        carb.log_error("NavMesh baking consumed failed.")
        pass

    interface = nav.acquire_interface()

    # Set NavMesh configuration before baking
    from OmniNavExt.envset.core.scene_manager import SceneManager
    radius = agent_radius if agent_radius is not None else 10.0
    SceneManager._set_navmesh_config(agent_radius=radius, max_step_height=max_step_height)

    loop = asyncio.get_event_loop()

    def _bake_blocking():
        interface.start_navmesh_baking_and_wait()
        return interface.get_navmesh()

    carb.log_info("[MP] NavMesh baking attempt #1 (async)")
    navmesh = await loop.run_in_executor(None, _bake_blocking)
    if navmesh is None:
        # Enlarge the volume once on the main thread and retry.
        stage = omni.usd.get_context().get_stage()
        vols = _list_navmesh_volumes(stage)
        if vols:
            try:
                x = UsdGeom.Xformable(vols[-1])
                ops = x.GetOrderedXformOps()
                for op in ops:
                    if op.GetOpName().lower().endswith("scale"):
                        cur = Gf.Vec3d(op.Get())
                        op.Set(Gf.Vec3d(cur[0] * 1.5, cur[1] * 1.5, cur[2] * 1.5))
                        break
            except Exception:
                pass
            try:
                import omni.kit.app as _kit_app
                _app = _kit_app.get_app()
                await _app.next_update_async()
            except Exception:
                pass
            carb.log_info("[MP] NavMesh baking attempt #2 (async, enlarged)")
            navmesh = await loop.run_in_executor(None, _bake_blocking)

    if navmesh is None:
        carb.log_error(
            "NavMesh building failed. Ensure the NavMeshVolume encloses walkable geometry and navigation extensions are enabled."
        )
        if status_callback:
            status_callback("failed")
        return None

    carb.log_info("[MP] NavMesh baking completed successfully")
    if status_callback:
        status_callback("ready")
    return navmesh
