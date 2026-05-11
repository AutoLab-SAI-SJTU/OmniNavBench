def clear_stage_by_prim_path(prim_path: str = None) -> None:
    """Deletes all prims in the stage without populating the undo command buffer

    Args:
        prim_path (str, optional): path of the stage. Defaults to None.
    """
    if not prim_path:
        return

    # Use USD traversal to avoid invalid prim handles during deletion.
    import omni.usd
    from pxr import Sdf, Usd
    from omni.usd.commands import DeletePrimsCommand

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim or not root_prim.IsValid():
        return

    prim_paths = []
    for prim in Usd.PrimRange(root_prim):
        if not prim or not prim.IsValid():
            continue
        path_str = str(prim.GetPath())
        if not path_str:
            continue
        if path_str in {"/", "/World"}:
            continue
        if path_str.startswith("/Render"):
            continue
        try:
            if prim.GetTypeName() == "PhysicsScene":
                continue
        except Exception:
            pass
        prim_paths.append(path_str)

    if not prim_paths:
        return

    # Delete children first to avoid invalidating parents mid-walk.
    prim_paths = sorted(set(prim_paths), key=len, reverse=True)
    prim_paths_to_delete = [Sdf.Path(p) for p in prim_paths if Sdf.Path.IsValidPathString(p)]
    if prim_paths_to_delete:
        DeletePrimsCommand(prim_paths_to_delete).do()
