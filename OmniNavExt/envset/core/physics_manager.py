"""Physics property management and articulation initialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from OmniNav.core.runner import SimulatorRunner


def is_grscenes(scene_cfg: Dict[str, Any]) -> bool:
    """Detect if scene is GRScenes type."""
    # Check category field
    category = scene_cfg.get("category")
    if category:
        category_str = str(category).strip().lower()
        if "grscenes" in category_str:
            return True

    # Check usd_path
    usd_path = scene_cfg.get("usd_path") or ""
    if isinstance(usd_path, str) and "grscene" in usd_path.lower():
        return True

    # Check navmesh_root_prim_path
    navmesh_root = scene_cfg.get("navmesh_root_prim_path")
    if navmesh_root and str(navmesh_root).startswith("/Root"):
        return True

    return False


class PhysicsManager:
    """Manages physics properties and articulation initialization."""

    @staticmethod
    def fix_grscenes_physics(scene_cfg: Dict[str, Any], scene_root: Optional[str] = None):
        """Remove RigidBodyAPI from static objects in GRScenes to prevent falling.

        Args:
            scene_cfg: Scene configuration dict
            scene_root: Scene root prim path (e.g., '/World/env_0/scene')
        """
        if not is_grscenes(scene_cfg):
            print("[PhysicsManager] No physics fixes needed for this scene type")
            return

        import omni.usd  # type: ignore
        from pxr import Usd, UsdPhysics  # type: ignore

        print("[PhysicsManager] GRScenes detected, removing RigidBodyAPI from static objects...")

        stage = omni.usd.get_context().get_stage()
        if not stage:
            raise RuntimeError("Stage is invalid, cannot fix physics properties")

        # Find scene root if not provided
        if not scene_root:
            scene_root = _find_scene_root_fallback(stage)

        if not scene_root:
            raise RuntimeError("Could not find scene root for physics fixes")

        print(f"[PhysicsManager] Scene root: {scene_root}")

        root_prim = stage.GetPrimAtPath(scene_root)
        if not root_prim or not root_prim.IsValid():
            raise RuntimeError(f"Scene root prim is invalid: {scene_root}")

        removed_count = 0
        skipped_count = 0

        for prim in Usd.PrimRange(root_prim):
            if not prim.IsValid() or not prim.IsActive():
                continue

            # Skip articulations (have joints) - they need physics
            if _is_articulation(prim):
                skipped_count += 1
                continue

            # Remove RigidBodyAPI from static objects
            try:
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    removed_count += 1

                if prim.HasAttribute("physics:rigidBodyEnabled"):
                    prim.RemoveProperty("physics:rigidBodyEnabled")
            except Exception as e:
                print(f"[PhysicsManager] Warning: Failed to remove RigidBodyAPI from {prim.GetPath()}: {e}")

        print(f"[PhysicsManager] Physics fixes completed: {removed_count} RigidBodyAPI removed, "
              f"{skipped_count} articulations kept physics")

    @staticmethod
    def wait_for_articulations(runner: "SimulatorRunner", max_frames: int = 100) -> bool:
        """Wait for all articulations to initialize.

        Args:
            runner: SimulatorRunner instance
            max_frames: Maximum frames to wait

        Returns:
            True if all articulations initialized, False on timeout
        """
        import carb  # type: ignore
        from omni.isaac.core.simulation_context import SimulationContext  # type: ignore

        world = getattr(runner, "_world", None)
        if not world:
            raise RuntimeError("World not available, cannot wait for articulations")

        carb.log_info("[PhysicsManager] Waiting for articulations to initialize...")

        for frame_idx in range(max_frames):
            SimulationContext.render(world)

            uninitialized = _get_uninitialized_robots(runner)

            if not uninitialized:
                carb.log_info(f"[PhysicsManager] All articulations initialized after {frame_idx + 1} frames")
                return True

            # Log progress every 10 frames
            if frame_idx % 10 == 0:
                robots_str = ", ".join(uninitialized[:5])
                if len(uninitialized) > 5:
                    robots_str += "..."
                carb.log_info(f"[PhysicsManager] Waiting ({frame_idx + 1}/{max_frames}): {robots_str}")

        carb.log_warn(f"[PhysicsManager] Timeout after {max_frames} frames")
        return False

    @staticmethod
    def are_articulations_ready(runner: "SimulatorRunner") -> bool:
        """Quick check if all articulations are initialized."""
        return len(_get_uninitialized_robots(runner)) == 0


def _find_scene_root_fallback(stage) -> Optional[str]:
    """Fallback scene root detection using env_N convention."""
    for env_id in range(10):
        candidate = f"/World/env_{env_id}/scene"
        prim = stage.GetPrimAtPath(candidate)
        if prim and prim.IsValid():
            return candidate
    return None


def _is_articulation(prim) -> bool:
    """Check if prim is part of an articulation (has physics joints)."""
    from pxr import Usd  # type: ignore

    try:
        joint_types = {"PhysicsJoint", "PhysicsRevoluteJoint",
                       "PhysicsPrismaticJoint", "PhysicsFixedJoint"}
        for child in Usd.PrimRange(prim):
            if child.GetTypeName() in joint_types:
                return True
    except Exception:
        pass
    return False


def _get_uninitialized_robots(runner: "SimulatorRunner") -> List[str]:
    """Get list of uninitialized robot names."""
    uninitialized = []

    for task_name, task in runner.current_tasks.items():
        robots = getattr(task, "robots", None)
        if not robots:
            continue

        for robot_name, robot in robots.items():
            articulation = getattr(robot, "articulation", None)
            if articulation is None:
                continue

            handles_initialized = getattr(articulation, "handles_initialized", True)
            if not handles_initialized:
                uninitialized.append(f"{task_name}/{robot_name}")

    return uninitialized
