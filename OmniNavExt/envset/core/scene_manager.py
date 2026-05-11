"""Scene loading and path resolution utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional


def is_matterport_scenario(scene_cfg: Dict[str, Any]) -> bool:
    """Detect if scene is Matterport/MP3D type."""
    try:
        category = str(scene_cfg.get("category") or "").lower()
        if "mp3d" in category or "matterport" in category:
            return True
    except Exception:
        pass
    return bool(scene_cfg.get("use_matterport"))


def should_use_camera_light(scene_cfg: Dict[str, Any]) -> bool:
    """Return True when the scene should switch viewport lighting to camera light.

    Rules:
    - Matterport scenes keep the existing behavior and always enable camera light.
    - Other scenes can opt in explicitly via:
      - scene.use_camera_light = true
      - scene.camera_light = true
      - scene.viewport_lighting = "camera"
      - scene.lighting_mode = "camera"
    """
    if is_matterport_scenario(scene_cfg):
        return True

    mode = str(
        scene_cfg.get("viewport_lighting")
        or scene_cfg.get("lighting_mode")
        or ""
    ).strip().lower()
    if mode == "camera":
        return True

    return bool(scene_cfg.get("use_camera_light") or scene_cfg.get("camera_light"))


def find_scene_root(stage, scenario: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Find the actual scene root prim path in the stage.

    Args:
        stage: USD stage
        scenario: Optional scenario dict with id field

    Returns:
        Scene root path (e.g., '/World/scene') or None
    """
    if stage is None:
        return None

    # Try scenario-based paths first
    scenario_id = None
    if scenario:
        scenario_id = scenario.get("id") or scenario.get("scenario_id")

    if scenario_id:
        # 1) Standard /World/scene layout
        candidate = "/World/scene"
        prim = stage.GetPrimAtPath(candidate)
        if prim and prim.IsValid():
            return candidate

        # 2) Search under /World for 'scene' child
        world_prim = stage.GetPrimAtPath("/World")
        if world_prim and world_prim.IsValid():
            for child in world_prim.GetChildren():
                if child.GetName() == "scene" and child.IsValid():
                    return str(child.GetPath())

    # 3) Fallback: env_N convention
    for env_id in range(10):
        candidate = f"/World/env_{env_id}/scene"
        prim = stage.GetPrimAtPath(candidate)
        if prim and prim.IsValid():
            return candidate

    return None


def map_to_stage_path(
    target_path: Optional[str],
    actual_scene_root: Optional[str],
    configured_scene_root: Optional[str],
) -> Optional[str]:
    """Map a configured prim path to the actual stage hierarchy.

    Args:
        target_path: Path from configuration
        actual_scene_root: Resolved runtime scene root
        configured_scene_root: Scene root from config

    Returns:
        Mapped path or original path
    """
    if not target_path:
        return None

    try:
        target_path = str(target_path)
    except Exception:
        return None

    # Case A: Relative path -> attach to actual scene root
    if not target_path.startswith("/"):
        if actual_scene_root:
            return f"{actual_scene_root}/{target_path.lstrip('/')}"
        return f"/World/{target_path.lstrip('/')}"

    # Case B: Path starts with configured root -> replace with actual root
    if actual_scene_root and configured_scene_root and target_path.startswith(configured_scene_root):
        relative = target_path[len(configured_scene_root):].lstrip("/")
        return f"{actual_scene_root}/{relative}" if relative else actual_scene_root

    # Case C: Legacy config compatibility (e.g., /World/Root/...)
    if actual_scene_root and configured_scene_root and configured_scene_root != "/" and target_path.startswith("/"):
        configured_parts = configured_scene_root.strip("/").split("/")
        target_parts = target_path.strip("/").split("/")
        if configured_parts and target_parts and configured_parts[0] == target_parts[0]:
            relative = "/".join(target_parts[1:])
            return f"{actual_scene_root}/{relative}" if relative else actual_scene_root

    return target_path


class SceneManager:
    """Manages scene loading and configuration."""

    def __init__(self, scene_cfg: Dict[str, Any], scenario: Optional[Dict[str, Any]] = None):
        self._scene_cfg = scene_cfg
        self._scenario = scenario

    @property
    def is_matterport(self) -> bool:
        return is_matterport_scenario(self._scene_cfg)

    def find_root(self, stage) -> Optional[str]:
        """Find scene root in the given stage."""
        return find_scene_root(stage, self._scenario)

    def map_path(self, target_path: str, stage) -> Optional[str]:
        """Map a configured path to stage path."""
        actual_root = self.find_root(stage)
        configured_root = self._scene_cfg.get("root_prim_path")
        return map_to_stage_path(target_path, actual_root, configured_root)

    @staticmethod
    def normalize_matterport_container(root_prim_path: str) -> str:
        """Normalize Matterport container path."""
        if not root_prim_path:
            return "/World"
        path = str(root_prim_path)
        if not path.startswith("/"):
            path = "/" + path
        if path.endswith("/Matterport"):
            parent = path.rsplit("/", 1)[0]
            return parent if parent else "/World"
        return path if path != "/" else "/World"

    @staticmethod
    def _set_navmesh_config(agent_radius: Optional[float] = None, max_step_height: Optional[float] = None):
        """Set NavMesh configuration parameters.
        
        Args:
            agent_radius: Agent radius in meters
            max_step_height: Maximum step height agent can climb in meters
            
        Raises:
            RuntimeError: If setting configuration fails
        """
        import omni.kit.commands  # type: ignore
        
        if agent_radius is not None:
            try:
                omni.kit.commands.execute(
                    "ChangeSetting",
                    path="/exts/omni.anim.navigation.core/navMesh/config/agentRadius",
                    value=float(agent_radius),
                )
                # Set cell size to 0.1 meters, for making sure the NavMesh is not too dense， so the simulation is not too slow.
                cell_size = 0.1
                omni.kit.commands.execute(
                    "ChangeSetting",
                    path="/exts/omni.anim.navigation.core/navMesh/config/cellSize",
                    value=cell_size,
                )
                omni.kit.commands.execute(
                    "ChangeSetting",
                    path="/exts/omni.anim.navigation.core/navMesh/config/cellHeight",
                    value=cell_size,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to set NavMesh agentRadius: {e}") from e
        
        if max_step_height is not None:
            try:
                omni.kit.commands.execute(
                    "ChangeSetting",
                    path="/exts/omni.anim.navigation.core/navMesh/config/maxStepHeight",
                    value=float(max_step_height),
                )
            except Exception as e:
                raise RuntimeError(f"Failed to set NavMesh maxStepHeight: {e}") from e

    @staticmethod
    def clear_navmesh_volumes(include_parent: Optional[str] = None) -> int:
        """Remove existing NavMesh volumes.

        Args:
            include_parent: Optional parent path to limit search

        Returns:
            Number of volumes cleared
        """
        import omni.usd  # type: ignore
        from pxr import Usd  # type: ignore

        try:
            stage = omni.usd.get_context().get_stage()
        except Exception:
            return 0

        if stage is None:
            return 0

        cleared = 0
        roots = [stage.GetPrimAtPath(include_parent)] if include_parent else [stage.GetPseudoRoot()]

        for root in roots:
            if not root or not root.IsValid():
                continue
            for prim in Usd.PrimRange(root):
                try:
                    prim_type = prim.GetTypeName()
                except Exception:
                    continue
                if prim_type in {"NavMeshVolume", "NavMeshIncludeVolume"}:
                    try:
                        stage.RemovePrim(prim.GetPath())
                        cleared += 1
                    except Exception:
                        continue

        return cleared

    @staticmethod
    def import_matterport_scene(
        scene_cfg: Dict[str, Any],
        scene_root: Optional[Any] = None,
    ) -> str:
        """Import Matterport scene into the stage.

        Args:
            scene_cfg: Scene configuration dict
            scene_root: Optional scene root for resolving relative paths

        Returns:
            Matterport prim path

        Raises:
            RuntimeError: If importer unavailable or import fails
        """
        # Lazy import Matterport
        try:
            from omni.isaac.matterport.scripts import import_matterport_asset  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Matterport importer unavailable (missing omni.isaac.matterport). "
                "Enable extension before running."
            ) from e

        mp_cfg = scene_cfg.get("matterport") or {}

        # Get import path
        import_path = (
            mp_cfg.get("usd_path")
            or mp_cfg.get("obj_path")
            or scene_cfg.get("usd_path")
            or scene_cfg.get("obj_path")
        )
        if not import_path:
            raise RuntimeError("Matterport import requested but no usd_path/obj_path provided.")

        # Get container path
        container_path = SceneManager.normalize_matterport_container(
            mp_cfg.get("root_prim_path") or scene_cfg.get("root_prim_path") or "/World/terrain/Matterport"
        )

        # Resolve relative paths using explicit scene_root.
        resolve_root = None
        if scene_root:
            try:
                from pathlib import Path
                resolve_root = str(Path(scene_root).expanduser())
            except Exception:
                pass

        # Import scene (no ground plane for Matterport, manage_simulation=False)
        matterport_prim = import_matterport_asset(
            prim_path=container_path,
            input_path=import_path,
            groundplane=False,
            manage_simulation=False,
            resolve_relative_to=resolve_root,
        )

        # Handle async import
        import inspect
        if inspect.isawaitable(matterport_prim):
            import asyncio
            loop = asyncio.get_event_loop()
            matterport_prim = loop.run_until_complete(matterport_prim)

        if not matterport_prim:
            raise RuntimeError("Matterport import failed: importer returned empty prim path.")

        # Extract path string
        if hasattr(matterport_prim, "GetPath"):
            matterport_prim = str(matterport_prim.GetPath())

        return str(matterport_prim)

    @staticmethod
    def prepare_matterport_navmesh(
        matterport_prim_path: str,
        navmesh_cfg: Dict[str, Any],
        scene_cfg: Dict[str, Any],
        simulation_app: Any,
        envset_cfg: Optional[Dict[str, Any]] = None,
    ):
        """Prepare and bake NavMesh for Matterport scene.

        Args:
            matterport_prim_path: Imported Matterport prim path
            navmesh_cfg: NavMesh configuration
            scene_cfg: Scene configuration
            simulation_app: SimulationApp for frame updates

        Raises:
            RuntimeError: If baking fails
        """
        from OmniNavExt.envset.navmesh_utils import ensure_navmesh_volume
        import omni.anim.navigation.core as nav  # type: ignore

        bake_root = navmesh_cfg.get("bake_root_prim_path") or matterport_prim_path
        include_parent = navmesh_cfg.get("include_volume_parent") or "/World/NavMesh"
        min_size = navmesh_cfg.get("min_include_volume_size") or {}
        min_xy = navmesh_cfg.get("min_include_xy") or min_size.get("xy")
        min_z = navmesh_cfg.get("min_include_z") or min_size.get("z")
        agent_radius = navmesh_cfg.get("agent_radius")
        z_padding = navmesh_cfg.get("z_padding")
        units_in_m = scene_cfg.get("units_in_meters")

        # Scale parameters if units specified
        def scale_param(val):
            if val is None or units_in_m is None:
                return val
            try:
                return float(val) * float(units_in_m)
            except Exception:
                return val

        min_xy = scale_param(min_xy)
        min_z = scale_param(min_z)
        agent_radius = scale_param(agent_radius)
        z_padding = scale_param(z_padding)
        max_step_height = scale_param(navmesh_cfg.get("max_step_height"))

        # Clear existing volumes
        SceneManager.clear_navmesh_volumes(include_parent)

        # Create NavMesh volume
        ensure_navmesh_volume(
            root_prim_path=bake_root,
            z_padding=z_padding or 2.0,
            include_volume_parent=include_parent,
            min_xy=min_xy,
            min_z=min_z,
        )

        # Set NavMesh configuration parameters
        SceneManager._set_navmesh_config(agent_radius=agent_radius, max_step_height=max_step_height)

        # Wait for USD updates
        for _ in range(3):
            try:
                simulation_app.update()
            except Exception:
                break

        # Bake NavMesh (blocking)
        interface = nav.acquire_interface()
        interface.start_navmesh_baking_and_wait()
        navmesh = interface.get_navmesh()

        if navmesh is None:
            raise RuntimeError("Matterport NavMesh baking failed (no navmesh returned).")

        # Mark NavMesh ready
        from OmniNavExt.envset.runtime_hooks import EnvsetTaskRuntime
        if envset_cfg:
            EnvsetTaskRuntime.mark_navmesh_ready(envset_cfg)
        else:
            EnvsetTaskRuntime._navmesh_ready = True
