"""NavMesh baking management - wraps existing navmesh_utils."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from isaacsim import SimulationApp

from .scene_manager import map_to_stage_path


@dataclass
class NavMeshConfig:
    """NavMesh configuration extracted from scenario."""
    root_prim_path: str
    include_volume_parent: str = "/World/NavMesh"
    z_padding: float = 2.0
    agent_radius: float = 10.0
    max_step_height: Optional[float] = None
    min_xy: Optional[float] = None
    min_z: Optional[float] = None


class NavMeshManager:
    """Manages NavMesh volume creation and baking."""

    _navmesh_ready: bool = False  # Global flag for navmesh status

    def __init__(self, config: NavMeshConfig):
        self._config = config

    @classmethod
    def from_scenario(
        cls,
        navmesh_cfg: Dict[str, Any],
        scene_cfg: Dict[str, Any],
        actual_scene_root: Optional[str] = None,
    ) -> "NavMeshManager":
        """Create NavMeshManager from scenario configuration.

        Args:
            navmesh_cfg: navmesh section from scenario
            scene_cfg: scene section from scenario
            actual_scene_root: Resolved scene root path
        """
        # Resolve root path
        configured_bake_root = navmesh_cfg.get("bake_root_prim_path")
        configured_scene_root = scene_cfg.get("root_prim_path")
        navmesh_root_override = (
            navmesh_cfg.get("navmesh_root_prim_path") or scene_cfg.get("navmesh_root_prim_path")
        )

        # Resolve root path with proper path mapping
        # Priority: configured_bake_root > navmesh_root_override > actual_scene_root > /World
        root_path = None

        # Try configured_bake_root first (may be relative path)
        if configured_bake_root:
            root_path = map_to_stage_path(
                configured_bake_root,
                actual_scene_root,
                configured_scene_root
            )

        # Fallback to navmesh_root_override
        if not root_path and navmesh_root_override:
            root_path = map_to_stage_path(
                navmesh_root_override,
                actual_scene_root,
                configured_scene_root
            )

        # Final fallback to scene roots
        if not root_path:
            root_path = actual_scene_root or configured_scene_root or "/World"

        # Extract other params
        min_size = navmesh_cfg.get("min_include_volume_size") or {}

        config = NavMeshConfig(
            root_prim_path=root_path,
            include_volume_parent=navmesh_cfg.get("include_volume_parent") or "/World/NavMesh",
            z_padding=navmesh_cfg.get("z_padding") or 2.0,
            agent_radius=navmesh_cfg.get("agent_radius") or 10.0,
            max_step_height=navmesh_cfg.get("max_step_height"),
            min_xy=navmesh_cfg.get("min_include_xy") or min_size.get("xy"),
            min_z=navmesh_cfg.get("min_include_z") or min_size.get("z"),
        )
        return cls(config)

    def bake_sync(self, simulation_app: "SimulationApp", envset_cfg: Optional[Dict[str, Any]] = None) -> bool:
        """Synchronous NavMesh baking.

        Args:
            simulation_app: SimulationApp for frame updates

        Returns:
            True if baking succeeded
        """
        import carb  # type: ignore
        import omni.anim.navigation.core as nav  # type: ignore

        from OmniNavExt.envset.navmesh_utils import ensure_navmesh_volume

        carb.log_info(f"[NavMeshManager] Baking NavMesh at root: {self._config.root_prim_path}")

        # Create NavMesh volume
        volumes = ensure_navmesh_volume(
            root_prim_path=self._config.root_prim_path,
            z_padding=self._config.z_padding,
            include_volume_parent=self._config.include_volume_parent,
            min_xy=self._config.min_xy,
            min_z=self._config.min_z,
        )

        if not volumes:
            raise RuntimeError("Failed to create NavMesh volume")

        # Wait a few frames for volume registration
        for _ in range(3):
            simulation_app.update()

        # Set NavMesh configuration
        self._set_navmesh_config(self._config.agent_radius, self._config.max_step_height)

        # Bake NavMesh (blocking)
        interface = nav.acquire_interface()
        carb.log_info("[NavMeshManager] Starting NavMesh baking...")
        interface.start_navmesh_baking_and_wait()

        navmesh = interface.get_navmesh()
        if navmesh is None:
            raise RuntimeError("NavMesh baking failed - no navmesh returned")

        carb.log_info("[NavMeshManager] NavMesh baking completed successfully")
        NavMeshManager._navmesh_ready = True
        # Sync flag to EnvsetTaskRuntime (consistent with SceneManager.prepare_matterport_navmesh)
        from OmniNavExt.envset.runtime_hooks import EnvsetTaskRuntime
        if envset_cfg:
            EnvsetTaskRuntime.mark_navmesh_ready(envset_cfg)
        else:
            EnvsetTaskRuntime._navmesh_ready = True
        return True

    @staticmethod
    def _set_navmesh_config(agent_radius: float, max_step_height: Optional[float] = None):
        """Set NavMesh configuration parameters.
        
        Args:
            agent_radius: Agent radius in meters
            max_step_height: Maximum step height agent can climb in meters
            
        Raises:
            RuntimeError: If setting configuration fails
        """
        from OmniNavExt.envset.core.scene_manager import SceneManager
        SceneManager._set_navmesh_config(agent_radius=agent_radius, max_step_height=max_step_height)

    @classmethod
    def is_ready(cls) -> bool:
        """Check if NavMesh is ready."""
        return cls._navmesh_ready

    @classmethod
    def reset_ready_flag(cls):
        """Reset the ready flag (for new scenario)."""
        cls._navmesh_ready = False
