"""
Core modules extracted from standalone.py for reusability.

Modules:
- simulation_bootstrap: SimulationApp initialization
- physics_manager: Physics fixes and articulation handling
- navmesh_manager: NavMesh baking
- scene_manager: Scene loading and Matterport import
- lifecycle: Simulation lifecycle management
"""

from .simulation_bootstrap import SimulationBootstrap, SimulationConfig
from .physics_manager import PhysicsManager
from .navmesh_manager import NavMeshManager
from .scene_manager import (
    SceneManager,
    map_to_stage_path,
    find_scene_root,
    is_matterport_scenario,
    should_use_camera_light,
)
from .lifecycle import SimulationLifecycle

__all__ = [
    "SimulationBootstrap",
    "SimulationConfig",
    "PhysicsManager",
    "NavMeshManager",
    "SceneManager",
    "SimulationLifecycle",
    # Utility functions
    "map_to_stage_path",
    "find_scene_root",
    "is_matterport_scenario",
    "should_use_camera_light",
]
