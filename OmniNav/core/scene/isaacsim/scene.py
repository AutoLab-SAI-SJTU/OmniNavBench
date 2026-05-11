from typing import List

from OmniNav.core.config import TaskCfg
from OmniNav.core.robot.rigid_body import IRigidBody
from OmniNav.core.scene import validate_scene_file
from OmniNav.core.scene.scene import IScene


class IsaacsimScene(IScene):
    """IsaacSim's implementation on `IScene` class."""

    def __init__(self):
        from omni.isaac.core import World
        from omni.isaac.core.scenes import Scene

        self._scene: Scene = World.instance().scene

    def load(self, task_config: TaskCfg, env_id: int, env_offset: List[float]):
        """See `IScene.load` for documentation."""
        import logging
        log = logging.getLogger(__name__)

        # Paths are normalized via envset scene_root to avoid cwd fallback.
        usd_path = str(task_config.scene_asset_path)
        prim_path_root = f'World/env_{env_id}/scene'
        source, prim_path = validate_scene_file(usd_path, prim_path_root)

        from omni.isaac.core.utils.prims import create_prim, is_prim_path_valid, get_prim_at_path

        # Check if scene prim already exists (for scene reuse)
        if is_prim_path_valid(prim_path):
            log.info(f"[SceneReuse] Scene already exists at {prim_path}, skipping load")
            self.scene_prim = get_prim_at_path(prim_path)
            return

        position = [env_offset[idx] + i for idx, i in enumerate(task_config.scene_position)]

        # When scene_units_in_meters is set, auto-scale; otherwise fall back to manual scene_scale.
        if task_config.scene_units_in_meters is not None:
            log.info(f"[SceneLoad] Loading scene with auto-scaling: units_in_meters={task_config.scene_units_in_meters}")

            # Auto-scale path: create the prim without a scale, then normalize via apply_scene_unit_scale.
            scene_prim = create_prim(prim_path, usd_path=source, translation=position)
            self.scene_prim = scene_prim

            import omni.usd
            from OmniNavExt.envset.scene_scale_utils import apply_scene_unit_scale

            stage = omni.usd.get_context().get_stage()
            apply_scene_unit_scale(
                stage=stage,
                scene_prim_path=prim_path,
                scene_units_in_meters=task_config.scene_units_in_meters,
                stage_meters_per_unit=1.0,
            )
        else:
            log.info(f"[SceneLoad] Loading scene with manual scale: scene_scale={task_config.scene_scale}")
            scene_prim = create_prim(prim_path, usd_path=source, scale=task_config.scene_scale, translation=position)
            self.scene_prim = scene_prim

    def add(self, target: any):
        """See `IScene.add` for documentation."""
        if hasattr(target, 'initialize') and hasattr(target, 'unwrap'):
            # TODO: Implement initialize method on IArticulation._articulation to make
            # 'self._scene._scene_registry.add_articulated_system' -> 'self._scene.add'
            self._scene._scene_registry.add_articulated_system(name=target.name, articulated_system=target)
        elif hasattr(target, 'unwrap'):
            self._scene.add(target.unwrap())
        else:
            # For instance of isaac-sim native classes
            self._scene.add(target)

    def remove(self, target: any, registry_only: bool = False):
        """See `IScene.remove` for documentation."""
        self._scene.remove_object(name=target, registry_only=registry_only)

    def object_exists(self, target: any) -> bool:
        """See `IScene.object_exists` for documentation."""
        return self._scene.object_exists(target)

    def get(self, target: any) -> IRigidBody:
        """See `IScene.get` for documentation."""
        object = self._scene.get_object(target)
        return IRigidBody.create(prim_path=object.prim_path, name=object.prim_path)

    def unwrap(self):
        """See `IScene.unwrap` for documentation."""
        return self._scene
