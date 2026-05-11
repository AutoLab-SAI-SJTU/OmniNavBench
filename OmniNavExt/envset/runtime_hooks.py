from __future__ import annotations
import json
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from OmniNavExt.envset.unit_utils import resolve_env_unit_scale, scale_env_position, try_scale_goto_command


def _import_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _require_module(module_name: str, reason: str):
    module = _import_module(module_name)
    if not module:
        raise ModuleNotFoundError(
            f"{module_name} is not available; {reason}. Initialize the simulator before calling."
        )
    return module


def _require_carb(reason: str):
    module = _import_module("carb")
    if not module:
        raise ModuleNotFoundError(
            f"carb is not available; {reason}. Initialize the simulator before calling."
        )
    return module


def _require_omni(module_name: str, reason: str):
    module = _import_module(module_name)
    if not module:
        raise ModuleNotFoundError(
            f"{module_name} is not available; {reason}. Initialize the simulator before calling."
        )
    return module


def _require_pxr(module_name: str, reason: str):
    module = _import_module(module_name)
    if not module:
        raise ModuleNotFoundError(
            f"{module_name} is not available; {reason}. Initialize the simulator before calling."
        )
    return module


def _log_info(message: str):
    carb = _import_module("carb")
    if carb and hasattr(carb, "log_info"):
        carb.log_info(message)
    else:
        print(message)


def _log_warn(message: str):
    carb = _import_module("carb")
    if carb and hasattr(carb, "log_warn"):
        carb.log_warn(message)
    else:
        print(message)


def _log_error(message: str):
    carb = _import_module("carb")
    if carb and hasattr(carb, "log_error"):
        carb.log_error(message)
    else:
        print(message)


# Robot type -> main body link name (used for dynamic-obstacle registration).
# Both ``config.type`` (e.g. 'CarterV1Robot') and ``config.name`` (e.g. 'carter_v1') are accepted.
ROBOT_BODY_LINK_MAP = {
    # config.type form (set on the class).
    'carterv1robot': 'chassis_link',
    'aliengorobot': 'trunk',
    'h1robot': 'torso_link',
    'h1withhandrobot': 'torso_link',
    'g1robot': 'pelvis',
    'gr1robot': 'pelvis',
    'kitt15robot': 'base_link',
    # config.name form (set in envset JSON or on the class).
    'kitt15': 'base_link',
    'carter_v1': 'chassis_link',
    'aliengo': 'trunk',
    'h1': 'torso_link',
    'h1_with_hand': 'torso_link',
    'g1': 'pelvis',
    'gr1': 'pelvis',
}


class EnvsetTaskRuntime:
    _navmesh_ready = False
    _navmesh_ready_by_key: Dict[str, bool] = {}
    _pending_routes: Dict[str, List[str]] = {}
    _route_subscription = None
    _vh_spawned = False
    _arrival_guard = None

    @classmethod
    def _get_arrival_guard(cls):
        if cls._arrival_guard is None:
            ArrivalGuard = _require_module(
                "OmniNavExt.guards.arrival_guard",
                "configuring arrival guard",
            ).ArrivalGuard
            cls._arrival_guard = ArrivalGuard()
        return cls._arrival_guard

    @classmethod
    def _debug_dump_characters(cls, stage, tag: str):
        try:
            if stage is None:
                print(f"[EnvsetRuntime][DEBUG] {tag}: stage is None")
                return
            print(f"[EnvsetRuntime][DEBUG] {tag}: stage id={id(stage)}")
            try:
                edit_target = stage.GetEditTarget().GetLayer().identifier
            except Exception:
                edit_target = None
            print(f"[EnvsetRuntime][DEBUG] {tag}: edit target={edit_target}")
            
            CharacterUtil = _require_module(
                "OmniNavExt.envset.stage_util",
                "debugging character dump",
            ).CharacterUtil
            
            prims = CharacterUtil.get_characters_root_in_stage(count_invisible=True)
            prim_paths = [str(p.GetPrimPath()) for p in prims if p and p.IsValid()]
            print(f"[EnvsetRuntime][DEBUG] {tag}: character roots={prim_paths}")
            for prim in prims:
                if not prim or not prim.IsValid():
                    continue
                try:
                    stack = [l.identifier for l in prim.GetPrimStack()]
                except Exception:
                    stack = []
                print(
                    f"[EnvsetRuntime][DEBUG] {tag}: {prim.GetPrimPath()} "
                    f"active={prim.IsActive()} stack={stack}"
                )
        except Exception as exc:
            print(f"[EnvsetRuntime][DEBUG] {tag}: dump failed: {exc}")

    @classmethod
    def configure_task(cls, task):
        envset_cfg = getattr(task.config, "envset", None)
        if not envset_cfg:
            envset_cfg = getattr(task.config, "__dict__", {}).get("envset")
        if not envset_cfg:
            return
        cls._setup_navmesh(envset_cfg)
        try:
            cls._setup_virtual_routes(envset_cfg)
        except Exception as exc:
            _log_warn(f"[EnvsetRuntime] Routes setup skipped: {exc}")
        try:
            cls._setup_virtual_characters(envset_cfg)
        except Exception as exc:
            _log_warn(f"[EnvsetRuntime] Character setup skipped: {exc}")

    @classmethod
    def _setup_navmesh(cls, envset_cfg):
        """Synchronous: create the NavMesh volume only; baking happens elsewhere."""
        if cls.is_navmesh_ready(envset_cfg):
            print("[EnvsetRuntime] NavMesh already ready, skipping setup")
            return
        
        navmesh_utils = _require_module("OmniNavExt.envset.navmesh_utils", "setting up navmesh")
        ensure_navmesh_volume = navmesh_utils.ensure_navmesh_volume
        
        navmesh_cfg = envset_cfg.get("navmesh") or {}
        if not navmesh_cfg:
            return
        scene_cfg = envset_cfg.get("scene") or {}
        
        # Get the configured bake root path (e.g., "/Root/Meshes/Base/ground")
        configured_bake_root = navmesh_cfg.get("bake_root_prim_path")
        configured_scene_root = scene_cfg.get("root_prim_path")  # e.g., "/Root"
        
        # Resolve the actual scene root path in the loaded stage (always /World/scene).
        omni_usd = _require_omni("omni.usd", "accessing the USD stage for navmesh setup")
        stage = omni_usd.get_context().get_stage()
        actual_scene_root = cls._resolve_scene_root(stage)
        
        # If we found the OmniNav scene path, map the bake root relative to it
        if actual_scene_root and configured_bake_root:
            # Check if bake_root is already a relative path (doesn't start with "/")
            if not configured_bake_root.startswith("/"):
                # It's already a relative path, use it directly
                root_path = f"{actual_scene_root}/{configured_bake_root}"
            elif configured_scene_root and configured_bake_root.startswith(configured_scene_root):
                # Extract the relative path from the configured scene root
                # e.g., "/Root/Meshes/Base/ground" -> "Meshes/Base/ground" (if scene_root="/Root")
                relative_path = configured_bake_root[len(configured_scene_root):].lstrip("/")
                root_path = f"{actual_scene_root}/{relative_path}" if relative_path else actual_scene_root
            else:
                # If bake_root doesn't start with scene_root, try to extract relative part
                # Extract just the last part (e.g., "Meshes/Base/ground" from "/Root/Meshes/Base/ground")
                parts = configured_bake_root.strip("/").split("/")
                if len(parts) > 1:
                    # Try to find the matching subpath under actual_scene_root
                    relative_path = "/".join(parts[1:])  # Skip the root part
                    root_path = f"{actual_scene_root}/{relative_path}"
                else:
                    root_path = configured_bake_root
        elif actual_scene_root:
            # No bake_root configured, use the actual scene root
            root_path = actual_scene_root
        else:
            # Fallback to configured paths or defaults
            root_path = configured_bake_root or configured_scene_root or "/World"
        
        include_parent = navmesh_cfg.get("include_volume_parent") or "/World/NavMesh"
        z_padding = navmesh_cfg.get("z_padding") or 2.0
        # Support both formats: direct fields or nested in min_include_volume_size
        min_size = navmesh_cfg.get("min_include_volume_size") or {}
        min_xy = navmesh_cfg.get("min_include_xy") or min_size.get("xy") or None
        min_z = navmesh_cfg.get("min_include_z") or min_size.get("z") or None
        
        _log_info(f"[EnvsetRuntime] NavMesh bake root resolved: {configured_bake_root} -> {root_path}")
        ensure_navmesh_volume(
            root_prim_path=root_path,
            z_padding=z_padding,
            include_volume_parent=include_parent,
            min_xy=min_xy,
            min_z=min_z,
        )
        # Note: only the volume is created here, not the bake.
        # cls._navmesh_ready is set after bake_navmesh_async succeeds.
        _log_info("[EnvsetRuntime] NavMesh volume created, ready for baking")

    @classmethod
    async def bake_navmesh_async(cls, envset_cfg):
        """Async: actually bake the NavMesh."""
        if cls.is_navmesh_ready(envset_cfg):
            _log_info("[EnvsetRuntime] NavMesh already baked, skipping")
            return True
        
        navmesh_cfg = envset_cfg.get("navmesh") or {}
        if not navmesh_cfg:
            _log_warn("[EnvsetRuntime] No navmesh config, skipping bake")
            return False
        
        scene_cfg = envset_cfg.get("scene") or {}
        configured_bake_root = navmesh_cfg.get("bake_root_prim_path")
        configured_scene_root = scene_cfg.get("root_prim_path")
        
        # Resolve root_path using the same logic as _setup_navmesh (relies on /World/scene).
        omni_usd = _require_omni("omni.usd", "accessing the USD stage for navmesh bake")
        stage = omni_usd.get_context().get_stage()
        actual_scene_root = cls._resolve_scene_root(stage)
        
        if actual_scene_root and configured_bake_root:
            if not configured_bake_root.startswith("/"):
                root_path = f"{actual_scene_root}/{configured_bake_root}"
            elif configured_scene_root and configured_bake_root.startswith(configured_scene_root):
                relative_path = configured_bake_root[len(configured_scene_root):].lstrip("/")
                root_path = f"{actual_scene_root}/{relative_path}" if relative_path else actual_scene_root
            else:
                parts = configured_bake_root.strip("/").split("/")
                if len(parts) > 1:
                    relative_path = "/".join(parts[1:])
                    root_path = f"{actual_scene_root}/{relative_path}"
                else:
                    root_path = configured_bake_root
        elif actual_scene_root:
            root_path = actual_scene_root
        else:
            root_path = configured_bake_root or configured_scene_root or "/World"
        
        include_parent = navmesh_cfg.get("include_volume_parent") or "/World/NavMesh"
        z_padding = navmesh_cfg.get("z_padding") or 2.0
        min_size = navmesh_cfg.get("min_include_volume_size") or {}
        min_xy = navmesh_cfg.get("min_include_xy") or min_size.get("xy") or None
        min_z = navmesh_cfg.get("min_include_z") or min_size.get("z") or None
        agent_radius = navmesh_cfg.get("agent_radius") or 10.0
        max_step_height = navmesh_cfg.get("max_step_height")
        
        _log_info(f"[EnvsetRuntime] Starting async NavMesh baking at: {root_path}")

        navmesh_utils = _require_module("OmniNavExt.envset.navmesh_utils", "baking navmesh async")
        ensure_navmesh_async = navmesh_utils.ensure_navmesh_async

        navmesh = await ensure_navmesh_async(
            root_prim_path=root_path,
            z_padding=z_padding,
            include_volume_parent=include_parent,
            min_xy=min_xy,
            min_z=min_z,
            agent_radius=agent_radius,
            max_step_height=max_step_height,
        )
        
        if navmesh is None:
            _log_error("[EnvsetRuntime] NavMesh baking failed!")
            return False
        
        cls.mark_navmesh_ready(envset_cfg)
        _log_info("[EnvsetRuntime] NavMesh baking completed successfully")
        return True

    @classmethod
    def _setup_virtual_characters(cls, envset_cfg):
        """Spawn virtual-human USD prims only; behavior scripts and animation graphs are attached later."""
        if cls._vh_spawned:
            return
        
        stage_util = _require_module(
            "OmniNavExt.envset.stage_util",
            "spawning virtual humans",
        )
        CharacterUtil = stage_util.CharacterUtil
        set_prim_scale = stage_util.StageUtil.set_prim_scale
        
        vh_cfg = (envset_cfg.get("virtual_humans") or {}) if envset_cfg else {}
        if not vh_cfg:
            return
        env_unit_scale = cls._get_env_unit_scale(envset_cfg)
        spawn_points = vh_cfg.get("spawn_points") or []
        name_sequence = vh_cfg.get("name_sequence") or []
        assets = vh_cfg.get("assets") or {}
        count = vh_cfg.get("count")
        if count is None:
            count = max(len(spawn_points), len(name_sequence), len(assets))
        try:
            count = int(count)
        except Exception:
            return
        if count <= 0:
            return

        asset_root = cls._resolve_asset_root(vh_cfg.get("asset_root"))
        fallback_asset = next(iter(assets.values()), None)
        default_scale = vh_cfg.get("scale")

        spawned_any = False
        spawned_prims = []
        for idx in range(count):
            name = cls._resolve_character_name(name_sequence, idx)
            asset = assets.get(name) or fallback_asset
            usd_path = cls._resolve_asset_path(asset, asset_root)
            if not usd_path:
                continue
            spawn = spawn_points[idx] if idx < len(spawn_points) else {}
            scaled = scale_env_position(spawn.get("position"), env_unit_scale)
            carb_mod = _require_carb("spawning virtual humans")
            pos = carb_mod.Float3(*scaled)
            rot = cls._safe_float(spawn.get("orientation_deg"), 0.0)
            prim = CharacterUtil.load_character_usd_to_stage(usd_path, pos, rot, name)
            if prim and prim.IsValid():
                scale_value = spawn.get("scale", default_scale)
                if scale_value is not None and not set_prim_scale(prim, scale_value):
                    _log_warn(
                        f"[EnvsetRuntime] Failed to apply scale={scale_value} to virtual human '{name}'"
                    )
                cls._exclude_from_navmesh(prim.GetPrimPath())
                spawned_prims.append(prim)
                spawned_any = True

        if spawned_any:
            # Optional: apply colliders to the freshly spawned virtual humans here.
            # cls._apply_colliders_to_spawned_characters(spawned_prims, envset_cfg)
            cls._vh_spawned = True
            debug_info = [
                f"name={prim.GetName()}, path={prim.GetPrimPath()}"
                for prim in spawned_prims
                if prim and prim.IsValid()
            ]
            print(
                f"[EnvsetRuntime] Spawned {len(spawned_prims)} virtual humans "
                f"(behaviors NOT yet initialized): {'; '.join(debug_info)}"
            )

    @classmethod
    def initialize_virtual_humans(cls, envset_cfg):
        """Attach behavior scripts and animation graphs to spawned virtual humans.

        Must be called after the NavMesh has been baked; agent registration fails otherwise.
        """
        if not cls._vh_spawned:
            print("[EnvsetRuntime] No virtual humans spawned, skipping initialization")
            return
        if not cls.is_navmesh_ready(envset_cfg):
            print("[EnvsetRuntime] NavMesh not ready, skipping virtual human initialization")
            return
        
        vh_cfg = (envset_cfg.get("virtual_humans") or {}) if envset_cfg else {}
        if not vh_cfg:
            _log_warn("[EnvsetRuntime] No virtual_humans config found")
            return

        # Use envset's asset_root to set a local fallback path for the default Biped.
        try:
            asset_root_cfg = vh_cfg.get("asset_root") or {}
            asset_root = cls._resolve_asset_root(asset_root_cfg) if asset_root_cfg else None
            if asset_root:
                settings_mod = _require_module("OmniNavExt.envset.settings", "configuring fallback assets")
                AssetPaths = settings_mod.AssetPaths
                
                carb_mod = _require_carb("accessing carb settings for fallback biped asset")
                settings = carb_mod.settings.get_settings()
                # Only write when no value is set, to avoid clobbering an explicit user setting.
                if not settings.get(AssetPaths.FALLBACK_BIPED_ASSET_PATH):
                    candidate = str(Path(asset_root).joinpath("Biped_Setup.usd"))
                    if Path(candidate).exists():
                        settings.set(AssetPaths.FALLBACK_BIPED_ASSET_PATH, candidate)
                        _log_info(
                            f"[EnvsetRuntime] Using local Biped fallback asset from envset.asset_root: {candidate}"
                        )
                    else:
                        _log_warn(
                            f"[EnvsetRuntime] Expected Biped_Setup.usd under asset_root '{asset_root}' "
                            f"but file does not exist; default biped may fail to load."
                        )
        except Exception as exc:
            _log_warn(
                f"[EnvsetRuntime] Failed to configure Biped fallback asset from envset.asset_root: {exc}"
            )

        print(
            f"[EnvsetRuntime] Initializing virtual humans "
            f"(attaching behaviors and animation graphs)... scenario={envset_cfg.get('id')}"
        )
        
        cls._setup_character_behaviors()

        cls._configure_arrival_guard(envset_cfg, vh_cfg)
        
        _log_info("[EnvsetRuntime] Virtual humans initialization completed")

    @classmethod
    def register_robots_as_dynamic_obstacles(cls, runner):
        """Register every robot's main body link as a dynamic obstacle so virtual humans can avoid it.

        Must be called after the NavMesh is ready and virtual humans have been initialized.

        Args:
            runner: A SimulatorRunner instance, used to access current_tasks and robots.
        """
        if not runner:
            return

        registered_count = 0
        for task_name, task in runner.current_tasks.items():
            if not hasattr(task, 'robots') or not task.robots:
                continue

            for robot_name, robot in task.robots.items():
                if not hasattr(robot, 'config'):
                    continue

                robot_prim_path = getattr(robot.config, 'prim_path', None)
                if not robot_prim_path:
                    continue

                # Try to read the robot type from the type and name fields.
                robot_type = getattr(robot.config, 'type', None)
                robot_name_cfg = getattr(robot.config, 'name', None)

                # Try ``type`` first, then ``name``.
                body_link = None
                used_key = None
                for key in [robot_type, robot_name_cfg]:
                    if key:
                        body_link = ROBOT_BODY_LINK_MAP.get(key.lower())
                        if body_link:
                            used_key = key
                            break

                if not body_link:
                    _log_warn(
                        f"[EnvsetRuntime] Unknown robot type/name '{robot_type}'/'{robot_name_cfg}', "
                        f"skipping dynamic obstacle registration"
                    )
                    continue

                obstacle_prim_path = f"{robot_prim_path}/{body_link}"

                try:
                    omni_anim = _require_omni(
                        "omni.anim.people.python_ext",
                        "registering robots as dynamic obstacles",
                    )
                    omni_anim.add_dynamic_obstacle_behavior_script(obstacle_prim_path)
                    registered_count += 1
                    _log_info(
                        f"[EnvsetRuntime] Registered robot '{robot_name}' (type={used_key}) "
                        f"as dynamic obstacle: {obstacle_prim_path}"
                    )
                except Exception as e:
                    _log_warn(
                        f"[EnvsetRuntime] Failed to register robot '{robot_name}' as dynamic obstacle: {e}"
                    )

        if registered_count > 0:
            print(f"[EnvsetRuntime] Registered {registered_count} robot(s) as dynamic obstacles")
        else:
            print("[EnvsetRuntime] No robots registered as dynamic obstacles")

    @classmethod
    def _setup_virtual_routes(cls, envset_cfg):
        vh = (envset_cfg.get("virtual_humans") or {}) if envset_cfg else {}
        options = vh.get("options") or {}
        routes = vh.get("move_routes") or vh.get("routes") or options.get("move_routes") or options.get("routes") or []
        env_unit_scale = cls._get_env_unit_scale(envset_cfg)
        pending = {}
        for entry in routes:
            name = entry.get("name")
            commands = entry.get("commands") or []
            if not name or not commands:
                continue
            scaled_cmds = [cls._scale_route_command(cmd, env_unit_scale) for cmd in commands]
            pending[name] = scaled_cmds
        if not pending:
            return
        cls._pending_routes.update(pending)
        cls._subscribe_route_events()
        cls._flush_routes()

    @classmethod
    def clear_virtual_routes(cls):
        cls._pending_routes.clear()

    @classmethod
    def reset_episode_state(cls, stage=None):
        print("[EnvsetRuntime][DEBUG] reset_episode_state called")
        cls.clear_virtual_humans(stage=stage)
        cls.clear_virtual_routes()

    @classmethod
    def clear_virtual_humans(cls, stage=None):
        """Remove virtual human prims and clear AgentManager state.

        This method properly cleans up AnimationGraph bindings BEFORE deleting prims
        to prevent PhysX deadlock caused by "ghost character" references.
        """
        Sdf = _require_pxr("pxr.Sdf", "clearing virtual humans")
        Usd = _require_pxr("pxr.Usd", "clearing virtual humans")

        AgentManager = _require_module(
            "OmniNavExt.envset.agent_manager",
            "clearing agent manager",
        ).AgentManager

        CharacterUtil = _require_module(
            "OmniNavExt.envset.stage_util",
            "clearing character prims",
        ).CharacterUtil

        if stage is None:
            try:
                omni_usd = _require_omni("omni.usd", "accessing the USD stage for character cleanup")
                stage = omni_usd.get_context().get_stage()
            except Exception:
                stage = None

        cls._debug_dump_characters(stage, "before_clear_virtual_humans")

        # Step 1: Clear AgentManager first to avoid re-registration during cleanup.
        try:
            if AgentManager.has_instance():
                mgr = AgentManager.get_instance()
                mgr.clear_agent()
                mgr.clear_agent_data_dicts()
                print("[EnvsetRuntime] AgentManager cleared")
        except Exception as exc:
            print(f"[EnvsetRuntime] Failed to clear AgentManager: {exc}")

        if stage is not None:
            # Collect character root prims.
            character_prims = CharacterUtil.get_characters_root_in_stage(count_invisible=True)

            prim_paths = []
            skelroot_prims = []  # Collect SkelRoot prims for AnimationGraph cleanup

            for prim in character_prims:
                try:
                    prim_path = prim.GetPrimPath()
                    if "Biped_Setup" in str(prim_path):
                        print(f"[EnvsetRuntime] Keeping Biped_Setup (shared animation resource): {prim_path}")
                        continue
                    prim_paths.append(prim_path)

                    # Collect all SkelRoot prims under this character
                    for child in Usd.PrimRange(prim):
                        if child.GetTypeName() == "SkelRoot":
                            skelroot_prims.append(child)
                except Exception:
                    print("[EnvsetRuntime] Failed to get prim path for virtual human root")

            # Step 2: Remove AnimationGraphAPI from all SkelRoots BEFORE deleting prims
            # This prevents PhysX deadlock from "ghost character" references
            if skelroot_prims:
                skelroot_sdf_paths = [Sdf.Path(p.GetPrimPath()) for p in skelroot_prims]
                print(f"[EnvsetRuntime] Removing AnimationGraphAPI from {len(skelroot_prims)} SkelRoots")
                try:
                    omni_kit = _require_omni("omni.kit.commands", "removing AnimationGraphAPI")
                    omni_kit.execute("RemoveAnimationGraphAPICommand", paths=skelroot_sdf_paths)
                except Exception as exc:
                    print(f"[EnvsetRuntime] RemoveAnimationGraphAPICommand failed: {exc}")

                # Condition-driven: Wait for AnimationGraphAPI to be truly removed
                cls._await_animation_graph_removed(skelroot_prims)

            # Step 3: Clear ScriptManager instances
            if skelroot_prims:
                cls._clear_script_instances(skelroot_prims)

            # Step 4: Delete character root prims
            if prim_paths:
                print(f"[EnvsetRuntime] Deleting {len(prim_paths)} character prims: {prim_paths}")
                try:
                    omni_kit = _require_omni("omni.kit.commands", "deleting character prims")
                    omni_kit.execute(
                        "DeletePrims",
                        paths=prim_paths,
                        destructive=True,
                    )
                except Exception as exc:
                    print(f"[EnvsetRuntime] DeletePrims failed: {exc}")
                    for prim_path in prim_paths:
                        try:
                            stage.RemovePrim(prim_path)
                        except Exception:
                            pass

                # Condition-driven: Wait for prims to be truly deleted
                cls._await_prims_deleted(prim_paths, stage)

        cls._debug_dump_characters(stage, "after_clear_virtual_humans")
        cls._vh_spawned = False

    @classmethod
    def _await_animation_graph_removed(cls, skelroot_prims, max_attempts: int = 30):
        """Condition-driven: Wait for AnimationGraphAPI to be removed from all SkelRoots."""
        try:
            import omni.anim.graph.schema as AnimGraphSchema
            from omni.kit.app import get_app
        except ImportError as exc:
            print(f"[EnvsetRuntime] Cannot check AnimationGraphAPI removal: {exc}")
            return

        app = get_app()

        def check_all_removed():
            """Check if AnimationGraphAPI is removed from all SkelRoots."""
            still_bound = []
            for prim in skelroot_prims:
                if not prim.IsValid():
                    continue  # Prim already deleted, treat as success
                try:
                    anim_graph_api = AnimGraphSchema.AnimationGraphAPI(prim)
                    relation = anim_graph_api.GetAnimationGraphRel()
                    if relation and relation.IsValid():
                        targets = relation.GetTargets()
                        if targets:
                            still_bound.append((prim.GetPrimPath(), targets))
                except Exception:
                    pass  # API not applied, treat as success
            return len(still_bound) == 0, still_bound

        print(f"[EnvsetRuntime] Polling for AnimationGraphAPI removal (max {max_attempts} attempts)...")
        for attempt in range(max_attempts):
            app.update()
            all_removed, still_bound = check_all_removed()
            if all_removed:
                print(f"[EnvsetRuntime] ✓ AnimationGraphAPI removed after {attempt + 1} attempts")
                return

            if attempt % 10 == 9:
                print(f"[EnvsetRuntime] Still waiting for API removal... {len(still_bound)} pending")

        _, still_bound = check_all_removed()
        # Don't raise exception since prims will be deleted anyway, but log warning
        print(f"[EnvsetRuntime] WARNING: AnimationGraphAPI not fully removed after {max_attempts} attempts: {still_bound}")

    @classmethod
    def _clear_script_instances(cls, skelroot_prims):
        """Clear ScriptManager instances for the given SkelRoot prims."""
        try:
            from omni.kit.scripting.scripts.script_manager import ScriptManager
            script_manager = ScriptManager.get_instance()
            if not script_manager:
                return

            for prim in skelroot_prims:
                prim_path = str(prim.GetPrimPath())
                if prim_path in (script_manager._prim_to_scripts or {}):
                    print(f"[EnvsetRuntime] Clearing script instances for {prim_path}")
                    # Call script cleanup methods if available
                    insts = script_manager._prim_to_scripts.get(prim_path, {})
                    for _, inst in insts.items():
                        if inst and hasattr(inst, "on_stop"):
                            try:
                                inst.on_stop()
                            except Exception:
                                pass
                    # Remove from ScriptManager
                    script_manager._prim_to_scripts.pop(prim_path, None)
        except Exception as exc:
            print(f"[EnvsetRuntime] Failed to clear script instances: {exc}")

    @classmethod
    def _await_prims_deleted(cls, prim_paths, stage, max_attempts: int = 30):
        """Condition-driven: Wait for prims to be truly deleted from stage."""
        try:
            from omni.kit.app import get_app
        except ImportError as exc:
            print(f"[EnvsetRuntime] Cannot check prim deletion: {exc}")
            return

        app = get_app()

        def check_all_deleted():
            """Check if all prims are deleted from stage."""
            still_exist = []
            for prim_path in prim_paths:
                prim = stage.GetPrimAtPath(prim_path)
                if prim and prim.IsValid():
                    still_exist.append(prim_path)
            return len(still_exist) == 0, still_exist

        print(f"[EnvsetRuntime] Polling for Prim deletion (max {max_attempts} attempts)...")
        for attempt in range(max_attempts):
            app.update()
            all_deleted, still_exist = check_all_deleted()
            if all_deleted:
                print(f"[EnvsetRuntime] ✓ All prims deleted after {attempt + 1} attempts")
                return

            if attempt % 10 == 9:
                print(f"[EnvsetRuntime] Still waiting for deletion... {len(still_exist)} pending")

        _, still_exist = check_all_deleted()
        raise RuntimeError(
            f"[EnvsetRuntime] FATAL: Prims not deleted after {max_attempts} attempts.\n"
            f"Still exist: {still_exist}\n"
            f"This will cause PhysX deadlock."
        )

    @classmethod
    def reconcile_virtual_humans(cls, envset_cfg, stage=None, defer_routes: bool = False):
        vh_cfg = (envset_cfg.get("virtual_humans") or {}) if envset_cfg else {}
        if not vh_cfg:
            cls.clear_virtual_humans(stage=stage)
            return
        cls.clear_virtual_humans(stage=stage)
        cls.clear_virtual_routes()
        cls._setup_virtual_characters(envset_cfg)
        cls.initialize_virtual_humans(envset_cfg)
        if not defer_routes:
            cls._setup_virtual_routes(envset_cfg)

    @classmethod
    def mark_navmesh_ready(cls, envset_cfg):
        key = cls._compute_navmesh_key(envset_cfg)
        if key:
            cls._navmesh_ready_by_key[key] = True
        cls._navmesh_ready = True

    @classmethod
    def is_navmesh_ready(cls, envset_cfg) -> bool:
        key = cls._compute_navmesh_key(envset_cfg)
        if key:
            return bool(cls._navmesh_ready_by_key.get(key))
        return cls._navmesh_ready

    @classmethod
    def reset_navmesh_cache(cls):
        cls._navmesh_ready_by_key.clear()
        cls._navmesh_ready = False

    @classmethod
    def _subscribe_route_events(cls):
        if cls._route_subscription is not None:
            return
        
        people_settings = _require_omni(
            "omni.anim.people.settings",
            "subscribing to AgentEvent",
        )
        AgentEvent = people_settings.AgentEvent
        
        carb_mod = _require_carb("subscribing to AgentEvent")
        dispatcher = carb_mod.eventdispatcher.get_eventdispatcher()
        
        cls._route_subscription = dispatcher.observe_event(
            event_name=AgentEvent.AgentRegistered,
            on_event=cls._on_agent_registered,
            observer_name="OmniNav/envset/runtime/routes",
        )
        print("[EnvsetRuntime] Subscribed to AgentEvent.AgentRegistered for route injection")

    @classmethod
    def _on_agent_registered(cls, event):
        payload = getattr(event, "payload", None) or {}
        agent_name = payload.get("agent_name")
        if not agent_name:
            print(f"[EnvsetRuntime] AgentRegistered event without agent_name: {payload}")
            return
        print(
            "[EnvsetRuntime] AgentRegistered event received for '%s' (pending routes: %s)",
            agent_name,
            list(cls._pending_routes.keys()),
        )
        cls._inject_route(agent_name)

    @classmethod
    def _flush_routes(cls):
        for agent_name in list(cls._pending_routes.keys()):
            cls._inject_route(agent_name)

    @classmethod
    def _inject_route(cls, agent_name: str):
        commands = cls._pending_routes.get(agent_name)
        if not commands:
            print(f"[EnvsetRuntime] No pending route for {agent_name} when trying to inject")
            return
        
        AgentManager = _require_module(
            "OmniNavExt.envset.agent_manager",
            "injecting routes",
        ).AgentManager
        
        mgr = AgentManager.get_instance()
        if not mgr.agent_registered(agent_name):
            print(f"[EnvsetRuntime] Agent '{agent_name}' not registered yet when injecting route")
            return

        # IMPORTANT: force_inject=True overrides any default command;
        # instant=True applies immediately rather than waiting for the current command to finish.
        try:
            mgr.inject_command(agent_name, commands, force_inject=True, instant=True)
            cls._pending_routes.pop(agent_name, None)
            print(f"[EnvsetRuntime] Successfully injected route to agent '{agent_name}': {commands}")
        except Exception as exc:
            print(f"[EnvsetRuntime] Failed to inject route to agent '{agent_name}': {exc}")
            # Do not pop on failure; keep the route around for a later retry.
            import traceback
            print(traceback.format_exc())

    @staticmethod
    def _resolve_scene_root(stage) -> str:
        if not stage:
            raise FileNotFoundError("[EnvsetRuntime] USD stage not available when resolving scene root")
        try:
            from OmniNavExt.envset.core.scene_manager import find_scene_root

            resolved = find_scene_root(stage)
        except Exception:
            resolved = None
        if not resolved:
            raise FileNotFoundError("[EnvsetRuntime] Scene root not found in loaded stage")
        return resolved

    @staticmethod
    def _compute_navmesh_key(envset_cfg: Dict[str, Any]) -> str:
        if not isinstance(envset_cfg, dict):
            return ""
        navmesh_cfg = envset_cfg.get("navmesh") or {}
        scene_cfg = envset_cfg.get("scene") or {}
        mp_cfg = scene_cfg.get("matterport") if isinstance(scene_cfg.get("matterport"), dict) else {}
        usd_path = (
            scene_cfg.get("usd_path")
            or scene_cfg.get("asset_path")
            or mp_cfg.get("usd_path")
            or mp_cfg.get("obj_path")
        )
        units = scene_cfg.get("units_in_meters")
        if units is None:
            units = envset_cfg.get("scene_units_in_meters")
        key_payload = {
            "scene": {
                "usd_path": usd_path,
                "root_prim_path": scene_cfg.get("root_prim_path"),
                "navmesh_root_prim_path": scene_cfg.get("navmesh_root_prim_path"),
                "units_in_meters": units,
            },
            "navmesh": {
                "bake_root_prim_path": navmesh_cfg.get("bake_root_prim_path"),
                "include_volume_parent": navmesh_cfg.get("include_volume_parent"),
                "z_padding": navmesh_cfg.get("z_padding"),
                "agent_radius": navmesh_cfg.get("agent_radius"),
                "max_step_height": navmesh_cfg.get("max_step_height"),
                "min_include_xy": navmesh_cfg.get("min_include_xy"),
                "min_include_z": navmesh_cfg.get("min_include_z"),
                "min_include_volume_size": navmesh_cfg.get("min_include_volume_size"),
            },
        }
        return json.dumps(key_payload, sort_keys=True)

    # --------------------- helpers ---------------------

    @staticmethod
    def _resolve_character_name(sequence, idx):
        if sequence and idx < len(sequence) and sequence[idx]:
            return str(sequence[idx])
        
        CharacterUtil = _require_module(
            "OmniNavExt.envset.stage_util",
            "resolving character name",
        ).CharacterUtil
        
        return CharacterUtil.get_character_name_by_index(idx)

    @staticmethod
    def _resolve_asset_root(asset_root_cfg) -> Optional[str]:
        if not asset_root_cfg:
            return None
        settings_key = asset_root_cfg.get("settings_key")
        if settings_key:
            try:
                carb_mod = _require_carb("reading carb settings for asset_root")
                value = carb_mod.settings.get_settings().get(settings_key)
                if value:
                    return str(value)
            except Exception:
                pass
        fallback = asset_root_cfg.get("fallback")
        if fallback:
            return str(fallback)
        path_val = asset_root_cfg.get("path")
        if path_val:
            return str(path_val)
        return None

    @staticmethod
    def _resolve_asset_path(asset: Optional[str], root: Optional[str]) -> Optional[str]:
        if not asset:
            raise ValueError("[EnvsetRuntime] virtual_humans.assets must define a USD file name")
        if not root:
            # Require explicit asset_root to avoid implicit cwd resolution.
            raise ValueError("[EnvsetRuntime] virtual_humans.asset_root.path is required for assets")
        asset_str = str(asset)
        resolved = Path(root).joinpath(asset_str)
        if not EnvsetTaskRuntime._asset_path_exists(str(resolved)):
            raise FileNotFoundError(
                f"[EnvsetRuntime] Character asset '{asset_str}' not found: {resolved}"
            )
        return str(resolved)

    @staticmethod
    def _asset_path_exists(path: str) -> bool:
        if not path:
            return False
        try:
            return Path(path).expanduser().exists()
        except Exception:
            return False

    @staticmethod
    def _safe_float(value, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    @classmethod
    def _apply_colliders_to_spawned_characters(cls, spawned_prims, envset_cfg):
        """Apply colliders to the virtual humans that have just been spawned."""
        if not spawned_prims:
            return
        
        try:
            from .virtual_human_colliders import VirtualHumanColliderApplier, ColliderConfig
            
            vh_cfg = envset_cfg.get("virtual_humans") or {}
            approx_shape = vh_cfg.get("collider_shape") or vh_cfg.get("approximation_shape") or "convexHull"
            kinematic_flag = vh_cfg.get("collider_kinematic")
            if kinematic_flag is None:
                kinematic_flag = True
            
            collider_cfg = ColliderConfig(
                approximation_shape=str(approx_shape),
                kinematic=bool(kinematic_flag)
            )
            
            character_paths = [str(prim.GetPrimPath()) for prim in spawned_prims if prim and prim.IsValid()]
            
            if character_paths:
                applier = VirtualHumanColliderApplier(
                    character_paths=character_paths,
                    collider_config=collider_cfg,
                )
                # Apply immediately rather than waiting for the timeline to start.
                applier.activate(apply_immediately=True)
                _log_info(f"[EnvsetRuntime] Applied colliders to {len(character_paths)} spawned characters")
        except Exception as exc:
            _log_warn(f"[EnvsetRuntime] Failed to apply colliders to spawned characters: {exc}")

    @staticmethod
    def _exclude_from_navmesh(prim_path):
        if prim_path is None:
            return
        prim_path_str = str(prim_path).strip()
        if not prim_path_str:
            return
        try:
            nav_schema = _require_pxr(
                "pxr.NavSchema",
                "applying NavMesh API to prims",
            )
            NavSchema = nav_schema.NavSchema
            
            omni_kit = _require_omni(
                "omni.kit.commands",
                "applying NavMesh API to prims",
            )
            omni_kit.execute(
                "ApplyNavMeshAPICommand", prim_path=prim_path_str, api=NavSchema.NavMeshExcludeAPI
            )
        except Exception:
            pass

    @classmethod
    def _setup_character_behaviors(cls):
        """
        Setup character behaviors with condition-driven waiting.
        Raises RuntimeError on any failure (no silent failures).
        """
        _log_info("[EnvsetRuntime] Setting up character behaviors...")

        stage_util = _require_module(
            "OmniNavExt.envset.stage_util",
            "initializing character behaviors",
        )
        CharacterUtil = stage_util.CharacterUtil
        populate_anim_graph = stage_util.populate_anim_graph

        # Step 1: Load biped - must succeed
        biped = CharacterUtil.load_default_biped_to_stage()
        if biped is None or not biped.IsValid():
            raise RuntimeError(
                "[EnvsetRuntime] FATAL: Default biped failed to load. "
                "Cannot proceed with virtual human initialization."
            )
        _log_info(f"[EnvsetRuntime] Default biped loaded: {biped.GetPath()}")

        # Step 2: Get character list - must not be empty
        character_list = CharacterUtil.get_characters_in_stage()
        character_roots = CharacterUtil.get_characters_root_in_stage()
        if not character_list:
            raise RuntimeError(
                "[EnvsetRuntime] FATAL: No characters found in stage after spawn."
            )
        character_paths = [str(prim.GetPrimPath()) for prim in character_list]
        print(f"[EnvsetRuntime] Characters detected: {len(character_list)} -> {character_paths}")

        # Step 3: Ensure AnimationGraph exists (condition-driven)
        anim_graph = cls._ensure_animation_graph(biped, populate_anim_graph)
        _log_info(f"[EnvsetRuntime] AnimationGraph: {anim_graph.GetPath()}")

        # Step 4: Apply AnimationGraph (has internal condition-driven polling)
        CharacterUtil.setup_animation_graph_to_character(character_list, anim_graph)

        # Step 5: Apply behavior scripts
        BehaviorScriptPaths = _require_module(
            "OmniNavExt.envset.settings",
            "resolving behavior script paths",
        ).BehaviorScriptPaths

        script_path = BehaviorScriptPaths.behavior_script_path()
        if not script_path:
            raise RuntimeError("[EnvsetRuntime] FATAL: Behavior script path is empty.")
        _log_info(f"[EnvsetRuntime] Attaching behavior script: {script_path}")
        CharacterUtil.setup_python_scripts_to_character(character_list, script_path)

        # Step 6: Register with World
        CharacterUtil.register_characters_with_world(character_roots, character_list)

        # Step 7: Wait for script instances (condition-driven)
        cls._await_script_instances_until_ready(character_list)

        # Step 8: Wait for agent registration (condition-driven)
        cls._await_agents_registered(character_list)

        # Step 9: Add metrosim semantics (optional, don't fail on error)
        try:
            metropolis_utils = _require_omni(
                "omni.metropolis.utils.semantics_util",
                "adding metrosim semantics to characters",
            )
            SemanticsUtils = metropolis_utils.SemanticsUtils
            SemanticsUtils.add_update_prim_metrosim_semantics(
                character_list,
                type_value="class",
                name="character",
            )
        except Exception as exc:
            _log_warn(f"[EnvsetRuntime] Failed to add metrosim semantics: {exc}")

        _log_info("[EnvsetRuntime] ✓ Character behaviors setup completed")

    @classmethod
    def _ensure_animation_graph(cls, biped, populate_anim_graph, max_attempts: int = 30):
        """
        Ensure AnimationGraph exists, using condition-driven polling.

        Raises:
            RuntimeError: If AnimationGraph not available after max_attempts
        """
        CharacterUtil = _require_module(
            "OmniNavExt.envset.stage_util",
            "ensuring animation graph",
        ).CharacterUtil

        anim_graph = CharacterUtil.get_anim_graph_from_character(biped)
        if anim_graph:
            return anim_graph

        _log_info("[EnvsetRuntime] AnimationGraph not found, populating...")
        import omni.anim.graph.core  # type: ignore  # noqa: F401
        populate_anim_graph()

        import omni.kit.app
        app = omni.kit.app.get_app()

        for attempt in range(max_attempts):
            app.update()
            anim_graph = CharacterUtil.get_anim_graph_from_character(biped)
            if anim_graph:
                print(f"[EnvsetRuntime] ✓ AnimationGraph ready after {attempt + 1} attempts")
                return anim_graph

        raise RuntimeError(
            f"[EnvsetRuntime] FATAL: AnimationGraph not available after {max_attempts} attempts. "
            f"Biped prim: {biped.GetPath()}"
        )

    @staticmethod
    def _await_script_manager_instances(character_list, max_attempts: int = 6):
        """Wait for ScriptManager to create behavior script instances, updating the Kit app between attempts."""
        try:
            from omni.kit.scripting.scripts.script_manager import ScriptManager  # type: ignore
            script_manager = ScriptManager.get_instance()
        except Exception as exc:
            print(f"[EnvsetRuntime] ScriptManager diagnostics unavailable: {exc}")
            return

        if not script_manager:
            print("[EnvsetRuntime] ScriptManager instance is None; cannot inspect behavior scripts.")
            return

        try:
            import omni.kit.app  # type: ignore
            app = omni.kit.app.get_app()
        except Exception:
            app = None

        def _dump_status() -> bool:
            script_map = script_manager._prim_to_scripts or {}
            print(f"[EnvsetRuntime][DEBUG] ScriptManager currently tracks {len(script_map)} prim entries")
            ready = True
            for prim in character_list:
                prim_path = str(prim.GetPrimPath())
                insts = script_map.get(prim_path)
                if not insts:
                    print(
                        f"[EnvsetRuntime][DEBUG] No script instance registered yet for {prim_path}; "
                        "behavior script may still be initializing."
                    )
                    ready = False
                    continue
                live_inst = False
                for _, inst in insts.items():
                    if inst:
                        live_inst = True
                        agent_name = inst.get_agent_name() if hasattr(inst, "get_agent_name") else None
                        print(
                            f"[EnvsetRuntime][DEBUG] Script instance detected for {prim_path}: "
                            f"{inst} (agent_name={agent_name})"
                        )
                if not live_inst:
                    print(
                        f"[EnvsetRuntime][DEBUG] Script entries exist for {prim_path} but all instances are None; "
                        "waiting for initialization."
                    )
                    ready = False
            return ready

        for attempt in range(max_attempts):
            if _dump_status():
                EnvsetTaskRuntime._register_scripts_with_agent_manager(character_list, script_manager)
                return
            if not app:
                break
            try:
                app.update()
            except Exception as exc:
                print(f"[EnvsetRuntime][DEBUG] app.update() failed while waiting for scripts: {exc}")
                break
        if _dump_status():
            EnvsetTaskRuntime._register_scripts_with_agent_manager(character_list, script_manager)

    @staticmethod
    def _register_scripts_with_agent_manager(character_list, script_manager):
        """Manually trigger character behavior initialization and AgentManager registration if needed."""
        if not character_list:
            return
        
        AgentManager = _require_module(
            "OmniNavExt.envset.agent_manager",
            "manual agent registration",
        ).AgentManager
        
        mgr = AgentManager.get_instance() if AgentManager.has_instance() else None

        for prim in character_list:
            prim_path = str(prim.GetPrimPath())
            insts = (script_manager._prim_to_scripts or {}).get(prim_path)
            if not insts:
                continue
            for _, inst in insts.items():
                if not inst:
                    continue
                try:
                    if hasattr(inst, "init_character"):
                        print(f"[EnvsetRuntime][DEBUG] Calling init_character() on {inst}")
                        inst.init_character()
                except Exception as exc:
                    print(f"[EnvsetRuntime][DEBUG] init_character failed for {inst}: {exc}")
                try:
                    if hasattr(inst, "on_play"):
                        print(f"[EnvsetRuntime][DEBUG] Calling on_play() on {inst}")
                        inst.on_play()
                except Exception as exc:
                    print(f"[EnvsetRuntime][DEBUG] on_play failed for {inst}: {exc}")
                if mgr and hasattr(inst, "get_agent_name"):
                    try:
                        agent_name = inst.get_agent_name()
                        print(f"[EnvsetRuntime][DEBUG] Manually registering agent {agent_name} for prim {prim_path}")
                        mgr.register_agent(agent_name, inst.prim_path)
                        print(f"[EnvsetRuntime][DEBUG] Route injection will be triggered by AgentRegistered event for {agent_name}")
                    except Exception as exc:
                        print(f"[EnvsetRuntime][DEBUG] register_agent failed for {inst}: {exc}")

    @classmethod
    def _await_script_instances_until_ready(cls, character_list, max_attempts: int = 50):
        """
        Condition-driven: Wait for all ScriptManager instances to be ready.

        Raises:
            RuntimeError: If script instances not ready after max_attempts
        """
        from omni.kit.scripting.scripts.script_manager import ScriptManager  # type: ignore
        script_manager = ScriptManager.get_instance()
        if not script_manager:
            raise RuntimeError("[EnvsetRuntime] FATAL: ScriptManager is None.")

        import omni.kit.app
        app = omni.kit.app.get_app()

        def check_all_ready():
            script_map = script_manager._prim_to_scripts or {}
            missing = []
            for prim in character_list:
                prim_path = str(prim.GetPrimPath())
                insts = script_map.get(prim_path)
                if not insts or not any(inst for inst in insts.values() if inst):
                    missing.append(prim_path)
            return len(missing) == 0, missing

        print(f"[EnvsetRuntime] Polling for script instances (max {max_attempts} attempts)...")
        for attempt in range(max_attempts):
            app.update()
            all_ready, missing = check_all_ready()
            if all_ready:
                print(f"[EnvsetRuntime] ✓ Script instances ready after {attempt + 1} attempts")
                cls._register_scripts_strict(character_list, script_manager)
                return

            if attempt % 10 == 9:
                print(f"[EnvsetRuntime] Still waiting for scripts... {len(missing)} pending")

        _, missing = check_all_ready()
        raise RuntimeError(
            f"[EnvsetRuntime] FATAL: Script instances not ready after {max_attempts} attempts.\n"
            f"Missing: {missing}"
        )

    @classmethod
    def _register_scripts_strict(cls, character_list, script_manager):
        """
        Register scripts with AgentManager. Raises on failure (no silent errors).

        Raises:
            RuntimeError: If any script or agent registration fails
        """
        AgentManager = _require_module(
            "OmniNavExt.envset.agent_manager",
            "agent registration",
        ).AgentManager

        mgr = AgentManager.get_instance()
        if not mgr:
            raise RuntimeError("[EnvsetRuntime] FATAL: AgentManager is None.")

        for prim in character_list:
            prim_path = str(prim.GetPrimPath())
            insts = (script_manager._prim_to_scripts or {}).get(prim_path)
            if not insts:
                raise RuntimeError(f"[EnvsetRuntime] FATAL: No script instances for {prim_path}")

            for _, inst in insts.items():
                if not inst:
                    continue
                # Call lifecycle methods - don't catch exceptions, let failures propagate
                if hasattr(inst, "init_character"):
                    print(f"[EnvsetRuntime][DEBUG] Calling init_character() on {inst}")
                    inst.init_character()
                if hasattr(inst, "on_play"):
                    print(f"[EnvsetRuntime][DEBUG] Calling on_play() on {inst}")
                    inst.on_play()
                if hasattr(inst, "get_agent_name"):
                    agent_name = inst.get_agent_name()
                    print(f"[EnvsetRuntime][DEBUG] Registering agent {agent_name} for {prim_path}")
                    mgr.register_agent(agent_name, inst.prim_path)
                    print(f"[EnvsetRuntime] Agent {agent_name} registration initiated")

    @classmethod
    def _await_agents_registered(cls, character_list, max_attempts: int = 30):
        """
        Condition-driven: Wait for all agents to be registered.

        Raises:
            RuntimeError: If agents not registered after max_attempts
        """
        AgentManager = _require_module(
            "OmniNavExt.envset.agent_manager",
            "agent registration check",
        ).AgentManager

        mgr = AgentManager.get_instance()
        if not mgr:
            raise RuntimeError("[EnvsetRuntime] FATAL: AgentManager is None.")

        # Collect expected agent names
        expected_agents = []
        from omni.kit.scripting.scripts.script_manager import ScriptManager  # type: ignore
        script_manager = ScriptManager.get_instance()
        for prim in character_list:
            prim_path = str(prim.GetPrimPath())
            insts = (script_manager._prim_to_scripts or {}).get(prim_path, {})
            for _, inst in insts.items():
                if inst and hasattr(inst, "get_agent_name"):
                    expected_agents.append(inst.get_agent_name())

        if not expected_agents:
            print("[EnvsetRuntime] No agents to wait for")
            return

        import omni.kit.app
        app = omni.kit.app.get_app()

        def check_all_registered():
            pending = [name for name in expected_agents if not mgr.agent_registered(name)]
            return len(pending) == 0, pending

        print(f"[EnvsetRuntime] Polling for agent registration: {expected_agents}")
        for attempt in range(max_attempts):
            app.update()
            all_registered, pending = check_all_registered()
            if all_registered:
                print(f"[EnvsetRuntime] ✓ All agents registered after {attempt + 1} attempts")
                return

        _, pending = check_all_registered()
        raise RuntimeError(
            f"[EnvsetRuntime] FATAL: Agents not registered after {max_attempts} attempts.\n"
            f"Pending: {pending}"
        )

    @classmethod
    def _configure_arrival_guard(cls, envset_cfg, vh_cfg):
        tol = cls._safe_float(vh_cfg.get("arrival_tolerance_m"), 0.5)
        scene_cfg = envset_cfg.get("scene") or {}
        scene_category = scene_cfg.get("category")
        guard = cls._get_arrival_guard()
        guard.enable_if_grscenes(scene_category, tolerance_m=tol)

    @staticmethod
    def _scale_route_command(command: str, env_unit_scale: float) -> str:
        if not command:
            return command
        try:
            return try_scale_goto_command(command, env_unit_scale)
        except ValueError as exc:
            raise ValueError(f"[EnvsetRuntime] Failed to scale route command '{command}': {exc}") from exc

    @staticmethod
    def _get_env_unit_scale(envset_cfg: Dict[str, any]) -> float:
        scene_cfg = envset_cfg.get("scene") or {}
        units_value = scene_cfg.get("units_in_meters")
        if units_value is None:
            units_value = envset_cfg.get("scene_units_in_meters")
            context = "envset.scene_units_in_meters"
        else:
            context = "envset.scene.units_in_meters"
        return resolve_env_unit_scale(units_value, context=context)
