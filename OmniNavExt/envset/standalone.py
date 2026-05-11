from __future__ import annotations

import argparse
import asyncio
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict

# Note: Do NOT import Isaac Sim modules (carb, omni, etc.) here!
# They must be imported AFTER SimulationApp is initialized.

from OmniNav.core.config import Config
from OmniNav.core.runner import SimulatorRunner
from OmniNav.core.task_config_manager.base import create_task_config_manager

from OmniNavExt import import_extensions
from OmniNavExt.envset.config_loader import EnvsetConfigLoader
from OmniNavExt.envset.recording import build_recording_payload, resolve_recording_dirs
from OmniNavExt.envset.waypoint_recording import WaypointRecorder

# Import refactored core modules
from OmniNavExt.envset.core import (
    SimulationBootstrap,
    SimulationConfig,
    PhysicsManager,
    NavMeshManager,
    SceneManager,
    SimulationLifecycle,
)
from OmniNavExt.envset.core.scene_manager import (
    is_matterport_scenario,
    find_scene_root,
    should_use_camera_light,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run envset scenarios through OmniNav's SimulatorRunner.")
    parser.add_argument("--config", required=True, type=Path, help="Path to OmniNav YAML config.")
    parser.add_argument("--envset", required=True, type=Path, help="Path to envset JSON file.")
    parser.add_argument("--scenario", default=None, help="Scenario id inside envset JSON.")
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=None,
        help="Base directory for envset asset paths.",
    )
    parser.add_argument("--headless", action="store_true", help="Force headless SimulationApp.")
    parser.add_argument(
        "--extension-path",
        action="append",
        dest="extension_paths",
        help="Additional extension search path (can be specified multiple times). "
             "Example: --extension-path /path/to/isaaclab/source",
    )
    parser.add_argument(
        "--skip-isaac-assets",
        action="store_true",
        help="Skip querying shared Isaac Sim asset root (defaults to querying).",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=None,
        help="Keep SimulationApp alive for N seconds (omit to keep running until window closes).",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Do not auto-start timeline playback after setup.",
    )
    parser.add_argument(
        "--run-data",
        action="store_true",
        help="(Future) Trigger data generation via Runner. Currently not supported.",
    )
    parser.add_argument("--label", default="standalone", help="Tag recorded in Infos.ext_version for diagnostics.")
    return parser.parse_args()


def _parse_config_model(config_dict: dict) -> Config:
    try:
        return Config.model_validate(config_dict)  # pydantic v2
    except AttributeError:  # pragma: no cover - fallback for v1
        return Config.parse_obj(config_dict)


class EnvsetStandaloneRunner:
    """Ties envset configuration into OmniNav's SimulatorRunner lifecycle."""

    def __init__(self, args: argparse.Namespace):
        self._args = args
        self._config_path = args.config.expanduser().resolve()
        self._envset_path = args.envset.expanduser().resolve()
        self._scene_root = args.scene_root.expanduser().resolve() if args.scene_root else None
        self._bundle = EnvsetConfigLoader(
            self._config_path,
            self._envset_path,
            args.scenario,
            self._scene_root,
        ).load()
        self._merged_dict = copy.deepcopy(self._bundle.config)
        self._runner: SimulatorRunner | None = None
        self._simulation_app = None
        self._data_gen = None
        self._keyboard = None
        self._keyboard_robots = []
        self._shutdown_flag = False
        # State for manual-timeline control.
        self._navmesh_success = False  # Result of NavMesh baking.
        self._timeline_init_done = False  # Whether the post-timeline-start init has run.

        self.robot_prim_path = None
        self._rec_enabled = True
        self._rec_fps = 30
        self._sim_step_count = 0

        # Output paths
        self._rec_base_dir: Path | None = None
        self._rec_rgb_video_path: Path | None = None
        self._rec_depth_video_path: Path | None = None
        self._rec_pose_log_path: Path | None = None

        # Capture state (datagen-aligned: IsaacsimCamera + MultiCameraAsyncVideoWriter)
        self._replay_cameras = None  # List[ReplayCamera]
        self._video_writer = None    # MultiCameraAsyncVideoWriter
        self._waypoint_recorder = None  # WaypointRecorder
        self._capture_robot = None   # first robot object for pose extraction
        self._capture_robot_name = None
        self._capture_dt = 0.0
        self._capture_elapsed = 0.0
        self._capture_next = 0.0
        self._capture_frame_idx = 0

        self._resolve_record_dirs()

    # =========================
    # NEW: output video path mapping
    # =========================
    def _resolve_record_dirs(self) -> None:
        """
        Store recordings under sibling `video/` and `path/` directories next to the envset file:
          <envset_dir>/video/<envset_stem>/rgb.mp4
          <envset_dir>/video/<envset_stem>/depth.mp4
          <envset_dir>/path/<envset_stem>/path.json
        """
        envset_path = self._envset_path.expanduser().resolve()
        video_dir, path_dir = resolve_recording_dirs(envset_path)

        video_dir.mkdir(parents=True, exist_ok=True)
        path_dir.mkdir(parents=True, exist_ok=True)

        self._rec_base_dir = video_dir
        self._rec_rgb_video_path = video_dir / "rgb.mp4"
        self._rec_depth_video_path = video_dir / "depth.mp4"
        self._rec_pose_log_path = path_dir / "path.json"

    def _log_scene_units_config(self):
        """Print scene-unit configuration info."""
        scenario = self._bundle.scenario
        scene_cfg = scenario.get("scene") or {}
        navmesh_cfg = scenario.get("navmesh") or {}

        units = scene_cfg.get("units_in_meters")
        print("=" * 70)
        print("[Config] Scene units configuration:")
        print(f"[Config]   - units_in_meters: {units}")
        if units is not None:
            print(f"[Config]   - Interpretation: 1 scene unit = {units} meters")
            if abs(units - 0.01) < 1e-6:
                print(f"[Config]   - Unit type: Centimeters")
            elif abs(units - 1.0) < 1e-6:
                print(f"[Config]   - Unit type: Meters")
        else:
            print(f"[Config]   - No units_in_meters specified (will use default)")

        print(f"[Config] NavMesh configuration (in scene units):")
        print(f"[Config]   - agent_radius: {navmesh_cfg.get('agent_radius', 'N/A')}")
        print(f"[Config]   - z_padding: {navmesh_cfg.get('z_padding', 'N/A')}")
        print("=" * 70)

    def request_shutdown(self):
        self._shutdown_flag = True

    def _log_extension_status(self):
        """Log key extension status for diagnostics."""
        pass  # Extension status now logged via carb only when needed

    def _print_runtime_snapshot(self, label: str):
        """Log runtime snapshot for diagnostics."""
        pass  # Runtime snapshots now logged via carb only when needed

    def run(self):
        """Main run method - orchestrates the simulation lifecycle."""
        self._log_scene_units_config()

        config_model = self._build_config_model()

        # Configure camera sensors before runner creation (aligned with datagen)
        if self._rec_enabled:
            from bench.replay.camera_config import configure_replay_robot_sensors
            for task_cfg in config_model.task_configs:
                for robot_cfg in task_cfg.robots:
                    configure_replay_robot_sensors(robot_cfg)

        self._simulation_app = self._initialize_simulation_app(config_model)
        import_extensions()
        self._prepare_runtime_settings()
        self._runner = self._create_runner_with_app(config_model)
        self._post_runner_initialize()

        scenario = self._bundle.scenario
        scene_cfg = scenario.get("scene") or {}
        navmesh_cfg = scenario.get("navmesh") or {}

        # Handle Matterport scenes
        is_mp = is_matterport_scenario(scene_cfg)
        if is_mp:
            print("[EnvsetStandalone] Importing Matterport scene...")
            matterport_prim = SceneManager.import_matterport_scene(scene_cfg, self._config_path)
            SceneManager.prepare_matterport_navmesh(
                matterport_prim,
                navmesh_cfg,
                scene_cfg,
                self._simulation_app
            )
            print("[EnvsetStandalone] Matterport scene import completed")

        # Load scene
        self._runner.reset(start_timeline=False)
        if should_use_camera_light(scene_cfg):
            self._set_camera_light()

        # Fix physics properties
        scene_root = find_scene_root(self._runner._stage, scenario)
        PhysicsManager.fix_grscenes_physics(scene_cfg, scene_root)

        # Exclude robots from NavMesh baking
        from OmniNavExt.envset.runtime_hooks import EnvsetTaskRuntime
        for task_name, task in self._runner.current_tasks.items():
            if hasattr(task, 'robots') and task.robots:
                for robot_name, robot in task.robots.items():
                    if hasattr(robot, 'config') and hasattr(robot.config, 'prim_path'):
                        self.robot_prim_path = robot.config.prim_path
                        EnvsetTaskRuntime._exclude_from_navmesh(self.robot_prim_path)
                        print(f"[EnvsetStandalone] Excluded robot '{robot_name}' from NavMesh: {self.robot_prim_path}")

        # Bake NavMesh
        if is_mp:
            navmesh_success = True
        else:
            try:
                navmesh_manager = NavMeshManager.from_scenario(navmesh_cfg, scene_cfg, scene_root)
                navmesh_success = navmesh_manager.bake_sync(self._simulation_app, envset_cfg=scenario)
                print("[EnvsetStandalone] NavMesh baking completed")
            except Exception as e:
                print(f"[EnvsetStandalone] NavMesh baking failed: {e}")
                navmesh_success = False

        # Save NavMesh status for use after the timeline starts.
        self._navmesh_success = navmesh_success

        # Disable NavMesh visualization in the viewport.
        import omni.kit.commands
        omni.kit.commands.execute(
            "ChangeSetting",
            path="/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh",
            value=False,
        )

        # Do not auto-start the timeline; wait for the user to click Start in the Isaac Sim UI.
        print("[EnvsetStandalone] Scene loaded. Please click 'Start' in Isaac Sim UI to begin simulation.")

        if self._args.run_data:
            self._init_data_generation()
            self._run_data_generation()
        else:
            self._main_loop()

        print("[EnvsetStandalone] Simulation completed")

    def shutdown(self):
        if self._video_writer is not None:
            print("[EnvsetStandalone] Closing video writer...")
            self._stop_capture()

        app = None
        if self._runner and self._runner.simulation_app:
            app = self._runner.simulation_app
        elif self._simulation_app:
            app = self._simulation_app

        if app:
            try:
                app.close()
            except Exception:
                pass

    # ---------- internal helpers ----------

    def _prepare_runtime_settings(self):
        import carb  # type: ignore
        import carb.settings  # type: ignore
        from OmniNavExt.envset.settings import AssetPaths, Infos
        from OmniNavExt.envset.simulation import (
            ENVSET_AUTOSTART_SETTING,
            ENVSET_PATH_SETTING,
            ENVSET_SCENARIO_SETTING,
        )

        print("[EnvsetStandalone] Preparing runtime settings (no explicit extension enabling)...")

        # Push envset-related parameters into carb settings for the extensions to read.
        settings_iface = carb.settings.get_settings()
        settings_iface.set(ENVSET_PATH_SETTING, str(self._envset_path))
        settings_iface.set(ENVSET_AUTOSTART_SETTING, False)
        settings_iface.set(ENVSET_SCENARIO_SETTING, self._bundle.scenario_id)
        settings_iface.set(AssetPaths.USE_ISAAC_SIM_ASSET_ROOT_SETTING, not self._args.skip_isaac_assets)

        # Stamp the standalone version info to aid later debugging.
        Infos.ext_version = str(self._args.label)
        Infos.ext_path = str(Path(__file__).resolve().parent)

        # Warp initialization (optional but useful).
        try:
            import warp  # type: ignore
            warp.init()
        except Exception as exc:
            carb.log_warn(f"[EnvsetStandalone] Warp init failed: {exc}")

        # Pre-cache the Isaac asset root path (faster over network / Nucleus).
        if not self._args.skip_isaac_assets:
            try:
                asyncio.run(AssetPaths.cache_paths_async())
            except Exception as exc:
                carb.log_warn(f"[EnvsetStandalone] Failed to cache asset root: {exc}")

    def _initialize_simulation_app(self, config: Config):
        from isaacsim import SimulationApp  # type: ignore
        from OmniNav.local_paths import apply_local_path_env, resolve_runtime_kit_path

        simulator_cfg = config.simulator

        apply_local_path_env()
        launch_config = {
            "headless": simulator_cfg.headless,
            "anti_aliasing": 0,
            "hide_ui": False,
            "multi_gpu": False,
            "enable_cameras": True,
            "experience": str(resolve_runtime_kit_path()),
        }
        import os
        os.environ.setdefault("ENABLE_CAMERAS", "1")

        sim_app = SimulationApp(launch_config)
        sim_app._carb_settings.set("/physics/cooking/ujitsoCollisionCooking", False)

        self._configure_streaming(sim_app, simulator_cfg)
        return sim_app

    def _configure_streaming(self, sim_app, simulator_cfg):
        """Configure streaming for Isaac Sim 5.0+."""
        native = getattr(simulator_cfg, "native", False)
        webrtc = getattr(simulator_cfg, "webrtc", False)

        if native:
            print("[EnvsetStandalone] native streaming is deprecated, enabling webrtc instead.")
        self._configure_streaming_500(sim_app, native or webrtc)

    @staticmethod
    def _configure_streaming_500(sim_app, enable_webrtc: bool):
        """Configure streaming for Isaac Sim 5.0+."""
        if not enable_webrtc:
            return
        from omni.isaac.core.utils.extensions import enable_extension  # type: ignore

        sim_app.set_setting("/app/window/drawMouse", True)
        enable_extension("omni.services.streamclient.webrtc")

    def _build_config_model(self) -> Config:
        merged = copy.deepcopy(self._merged_dict)
        sim_section = merged.setdefault("simulator", {})
        if self._args.headless:
            sim_section["headless"] = True

        # Add extension paths from command line and/or config
        extension_folders = sim_section.get("extension_folders", [])
        if self._args.extension_paths:
            # Extend with CLI-provided paths
            extension_folders.extend(self._args.extension_paths)
        if extension_folders:
            sim_section["extension_folders"] = extension_folders

        config_model = _parse_config_model(merged)
        return config_model

    def _create_runner_with_app(self, config: Config) -> SimulatorRunner:
        """Create the SimulatorRunner, reusing the SimulationApp we already initialized."""
        task_manager = create_task_config_manager(config)
        if self._simulation_app is None:
            raise RuntimeError("SimulationApp must be initialized before creating SimulatorRunner.")

        # Temporarily replace setup_isaacsim so the SimulatorRunner reuses our SimulationApp.
        original_setup = SimulatorRunner.setup_isaacsim

        def _reuse_setup(runner_self):
            # Assign the private _simulation_app directly; there is no property setter.
            runner_self._simulation_app = self._simulation_app
            runner_self._simulation_app._carb_settings.set("/physics/cooking/ujitsoCollisionCooking", False)
            self._reuse_streaming_configuration(runner_self)

        SimulatorRunner.setup_isaacsim = _reuse_setup
        try:
            runner = SimulatorRunner(config=config, task_config_manager=task_manager)
        finally:
            SimulatorRunner.setup_isaacsim = original_setup

        return runner

    def _reuse_streaming_configuration(self, runner: SimulatorRunner):
        """Configure streaming for reused SimulationApp (Isaac Sim 5.0+)."""
        native = getattr(runner.config.simulator, "native", False)
        webrtc = getattr(runner.config.simulator, "webrtc", False)

        if native:
            from OmniNav.core.util import log
            log.warning("native streaming is deprecated, enabling webrtc instead")

        runner.setup_streaming_500(native or webrtc)

    def _post_runner_initialize(self):
        # Import Isaac Sim modules (can now be safely imported)
        from OmniNavExt.envset.world_utils import bootstrap_world_if_needed
        from OmniNavExt.envset.agent_manager import AgentManager
        from OmniNavExt.envset.patches import install_safe_simtimes_guard

        bootstrap_world_if_needed()
        AgentManager.get_instance()
        if not getattr(self, "_rec_enabled", False):
            install_safe_simtimes_guard()

    def _init_data_generation(self):
        """Initialize DataGeneration for recording simulation data."""
        import carb  # type: ignore

        try:
            from OmniNavExt.data_generation.data_generation import DataGeneration
        except ImportError as e:
            carb.log_error(f"[EnvsetStandalone] Failed to import DataGeneration: {e}")
            return

        self._data_gen = DataGeneration()

        # Get data generation config from envset scenario or use defaults
        scenario = self._bundle.scenario
        data_gen_cfg = scenario.get("data_generation") or {}

        # Set writer name and params
        self._data_gen.writer_name = data_gen_cfg.get("writer", "BasicWriter")
        self._data_gen.writer_params = data_gen_cfg.get("writer_params") or {
            "output_dir": "_out_envset",
            "rgb": True,
            "semantic_segmentation": False,
        }

        # Set number of frames (default to 300 if not specified)
        self._data_gen._num_frames = data_gen_cfg.get("num_frames", 300)

        # Camera path list (empty means auto-detect from stage)
        self._data_gen._camera_path_list = data_gen_cfg.get("camera_paths") or []

        carb.log_info(
            f"[EnvsetStandalone] DataGeneration initialized: writer={self._data_gen.writer_name}, "
            f"frames={self._data_gen._num_frames}"
        )

    def _run_data_generation(self):
        """Run data generation asynchronously."""
        import carb  # type: ignore

        if self._data_gen is None:
            carb.log_error("[EnvsetStandalone] DataGeneration not initialized")
            return

        carb.log_info("[EnvsetStandalone] Starting data generation...")
        try:
            asyncio.run(self._data_gen.run_async(will_wait_until_complete=True))
            carb.log_info("[EnvsetStandalone] Data generation completed successfully")
        except Exception as e:
            carb.log_error(f"[EnvsetStandalone] Data generation failed: {e}")
            import traceback
            carb.log_error(traceback.format_exc())
            self._main_loop()

    def _wait_for_initialization(self):
        import carb  # type: ignore
        from omni.isaac.core.simulation_context import SimulationContext  # type: ignore

        try:
            world = self._runner._world if hasattr(self._runner, '_world') else None
            if not world:
                carb.log_warn("[EnvsetStandalone] World not available, skipping initialization wait")
                return

            # Physics warm-up: 30 steps to settle physics state (aligned with datagen)
            for i in range(30):
                try:
                    world.step(render=False)
                except Exception as e:
                    print(f"[EnvsetStandalone] Physics step {i+1} failed: {e}")

            # Render warm-up: 3 steps to flush stale camera buffers (aligned with datagen)
            for i in range(3):
                try:
                    SimulationContext.render(world)
                except Exception as e:
                    print(f"[EnvsetStandalone] Render step {i+1} failed: {e}")

            print("[EnvsetStandalone] Initialization completed (30 physics + 3 render steps)")

        except Exception as e:
            print(f"[EnvsetStandalone] Initialization wait failed: {e}, continuing anyway")

    def _fix_physics_properties_if_needed(self):
        """
        Remove ALL physics properties (RigidBodyAPI, CollisionAPI, etc.) from static objects
        to prevent them from falling. This method is called after scene loading but before timeline starts.

        For GRScenes, we remove all physics APIs from static objects (non-articulations),
        which completely prevents them from being affected by physics simulation.
        """
        import omni.usd  # type: ignore
        from pxr import Usd, UsdPhysics, PhysxSchema  # type: ignore

        scenario = self._bundle.scenario
        scene_cfg = scenario.get("scene") or {}

        # Use the same detection logic as simulation.py
        is_grscenes = False

        # Check category field first
        category = scene_cfg.get("category")
        if category:
            category_str = str(category).strip().lower()
            if "grscenes" in category_str:
                is_grscenes = True

        # Fallback: check usd_path
        if not is_grscenes:
            usd_path = scene_cfg.get("usd_path") or ""
            if isinstance(usd_path, str) and "grscene" in usd_path.lower():
                is_grscenes = True

        # Fallback: check navmesh_root_prim_path
        if not is_grscenes:
            navmesh_root = scene_cfg.get("navmesh_root_prim_path")
            if navmesh_root and str(navmesh_root).startswith("/Root"):
                is_grscenes = True

        if not is_grscenes:
            print("[EnvsetStandalone] No specific physics fixes needed for this scene type")
            return

        print("[EnvsetStandalone] GRScenes detected, removing RigidBodyAPI from static objects...")

        stage = omni.usd.get_context().get_stage()
        if not stage:
            print("[EnvsetStandalone] Warning: Stage is invalid, cannot remove physics properties")
            return

        # Find scene root path
        scene_root = self._find_actual_scene_root(stage, scenario)
        if not scene_root:
            # Try default paths
            for env_id in range(10):
                candidate = f"/World/env_{env_id}/scene"
                prim = stage.GetPrimAtPath(candidate)
                if prim and prim.IsValid():
                    scene_root = candidate
                    break

        if not scene_root:
            print("[EnvsetStandalone] Warning: Could not find scene root, skipping physics fixes")
            return

        print(f"[EnvsetStandalone] Found scene root: {scene_root}")

        # Traverse scene and remove RigidBodyAPI from static objects
        root_prim = stage.GetPrimAtPath(scene_root)
        if not root_prim or not root_prim.IsValid():
            print(f"[EnvsetStandalone] Warning: Scene root prim is invalid: {scene_root}")
            return

        removed_rigid_count = 0
        skipped_count = 0

        # Traverse all prims in the scene
        for prim in Usd.PrimRange(root_prim):
            # Skip if prim is not valid or inactive
            if not prim.IsValid() or not prim.IsActive():
                continue

            # Check if prim is an articulation (has joints) - these should keep physics
            is_articulation = False
            try:
                # Check if this prim or its children have joints
                for child in Usd.PrimRange(prim):
                    if child.GetTypeName() in ["PhysicsJoint", "PhysicsRevoluteJoint",
                                                 "PhysicsPrismaticJoint", "PhysicsFixedJoint"]:
                        is_articulation = True
                        break
            except Exception:
                pass

            # Skip articulations - they need physics
            if is_articulation:
                skipped_count += 1
                continue

            # Remove RigidBodyAPI only
            try:
                # Remove RigidBodyAPI
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    removed_rigid_count += 1

                # Remove rigidBodyEnabled attribute if it exists
                if prim.HasAttribute('physics:rigidBodyEnabled'):
                    prim.RemoveProperty('physics:rigidBodyEnabled')

            except Exception as e:
                print(f"[EnvsetStandalone] Warning: Failed to remove RigidBodyAPI from {prim.GetPath()}: {e}")
                continue

        print(f"[EnvsetStandalone] Physics fixes completed: {removed_rigid_count} RigidBodyAPI removed, "
              f"{skipped_count} articulations kept physics")

    def _start_timeline(self):
        """
        Start the timeline after all scene loading and physics fixes are complete.
        This ensures that the simulation starts with all objects properly configured.
        """
        import omni.timeline  # type: ignore

        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            timeline.play()
            print("[EnvsetStandalone] Timeline started successfully")
        else:
            print("[EnvsetStandalone] Timeline is already playing")

    def _on_timeline_first_start(self):
        """One-shot init that runs the first time the timeline starts.

        Mirrors the auto-start pipeline: virtual-human init -> physics/render warm-up -> capture init.
        """
        if self._timeline_init_done:
            return

        import carb  # type: ignore

        if self._navmesh_success:
            from OmniNavExt.envset.runtime_hooks import EnvsetTaskRuntime
            scenario = self._bundle.scenario
            EnvsetTaskRuntime.initialize_virtual_humans(scenario)
            EnvsetTaskRuntime.register_robots_as_dynamic_obstacles(self._runner)
            carb.log_info("[EnvsetStandalone] Virtual humans initialized")

        # physics + render warm-up (aligned with datagen: 30 + 3)
        self._wait_for_initialization()

        if self._rec_enabled:
            self._init_capture()
            print("[EnvsetStandalone] camera recording initialized (datagen-aligned)")

        self._timeline_init_done = True
        carb.log_info("[EnvsetStandalone] Timeline first start initialization completed")

    def _init_capture(self):
        """Initialize datagen-aligned capture: IsaacsimCamera + MultiCameraAsyncVideoWriter."""
        import carb  # type: ignore
        from bench.replay.replay_runner import ReplayRunner
        from bench.utils.visualizer import MultiCameraAsyncVideoWriter

        world = self._runner._world if self._runner else None
        if world is None:
            raise RuntimeError("World not available for capture init.")
        self._capture_dt = float(world.get_physics_dt())

        # Resolve first robot for pose extraction
        task = list(self._runner.current_tasks.values())[0]
        robot_name = list(task.robots.keys())[0]
        robot = task.robots[robot_name]
        self._capture_robot = robot
        self._capture_robot_name = robot_name

        # Resolve cameras from robot sensors (same as datagen)
        # _unwrap_camera_sensor is a plain method (no @staticmethod) that only takes `sensor`,
        # so we wrap it in a lambda to absorb the implicit `self` argument.
        replay_cfg_stub = type("_Stub", (), {"num_cameras": 1})()
        replay_stub = type("_Stub", (), {
            "config": replay_cfg_stub,
            "_normalize_num_cameras": lambda self: 1,
            "_unwrap_camera_sensor": lambda self, sensor: getattr(sensor, "_camera", None) or sensor,
        })()
        self._replay_cameras = ReplayRunner._resolve_replay_cameras(replay_stub, robot)
        camera_names = [cam.name for cam in self._replay_cameras]
        carb.log_info(f"[Record] resolved {len(self._replay_cameras)} camera(s): {camera_names}")

        # Ensure output dirs
        if self._rec_rgb_video_path is None:
            self._resolve_record_dirs()

        # Build camera_outputs dict for MultiCameraAsyncVideoWriter
        video_dir = self._rec_base_dir
        if len(self._replay_cameras) == 1:
            camera_outputs = {
                camera_names[0]: {
                    "rgb": video_dir / "rgb.mp4",
                    "depth": video_dir / "depth.mp4",
                }
            }
        else:
            camera_outputs = {
                name: {
                    "rgb": video_dir / name / "rgb.mp4",
                    "depth": video_dir / name / "depth.mp4",
                }
                for name in camera_names
            }

        self._video_writer = MultiCameraAsyncVideoWriter(
            camera_outputs=camera_outputs,
            fps=int(self._rec_fps),
            recording_json_path=None,
            recording_instruction=None,
        )

        # Direct WaypointRecorder (no JSONL intermediate)
        self._waypoint_recorder = WaypointRecorder()

        # Timer-based FPS control
        self._capture_elapsed = 0.0
        self._capture_next = 0.0
        self._capture_frame_idx = 0

        carb.log_info(
            f"[Record] init ok: dt={self._capture_dt:.6f}s fps={self._rec_fps} "
            f"video_dir={video_dir}"
        )

    def _capture_frame(self):
        """Capture one frame from all cameras and record pose (datagen-aligned)."""
        from bench.replay.replay_runner import ReplayRunner

        if self._replay_cameras is None or self._video_writer is None:
            return

        frames = {}
        for replay_camera in self._replay_cameras:
            camera = replay_camera.camera
            rgba = camera.get_rgba()
            if rgba is None:
                raise RuntimeError(f"Camera '{replay_camera.name}'.get_rgba() returned None")
            rgb = rgba[:, :, :3]
            depth = None
            if hasattr(camera, "get_distance_to_image_plane"):
                try:
                    depth = camera.get_distance_to_image_plane()
                except Exception:
                    depth = None
            frames[replay_camera.name] = (rgb, depth)

        # Build metadata (same format as ReplayRunner._build_video_frame_metadata)
        x, y, z, yaw = ReplayRunner._get_robot_pose_xyyaw(self._capture_robot)
        metadata = {
            "frame": int(self._capture_frame_idx),
            "sim_step": int(self._sim_step_count),
            "timestamp": float(time.time()),
            "timestamp_ms": int(time.time() * 1000.0),
            "sim_time_s": float(self._capture_elapsed),
            "pose": {"x": float(x), "y": float(y), "z": float(z), "yaw": float(yaw)},
        }

        self._video_writer.push(self._capture_frame_idx, frames, metadata=metadata)

        # Direct WaypointRecorder (same as datagen pipeline)
        self._waypoint_recorder.add_sample(
            frame=int(self._capture_frame_idx),
            sim_step=int(self._sim_step_count),
            time_s=float(self._capture_elapsed),
            xyz=(float(x), float(y), float(z)),
            yaw_rad=float(yaw),
        )

        self._capture_frame_idx += 1

    def _stop_capture(self):
        """Close video writer and finalize path.json."""
        if self._video_writer is not None:
            try:
                self._video_writer.close()
            except Exception:
                pass
            self._video_writer = None

        if self._waypoint_recorder is not None and self._rec_pose_log_path is not None:
            path_entries = self._waypoint_recorder.build()
            metadata = {
                "source": "standalone",
                "distance_threshold_xy": 0.0,
            }
            payload = build_recording_payload(
                instruction="",
                gt_path=path_entries,
                metadata=metadata,
            )
            self._rec_pose_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._rec_pose_log_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
            )
            print(f"[EnvsetStandalone] Recording saved: {self._rec_pose_log_path}")
            self._waypoint_recorder = None

        self._replay_cameras = None

    @staticmethod
    def _find_actual_scene_root(stage, scenario):
        """Resolve the runtime scene root (e.g., /World/<scenario_id>/scene)."""
        if stage is None:
            return None
        # Read scenario id from the scene config (historical fields: id / scenario_id).
        scenario_id = None
        if scenario:
            scenario_id = scenario.get("id") or scenario.get("scenario_id")
        if scenario_id:
            # 1) Prefer the standard /World/scene layout.
            candidate_path = f"/World/scene"
            prim = stage.GetPrimAtPath(candidate_path)
            if prim and prim.IsValid():
                return candidate_path
            # 2) If there is no /World/scene child, fall back to scanning /World's children
            #    for a prim named "scene".
            scenario_prim = stage.GetPrimAtPath(f"/World")
            if scenario_prim and scenario_prim.IsValid():
                for child in scenario_prim.GetChildren():
                    if child.GetName() == "scene" and child.IsValid():
                        return str(child.GetPath())
        # 3) Without a scenario id, walk the default env_{i} layout.
        for env_id in range(10):
            candidate_path = f"/World/env_{env_id}/scene"
            prim = stage.GetPrimAtPath(candidate_path)
            if prim and prim.IsValid():
                return candidate_path
        return None

    @staticmethod
    def _map_to_stage_path(target_path, actual_scene_root, configured_scene_root):
        """Map a configured prim path into the loaded stage hierarchy."""
        if not target_path:
            return None
        try:
            target_path = str(target_path)
        except Exception:
            return None
        # Case A: target is a relative path; re-anchor it under the actual scene root.
        if not target_path.startswith("/"):
            if actual_scene_root:
                return f"{actual_scene_root}/{target_path.lstrip('/')}"
            return f"/World/{target_path.lstrip('/')}"
        # Case B: target begins with the configured scene root; swap in the runtime scene root.
        if actual_scene_root and configured_scene_root and target_path.startswith(configured_scene_root):
            relative = target_path[len(configured_scene_root):].lstrip("/")
            return f"{actual_scene_root}/{relative}" if relative else actual_scene_root
        # Case C: legacy fallback (e.g. /World/Root/...). If the first segment matches,
        # rebuild the relative path beneath the actual scene root.
        if actual_scene_root and configured_scene_root and configured_scene_root != "/" and target_path.startswith("/"):
            configured_parts = configured_scene_root.strip("/").split("/")
            target_parts = target_path.strip("/").split("/")
            if configured_parts and target_parts and configured_parts[0] == target_parts[0]:
                relative = "/".join(target_parts[1:])
                return f"{actual_scene_root}/{relative}" if relative else actual_scene_root
        return target_path

    def _detect_keyboard_control(self):
        """Detect if any robot requires keyboard control."""
        import carb  # type: ignore

        scenario = self._bundle.scenario
        robots_cfg = scenario.get("robots", {})
        robot_entries = robots_cfg.get("entries", [])

        keyboard_robots = []
        for robot in robot_entries:
            control = robot.get("control", {})
            control_mode = (control.get("mode") or "").lower()
            robot_name = robot.get("label") or robot.get("type", "unknown")

            if "keyboard" in control_mode:
                robot_type = (robot.get("type") or "").lower()
                controller_name = self._get_controller_name_for_robot(robot_type)
                if controller_name:
                    keyboard_robots.append({
                        "name": robot_name,
                        "controller": controller_name,
                        "type": robot_type,
                        "max_lin_vel": 1.0,
                        "max_ang_vel": 1.0,
                    })
                    carb.log_info(f"[EnvsetStandalone] Keyboard control enabled for robot: {robot_name}")

        return keyboard_robots

    def _get_controller_name_for_robot(self, robot_type: str) -> str | None:
        """Map robot type to its base controller name."""
        # Differential drive robots
        if robot_type in {"carter", "carter_v1", "jetbot", "differential_drive", "kitt15"}:
            return "move_by_speed"
        # Legged robots
        if robot_type in {"aliengo", "h1", "g1", "gr1", "human"}:
            return "move_by_speed"
        # Unknown type
        return None

    def _init_keyboard(self):
        """Initialize keyboard interaction if needed."""
        import carb  # type: ignore

        self._keyboard_robots = self._detect_keyboard_control()
        self._sync_keyboard_robot_profiles()

        if self._keyboard_robots:
            try:
                from OmniNavExt.interactions.keyboard import KeyboardInteraction
                self._keyboard = KeyboardInteraction()
                carb.log_info(f"[EnvsetStandalone] Keyboard control initialized for {len(self._keyboard_robots)} robot(s)")
                print("[EnvsetStandalone] Keyboard control ready - Diff-drive uses I/K (forward/back), J/L (rotate); U/O kept as rotate aliases")
            except ImportError as e:
                carb.log_error(f"[EnvsetStandalone] Failed to import KeyboardInteraction: {e}")
                self._keyboard = None
                self._keyboard_robots = []

    def _sync_keyboard_robot_profiles(self):
        """Populate keyboard-controlled robots with execution-profile velocity scales."""
        import carb  # type: ignore

        if not self._runner or not self._keyboard_robots:
            return

        runtime_robots = {}
        for task in self._runner.current_tasks.values():
            robots = getattr(task, "robots", {}) or {}
            for robot_name, robot in robots.items():
                runtime_robots[robot_name] = robot

        for robot_cfg in self._keyboard_robots:
            robot_name = robot_cfg["name"]
            robot = runtime_robots.get(robot_name)
            if robot is None:
                carb.log_warn(f"[EnvsetStandalone] Keyboard robot '{robot_name}' not found in current tasks")
                continue

            if not hasattr(robot, "get_execution_profile"):
                continue

            try:
                profile = robot.get_execution_profile()
            except Exception as exc:
                carb.log_warn(f"[EnvsetStandalone] Failed to get execution profile for '{robot_name}': {exc}")
                continue

            max_lin_vel = getattr(profile, "max_lin_vel", None)
            max_ang_vel = getattr(profile, "max_ang_vel", None)
            if max_lin_vel is not None:
                robot_cfg["max_lin_vel"] = float(max_lin_vel)
            if max_ang_vel is not None:
                robot_cfg["max_ang_vel"] = float(max_ang_vel)

            carb.log_info(
                f"[EnvsetStandalone] Keyboard profile for '{robot_name}': "
                f"max_lin_vel={robot_cfg['max_lin_vel']:.3f}, max_ang_vel={robot_cfg['max_ang_vel']:.3f}"
            )

    def _collect_actions(self):
        """Collect actions from keyboard or return empty actions for autonomous mode."""
        if not self._keyboard or not self._keyboard_robots:
            # Autonomous mode: return empty actions for all envs
            return [{}] * self._runner.env_num

        # Read keyboard input
        command = self._keyboard.get_input()
        forward_axis = float(command[0] - command[1])  # I/K keys
        turn_axis_primary = float(command[2] - command[3])  # J/L keys
        turn_axis_alias = float(command[4] - command[5])  # U/O keys
        turn_axis = max(-1.0, min(1.0, turn_axis_primary + turn_axis_alias))

        # Build actions for all keyboard-controlled robots
        actions = []
        for env_id in range(self._runner.env_num):
            env_action = {}
            for robot_cfg in self._keyboard_robots:
                robot_name = robot_cfg["name"]
                controller_name = robot_cfg["controller"]
                robot_type = robot_cfg["type"]

                if robot_type in {"carter", "carter_v1", "jetbot", "differential_drive"}:
                    controller_action = (
                        forward_axis * float(robot_cfg.get("max_lin_vel", 1.0)),
                        0.0,
                        turn_axis * float(robot_cfg.get("max_ang_vel", 1.0)),
                    )
                else:
                    y_speed = turn_axis_primary
                    z_speed = turn_axis_alias
                    controller_action = (
                        forward_axis,
                        y_speed,
                        z_speed,
                    )

                env_action[robot_name] = {
                    controller_name: controller_action
                }
            actions.append(env_action)

        return actions

    def _wait_for_articulations_initialized(self, max_wait_frames: int = 100):
        """Wait for all articulations to initialize and observe NavMesh baking status."""
        import carb  # type: ignore
        from omni.isaac.core.simulation_context import SimulationContext  # type: ignore

        if not self._runner:
            return False

        world = self._runner._world if hasattr(self._runner, '_world') else None
        if not world:
            carb.log_warn("[EnvsetStandalone] World not available, cannot wait for articulations")
            return False

        navmesh_interface = None
        try:
            import omni.anim.navigation.core as nav  # type: ignore
            navmesh_interface = nav.acquire_interface()
        except Exception:
            pass

        carb.log_info("[EnvsetStandalone] Waiting for all articulations to initialize...")
        if navmesh_interface:
            carb.log_info("[EnvsetStandalone] Also checking NavMesh baking status...")

        navmesh_ready = False

        for frame_idx in range(max_wait_frames):
            # Render one frame so physics has a chance to initialize.
            SimulationContext.render(world)

            all_initialized = True
            uninitialized_robots = []

            for task_name, task in self._runner.current_tasks.items():
                if not hasattr(task, 'robots') or not task.robots:
                    continue

                for robot_name, robot in task.robots.items():
                    if not hasattr(robot, 'articulation'):
                        continue

                    if not hasattr(robot.articulation, 'handles_initialized'):
                        continue

                    if not robot.articulation.handles_initialized:
                        all_initialized = False
                        uninitialized_robots.append(f"{task_name}/{robot_name}")

            navmesh_status_msg = ""
            if navmesh_interface:
                try:
                    navmesh = navmesh_interface.get_navmesh()
                    if navmesh is not None:
                        if not navmesh_ready:
                            navmesh_ready = True
                            try:
                                area_count = navmesh.get_area_count()
                                navmesh_status_msg = f", NavMesh ready (areas={area_count})"
                            except Exception:
                                navmesh_status_msg = ", NavMesh ready"
                    else:
                        navmesh_status_msg = ", NavMesh baking..."
                except Exception:
                    navmesh_status_msg = ", NavMesh status unknown"

            if all_initialized:
                if navmesh_ready:
                    carb.log_info(
                        f"[EnvsetStandalone] All articulations initialized and NavMesh ready after {frame_idx + 1} frames"
                    )
                else:
                    carb.log_info(
                        f"[EnvsetStandalone] All articulations initialized after {frame_idx + 1} frames"
                        f"{navmesh_status_msg}"
                    )
                return True

            # Print status every 10 frames.
            if frame_idx % 10 == 0:
                status_parts = []
                if uninitialized_robots:
                    status_parts.append(
                        f"Uninitialized robots: {', '.join(uninitialized_robots[:5])}"
                        f"{'...' if len(uninitialized_robots) > 5 else ''}"
                    )
                if navmesh_status_msg:
                    status_parts.append(navmesh_status_msg.strip(', '))

                status_str = " | ".join(status_parts) if status_parts else "Waiting..."
                carb.log_info(
                    f"[EnvsetStandalone] Waiting... ({frame_idx + 1}/{max_wait_frames} frames) | {status_str}"
                )

        final_status = []
        if uninitialized_robots:
            final_status.append(f"Some articulations not initialized: {', '.join(uninitialized_robots[:3])}")
        if navmesh_interface:
            try:
                navmesh = navmesh_interface.get_navmesh()
                if navmesh is None:
                    final_status.append("NavMesh not ready")
            except Exception:
                pass

        if final_status:
            carb.log_warn(
                f"[EnvsetStandalone] Timeout after {max_wait_frames} frames. "
                f"{' | '.join(final_status)}. Continuing anyway, but errors may occur."
            )
        else:
            carb.log_warn(
                f"[EnvsetStandalone] Timeout after {max_wait_frames} frames. "
                f"Continuing anyway, but errors may occur."
            )
        return False

    def _are_articulations_ready(self) -> bool:
        """Quick check that every articulation has finished initializing."""
        if not self._runner:
            return False

        for task_name, task in self._runner.current_tasks.items():
            if not hasattr(task, 'robots') or not task.robots:
                continue

            for robot_name, robot in task.robots.items():
                if not hasattr(robot, 'articulation'):
                    continue

                if not hasattr(robot.articulation, 'handles_initialized'):
                    continue

                if not robot.articulation.handles_initialized:
                    return False

        return True

    @staticmethod
    def _set_camera_light() -> None:
        """Switch viewport lighting to camera light for Matterport scenes."""
        try:
            import omni.kit.actions.core as actions  # type: ignore
        except Exception as exc:
            raise RuntimeError("Failed to import omni.kit.actions.core for camera light control") from exc

        registry = actions.get_action_registry()
        action = registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera")
        if action is None:
            raise RuntimeError("Viewport lighting action not found; ensure lighting extension is enabled.")
        action.execute()

    def _main_loop(self):
        import carb  # type: ignore

        sim_app = self._runner.simulation_app if self._runner else None
        if sim_app is None:
            return

        # Initialize keyboard control if needed
        self._init_keyboard()

        # Check if timeline is already playing and wait for articulation initialization
        import omni.timeline  # type: ignore
        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing():
            carb.log_info("[EnvsetStandalone] Timeline already playing, waiting for articulations...")
            self._wait_for_articulations_initialized()
            for _ in range(5):
                sim_app.update()
            self._print_runtime_snapshot("After timeline auto-started")

        deadline = None
        if self._args.hold_seconds is not None:
            deadline = time.monotonic() + max(0.0, self._args.hold_seconds)

        print(f"[EnvsetStandalone] Main loop started (keyboard={'enabled' if self._keyboard else 'disabled'})")

        timeline_was_playing = timeline.is_playing()

        while sim_app.is_running() and not self._shutdown_flag:
            if deadline is not None and time.monotonic() >= deadline:
                break

            # Check timeline status changes
            timeline_is_playing = timeline.is_playing()
            if timeline_is_playing and not timeline_was_playing:
                carb.log_info("[EnvsetStandalone] Timeline started by user")
                # Run the full first-start init (mirrors the auto-start path).
                self._on_timeline_first_start()
                self._wait_for_articulations_initialized()
                timeline_was_playing = True
            elif not timeline_is_playing and timeline_was_playing:
                self._stop_capture()
                carb.log_info("[EnvsetStandalone] Timeline stopped, recording closed")
                timeline_was_playing = False
            elif not timeline_is_playing:
                timeline_was_playing = False

            # Skip step() if articulations not ready or timeline not playing
            if timeline_is_playing:
                if not self._are_articulations_ready():
                    sim_app.update()
                    continue

            if not timeline_is_playing:
                sim_app.update()
                continue

            # Collect actions and step
            actions = self._collect_actions()
            try:
                self._runner.step(actions=actions, render=True)
                self._sim_step_count += 1

                # Timer-based capture (aligned with datagen)
                if self._rec_enabled and self._replay_cameras is not None:
                    fps_interval = 1.0 / float(self._rec_fps)
                    self._capture_elapsed += self._capture_dt
                    if self._capture_elapsed >= self._capture_next - 1e-9:
                        self._capture_frame()
                        self._capture_next += fps_interval

            except Exception as e:
                if "Failed to get root link transforms" in str(e) or "handles_initialized" in str(e):
                    for _ in range(5):
                        sim_app.update()
                    if not self._are_articulations_ready():
                        sim_app.update()
                        continue
                else:
                    carb.log_error(f"[EnvsetStandalone] Error in runner.step(): {e}")
                    sim_app.update()


def main():
    args = _parse_args()
    if not args.config.expanduser().exists():
        raise SystemExit(f"Config file not found: {args.config}")
    if not args.envset.expanduser().exists():
        raise SystemExit(f"Envset file not found: {args.envset}")

    runner = EnvsetStandaloneRunner(args)
    try:
        runner.run()
    except KeyboardInterrupt:
        print("[EnvsetStandalone] Interrupted by user")
        runner.request_shutdown()
    except Exception as e:
        print(f"[EnvsetStandalone] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        runner.shutdown()


if __name__ == "__main__":
    main()
