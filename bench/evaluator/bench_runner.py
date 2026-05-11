# pyright: reportMissingImports=false

"""Batch benchmark runner for VLN evaluation."""

from __future__ import annotations

import json
import time
import importlib
import threading
import queue
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from OmniNavExt.envset.recording import (
    resolve_recording_dirs,
    resolve_recording_waypoints,
)
from bench.utils.visualizer import Visualizer

from ..policy.base import BasePolicy
from .episode_runner import EpisodeRunner, EpisodeConfig, EpisodeResult
from .termination import StuckCondition, TimeoutCondition, TerminationCondition
from bench.configs.execution import ExecutionConfig
from bench.execution.policy_modes import default_policy_mode_map

if TYPE_CHECKING:
    from OmniNav.core.runner import SimulatorRunner


@dataclass
class BenchConfig:
    """Configuration for batch benchmark run.

    Attributes:
        uninav_config: Path to OmniNav YAML config
        envset_path: Path to envset JSON file or directory containing multiple JSON files
        output_dir: Directory for results output
        scene_root: Base directory for envset file paths
        scenario_ids: Specific scenario IDs to run (None for all)
        headless: Run in headless mode
        timeout_multiplier: Multiplier for expert path time/frames to compute timeout
        success_threshold: Default goal distance threshold
        record_trajectory: Whether to record full trajectories
        save_per_episode: Save results after each episode
        sort_by_scene: Sort scenarios by scene path to maximize scene reuse
        skip_completed: Skip scenarios with existing output files
        record_video: Whether to record video during evaluation
        video_fps: Video frame rate
        save_depth_video: Whether to save depth video
        record_images: Whether to save RGB still images during evaluation
        image_interval_s: Sim-time interval between saved RGB still images
        policy_name: Name of the policy (for subprocess mode)
        policy_args: Additional policy arguments (for subprocess mode)
    """
    uninav_config: Path
    envset_path: Path
    output_dir: Path
    scene_root: Optional[Path] = None
    scenario_ids: Optional[List[str]] = None
    headless: bool = True
    timeout_multiplier: float = 2.0
    success_threshold: float = 1.0
    record_trajectory: bool = True
    save_per_episode: bool = True
    sort_by_scene: bool = True
    skip_completed: bool = True
    record_video: bool = False
    video_fps: int = 30
    save_depth_video: bool = True
    record_images: bool = False
    image_interval_s: float = 1.0
    policy_name: str = "forward"
    policy_args: Dict[str, Any] = field(default_factory=dict)
    isolate_episodes: bool = False  # Run each episode in a separate process


@dataclass
class BenchResult:
    """Aggregated benchmark results.

    Scoring fields (success rate, SPL, etc.) are intentionally absent: scoring
    is performed offline via :mod:`bench.evaluator.offline_test` against
    private GT, not by the runtime.

    Attributes:
        results: List of per-episode results
        avg_steps: Average steps per episode
        avg_time_s: Average time per episode
        total_time_s: Total benchmark runtime
    """
    results: List[EpisodeResult] = field(default_factory=list)
    avg_steps: float = 0.0
    avg_time_s: float = 0.0
    total_time_s: float = 0.0


@dataclass(frozen=True)
class ScenarioRef:
    """Reference to a scenario for grouping and execution.

    Attributes:
        source_envset_path: Path to the source envset JSON file
        scenario: The scenario dict from envset
        scenario_id: Unique identifier for this scenario
        is_matterport: Whether this is a Matterport scene
    """
    source_envset_path: Path
    scenario: Dict[str, Any]
    scenario_id: str
    is_matterport: bool

    @property
    def unique_key(self) -> str:
        """Generate a unique key combining scenario_id and source file path.

        This is necessary because multiple scenarios from different envset files
        can have the same scenario_id.
        """
        return f"{self.scenario_id}:{self.source_envset_path}"


class BenchRunner:
    """Runs batch benchmark evaluation across multiple scenarios.

    Each scenario restarts SimulationApp for clean state.
    """

    def __init__(self, config: BenchConfig, policy: BasePolicy):
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration
            policy: Navigation policy to evaluate
        """
        self.config = config
        self.policy = policy
        self._results: List[EpisodeResult] = []

    def run(self) -> BenchResult:
        """Execute full benchmark with scene reuse optimization.

        Scenarios are grouped strictly by their parent folder (group_folder) to reuse
        SimulationApp and NavMesh within each group. Multiple groups require separate
        processes due to Isaac Sim limitation (SimulationApp can only be initialized once).

        Returns:
            BenchResult with aggregated metrics
        """
        start_time = time.monotonic()

        # Load and group scenarios
        scenario_refs = self._load_scenario_refs()
        self._log(f"[BenchRunner] Loaded {len(scenario_refs)} scenarios")

        if not scenario_refs:
            self._log("[BenchRunner] No scenarios to run")
            return BenchResult(total_time_s=0.0)

        groups = self._group_scenarios(scenario_refs)
        self._log(f"[BenchRunner] Grouped into {len(groups)} folder group(s)")

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Run scenarios
        if len(groups) > 1:
            # Multiple scene groups require separate processes
            self._log(f"[BenchRunner] Detected {len(groups)} folder groups, using multiprocess mode")
            self._run_groups_multiprocess(groups)
        else:
            # Single group can run in current process
            for idx, (group_dir, group_refs) in enumerate(groups):
                self._log(
                    f"[BenchRunner] Group {idx + 1}/{len(groups)}: "
                    f"group_dir={group_dir} scenarios={len(group_refs)}"
                )
                results = self._run_scenario_group(group_refs)
                self._results.extend(results)

        # Compute aggregated metrics
        total_time = time.monotonic() - start_time
        bench_result = self._compute_aggregates(total_time)

        # Save final results
        self._save_final_results(bench_result)
        self._log(f"[BenchRunner] Saved summary to {self.config.output_dir / 'summary.json'}")

        return bench_result

    def run_legacy(self) -> BenchResult:
        """Execute full benchmark (legacy mode without scene reuse).

        This method is kept for backward compatibility and debugging.

        Returns:
            BenchResult with aggregated metrics
        """
        start_time = time.monotonic()

        # Load scenarios
        scenarios = self._load_scenarios()
        print(f"[BenchRunner] Loaded {len(scenarios)} scenarios")

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Run each scenario
        for idx, scenario in enumerate(scenarios):
            scenario_id = scenario.get("id", f"scenario_{idx}")
            print(f"\n[BenchRunner] Running scenario {idx + 1}/{len(scenarios)}: {scenario_id}")

            try:
                print("run_scenario")
                result = self._run_scenario(scenario)
                print("run_end")
                try:
                    self._results.append(result)
                except Exception as e:
                    print(f"[BenchRunner] ERROR appending result: {e}")

                print(f"[BenchRunner] Scenario {scenario_id}: steps={result.steps}, time_s={result.time_s:.2f}")

            except Exception as e:
                print(f"[BenchRunner] ERROR in scenario {scenario_id}: {e}")
                import traceback
                traceback.print_exc()

        # Compute aggregated metrics
        try:
            total_time = time.monotonic() - start_time
        except Exception as e:
            print(f"[BenchRunner] ERROR computing total time: {e}")
        try:
            bench_result = self._compute_aggregates(total_time)
        except Exception as e:
            print(f"[BenchRunner] ERROR computing aggregates: {e}")
        # Save final results
        try:
            self._save_final_results(bench_result)
        except Exception as e:
            print(f"[BenchRunner] ERROR saving final results: {e}")
        print(f"[BenchRunner] Saved summary to {self.config.output_dir / 'summary.json'}")

        return bench_result

    def _load_scenarios(self) -> List[Dict[str, Any]]:
        """Load scenarios from envset JSON file(s).

        Supports:
        - Single JSON file
        - Directory containing multiple JSON files (root/*.json + root/*/*.json)

        If sort_by_scene is True, scenarios are sorted by scene path
        to maximize scene reuse between consecutive episodes.
        """
        scenarios = []

        if self.config.envset_path.is_file():
            # Single file
            scenarios = self._load_scenarios_from_file(self.config.envset_path)
        elif self.config.envset_path.is_dir():
            # Directory: load all JSON files (matches replay scan: root/*.json + root/*/*.json)
            root = self.config.envset_path
            json_files = []
            json_files.extend(sorted(p for p in root.glob("*.json") if not p.name.endswith(".backup.json")))
            json_files.extend(sorted(p for p in root.glob("*/*.json") if not p.name.endswith(".backup.json")))
            if not json_files:
                raise FileNotFoundError(f"No JSON files found in: {self.config.envset_path}")
            for json_file in json_files:
                scenarios.extend(self._load_scenarios_from_file(json_file))
        else:
            raise FileNotFoundError(f"Envset path not found: {self.config.envset_path}")

        # Filter by scenario_ids if specified
        if self.config.scenario_ids:
            id_set = set(self.config.scenario_ids)
            scenarios = [s for s in scenarios if s.get("id") in id_set]

        # Sort by scene path to maximize scene reuse
        if self.config.sort_by_scene and scenarios:
            scenarios = self._sort_scenarios_by_scene(scenarios)

        return scenarios

    def _load_scenarios_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load scenarios from a single JSON file."""
        with file_path.open("r", encoding="utf-8") as f:
            envset = json.load(f)
        return envset.get("scenarios", [])

    def _sort_scenarios_by_scene(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort scenarios by scene path to group same scenes together."""
        def get_scene_key(s: Dict[str, Any]) -> str:
            scene = s.get("scene", {})
            return scene.get("usd_path", "") or scene.get("asset_path", "")

        return sorted(scenarios, key=get_scene_key)

    def _run_scenario(self, scenario: Dict[str, Any]) -> EpisodeResult:
        """Run single scenario with fresh SimulationApp.

        This method handles the full lifecycle:
        1. Initialize SimulationApp
        2. Load scene and setup
        3. Run episode
        4. Shutdown SimulationApp
        """
        # Import here to avoid loading Isaac Sim at module level
        from OmniNavExt.envset.config_loader import EnvsetConfigLoader
        from OmniNavExt.envset.core import (
            SimulationBootstrap,
            SimulationConfig,
            PhysicsManager,
            NavMeshManager,
        )
        from OmniNavExt.envset.core.scene_manager import (
            SceneManager,
            find_scene_root,
            is_matterport_scenario,
            should_use_camera_light,
        )
        from OmniNav.core.config import Config
        from OmniNav.core.runner import SimulatorRunner
        from OmniNav.core.task_config_manager.base import create_task_config_manager

        scenario_id = scenario.get("id", "unknown")
        robot_name = self._get_robot_name(scenario)

        # Normalize envset file paths before scenario use.
        EnvsetConfigLoader.normalize_scenario_paths(scenario, self.config.scene_root)
        # Build episode config from scenario
        episode_config = self._build_episode_config(scenario, source_envset_path=self.config.envset_path)
        if episode_config is None:
            raise ValueError(f"Scenario '{scenario_id}' should be skipped (single task parser returned None)")

        # Print episode configuration
        print(f"[BenchRunner] Episode config:")
        print(f"  Start:  {episode_config.start_position}")
        print(f"  Goal:   {episode_config.goal_position}")
        print(f"  Max steps: {episode_config.max_steps}")
        print(f"  Success threshold: {episode_config.success_threshold}m")

        # Build termination conditions
        termination_conditions = self._build_termination_conditions(episode_config)

        # Load and merge config
        bundle = EnvsetConfigLoader(
            config_path=self.config.uninav_config,
            envset_path=self.config.envset_path,
            scenario_id=scenario_id,
            scene_root=self.config.scene_root,
        ).load()

        # Apply headless setting
        merged_config = bundle.config
        sim_section = merged_config.setdefault("simulator", {})
        sim_section["headless"] = self.config.headless

        # Apply per-policy robot sensor configuration (if available) before validating config model.
        self._apply_policy_robot_config(merged_config, robot_name, episode_config)

        # Parse config model
        try:
            config_model = Config.model_validate(merged_config)
        except AttributeError:
            config_model = Config.parse_obj(merged_config)

        # Re-apply policy sensor configuration to the validated model
        # (Pydantic may recreate RobotCfg objects during validation, discarding earlier modifications)
        self._apply_policy_robot_config_to_model(config_model, robot_name)

        # Initialize SimulationApp
        scenario_id = scenario.get("id", "unknown")
        print(f"[BenchRunner] Initializing SimulationApp for {scenario_id}")

        sim_config = SimulationConfig(headless=self.config.headless)
        sim_bootstrap = SimulationBootstrap(sim_config)
        simulation_app = sim_bootstrap.initialize()

        try:
            # Import extensions after SimulationApp init
            from OmniNavExt import import_extensions
            import_extensions()

            # Create SimulatorRunner with reused app
            task_manager = create_task_config_manager(config_model)
            print(f"[BenchRunner] Creating SimulatorRunner for {scenario_id}")
            runner = self._create_runner_with_app(config_model, task_manager, simulation_app)

            # Initialize world
            from OmniNavExt.envset.world_utils import bootstrap_world_if_needed
            bootstrap_world_if_needed()

            # Load scene
            scene_cfg = scenario.get("scene", {})
            navmesh_cfg = scenario.get("navmesh", {})
            from OmniNavExt.envset.runtime_hooks import EnvsetTaskRuntime
            EnvsetTaskRuntime.reset_navmesh_cache()

            # Matterport/MP3D: explicit import + navmesh preparation
            is_matterport = is_matterport_scenario(scene_cfg)
            use_camera_light = should_use_camera_light(scene_cfg)
            if is_matterport:
                print(f"[BenchRunner] Importing Matterport scene for {scenario_id}...")
                matterport_prim = SceneManager.import_matterport_scene(
                    scene_cfg, self.config.scene_root
                )
                SceneManager.prepare_matterport_navmesh(
                    matterport_prim, navmesh_cfg, scene_cfg, simulation_app, envset_cfg=scenario
                )

            scene_root_path = scenario.get("scene", {}).get("root_prim_path", "/World")

            def _collision_hook(stage):
                self._apply_large_mesh_convex_hull_collision(stage, scene_root_path, log_prefix="[FIX-Hook]")

            runner._pre_physics_hook = _collision_hook
            print(f"[BenchRunner] Resetting runner for {scenario_id}")
            runner.reset(start_timeline=False)

            # Policy-specific post-reset hook (e.g. align multi-camera poses).
            self._run_policy_post_reset_hook(runner, robot_name)

            # Fix physics for GRScenes
            scene_root = find_scene_root(runner._stage, scenario)
            PhysicsManager.fix_grscenes_physics(scene_cfg, scene_root)

            # Exclude robots from NavMesh baking
            for task_name, task in runner.current_tasks.items():
                if hasattr(task, 'robots') and task.robots:
                    for robot_name, robot in task.robots.items():
                        if hasattr(robot, 'config') and hasattr(robot.config, 'prim_path'):
                            robot_prim_path = robot.config.prim_path
                            EnvsetTaskRuntime._exclude_from_navmesh(robot_prim_path)
                            print(f"[BenchRunner] Excluded robot '{robot_name}' from NavMesh: {robot_prim_path}")

            print(f"[BenchRunner] Baking NavMesh for {scenario_id}")
            navmesh_manager = NavMeshManager.from_scenario(navmesh_cfg, scene_cfg, scene_root)
            navmesh_manager.bake_sync(simulation_app, envset_cfg=scenario)
            # Disable NavMesh debug visualization (persistent setting, otherwise
            # it re-opens on next launch).
            import omni.kit.commands
            omni.kit.commands.execute(
                "ChangeSetting",
                path="/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh",
                value=False,
            )

            # Start timeline
            import omni.timeline
            timeline = omni.timeline.get_timeline_interface()
            timeline.play()

            # Initialize virtual humans (after NavMesh is ready)
            from OmniNavExt.envset.runtime_hooks import EnvsetTaskRuntime
            EnvsetTaskRuntime.initialize_virtual_humans(scenario)
            # Register robots as dynamic obstacles so virtual humans can avoid them.
            EnvsetTaskRuntime.register_robots_as_dynamic_obstacles(runner)

            # Wait for initialization
            PhysicsManager.wait_for_articulations(runner, max_frames=50)

            # Reset robot pose using the same method as runReplay for consistency.
            try:
                task = list(runner.current_tasks.values())[0]
                target_robot = task.robots.get(robot_name)
                if target_robot is not None:
                    self._snap_robot_to_initial_pose(scenario, target_robot)
            except Exception as exc:
                print(f"[BenchRunner] Set initial pose failed: {exc}")

            # Robot needs a few physics steps after timeline start to settle into the initial pose.
            print(f"[BenchRunner] Stabilizing robot pose with physics warmup")
            runner.warm_up(steps=20, render=True, physics=True)

            # For Matterport scenes, switch viewport lighting to camera light so that
            # the main camera drives the lighting (better for navigation tasks).
            if use_camera_light:
                self._set_camera_light()
                if self._CAMERA_LIGHT_WARMUP_STEPS > 0:
                    runner.warm_up(steps=self._CAMERA_LIGHT_WARMUP_STEPS, render=True, physics=True)

            # Create episode runner and execute
            print(f"[BenchRunner] Starting EpisodeRunner for {scenario_id}")
            execution_config = ExecutionConfig(policy_mode_map=default_policy_mode_map())
            robot_profile = self._get_robot_execution_profile(runner, scenario)
            episode_runner = EpisodeRunner(
                policy=self.policy,
                termination_conditions=termination_conditions,
                record_trajectory=self.config.record_trajectory,
                execution_config=execution_config,
                robot_profile=robot_profile,
            )

            result = episode_runner.run(runner, episode_config, robot_name)
            print(f"[BenchRunner] Episode finished:")
            print(f"  Reason:  {result.termination_reason}")
            print(f"  Steps:   {result.steps}")
            print(f"  Time:    {result.time_s:.2f}s")

            if self.config.save_per_episode:
                self._save_episode_result(result)
                print(f"[BenchRunner] Saved episode result for {result.scenario_id}")

            return result

        finally:
            print(f"[BenchRunner] Shutting down SimulationApp for {scenario_id}")
            sim_bootstrap.shutdown()

    def _create_runner_with_app(self, config, task_manager, simulation_app) -> "SimulatorRunner":
        """Create SimulatorRunner reusing existing SimulationApp."""
        from OmniNav.core.runner import SimulatorRunner

        # Temporarily replace setup_isaacsim
        original_setup = SimulatorRunner.setup_isaacsim

        def _reuse_setup(runner_self):
            runner_self._simulation_app = simulation_app
            runner_self._simulation_app._carb_settings.set(
                "/physics/cooking/ujitsoCollisionCooking", False
            )

        SimulatorRunner.setup_isaacsim = _reuse_setup
        try:
            runner = SimulatorRunner(config=config, task_config_manager=task_manager)
        finally:
            SimulatorRunner.setup_isaacsim = original_setup

        return runner

    def _get_policy_key(self) -> Optional[str]:
        """Return policy key derived from policy module path: bench.policy.<key>.*"""
        module = getattr(self.policy.__class__, "__module__", "") or ""
        parts = module.split(".")
        if len(parts) >= 3 and parts[0] == "bench" and parts[1] == "policy":
            return parts[2]
        return None

    def _import_policy_robot_config(self):
        key = self._get_policy_key()
        if key is None:
            return None
        module_name = f"bench.policy.{key}.robot_config"
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # Only ignore if the missing module is exactly the robot_config module.
            if exc.name == module_name:
                return None
            raise

    @staticmethod
    def _find_robot_cfg_in_merged_config(merged_config: Dict[str, Any], robot_name: str):
        tasks = merged_config.get("task_configs") or []
        if not isinstance(tasks, list):
            raise ValueError("merged_config.task_configs must be a list")
        for task in tasks:
            if not isinstance(task, dict):
                continue
            robots = task.get("robots") or []
            if not isinstance(robots, list):
                continue
            for robot_cfg in robots:
                if getattr(robot_cfg, "name", None) == robot_name:
                    return robot_cfg
        raise RuntimeError(f"RobotCfg '{robot_name}' not found in merged_config.task_configs[].robots")

    def _apply_policy_robot_config(
        self,
        merged_config: Dict[str, Any],
        robot_name: str,
        episode_config: EpisodeConfig,
    ) -> None:
        """Apply policy-local robot sensor configuration, and set EpisodeConfig extras.
        
        Note: This modifies merged_config before Config.model_validate(). However,
        Pydantic may recreate objects during validation. Use _apply_policy_robot_config_to_model()
        after validation to ensure sensors are properly set on the final config_model.
        """
        module = self._import_policy_robot_config()
        if module is None:
            return

        robot_cfg = self._find_robot_cfg_in_merged_config(merged_config, robot_name)

        configure = getattr(module, "configure_robot_sensors", None)
        if callable(configure):
            configure(robot_cfg)

        get_required = getattr(module, "get_required_cameras", None)
        if callable(get_required):
            required = get_required(robot_cfg)
            if episode_config.extra is None:
                episode_config.extra = {}
            episode_config.extra["required_cameras"] = required

    def _apply_policy_robot_config_to_model(
        self,
        config_model,
        robot_name: str,
    ) -> None:
        """Apply policy sensor configuration to the validated Config model.
        
        This must be called AFTER Config.model_validate() because Pydantic may
        recreate RobotCfg objects during validation, discarding earlier modifications.
        """
        module = self._import_policy_robot_config()
        if module is None:
            return

        configure = getattr(module, "configure_robot_sensors", None)
        if not callable(configure):
            return

        # Find robot_cfg in config_model.task_configs[].robots
        for task_cfg in (config_model.task_configs or []):
            robots = getattr(task_cfg, "robots", None) or []
            for robot_cfg in robots:
                if getattr(robot_cfg, "name", None) == robot_name:
                    configure(robot_cfg)
                    print(f"[BenchRunner] Applied policy sensor config to robot '{robot_name}': "
                          f"sensors={[s.name for s in (robot_cfg.sensors or [])]}")
                    return

    def _run_policy_post_reset_hook(self, runner: "SimulatorRunner", robot_name: str) -> None:
        module = self._import_policy_robot_config()
        if module is None:
            return
        hook = getattr(module, "post_reset", None)
        if callable(hook):
            hook(runner, robot_name)

    def _apply_policy_required_cameras(
        self,
        episode_config: EpisodeConfig,
        robot_cfg,
    ) -> None:
        """Apply policy-required camera names to EpisodeConfig.extra."""
        module = self._import_policy_robot_config()
        if module is None:
            return
        get_required = getattr(module, "get_required_cameras", None)
        if not callable(get_required):
            return
        required = get_required(robot_cfg)
        if episode_config.extra is None:
            episode_config.extra = {}
        episode_config.extra["required_cameras"] = required

    def _build_episode_config(
        self,
        scenario: Dict[str, Any],
        *,
        source_envset_path: Optional[Path] = None,
    ) -> Optional[EpisodeConfig]:
        """Build EpisodeConfig from scenario data.

        Returns:
            EpisodeConfig if valid, None if episode should be skipped (single mode only).
        """
        scenario_id = scenario.get("id", "unknown")

        scene_cfg = scenario.get("scene", {}) if isinstance(scenario.get("scene"), dict) else {}
        units_in_meters = scene_cfg.get("units_in_meters")
        try:
            env_unit_scale = float(units_in_meters) if units_in_meters is not None else 1.0
        except (TypeError, ValueError):
            env_unit_scale = 1.0

        def _scale_xyz(value):
            if not (isinstance(value, (list, tuple)) and len(value) >= 3):
                return value
            try:
                return [
                    float(value[0]) * env_unit_scale,
                    float(value[1]) * env_unit_scale,
                    float(value[2]) * env_unit_scale,
                ]
            except (TypeError, ValueError):
                return value

        task_cfg = scenario.get("task", {}) if isinstance(scenario.get("task"), dict) else {}
        nav_cfg = task_cfg.get("navigation") if isinstance(task_cfg.get("navigation"), dict) else None

        instruction = ""
        goal = (0, 0, 0)

        if isinstance(nav_cfg, dict):
            instruction = nav_cfg.get("instruction", "")
            goal = _scale_xyz(nav_cfg.get("goal_position", (0, 0, 0)))
        else:
            instruction = task_cfg.get("instruction", "")
            legacy_goal = task_cfg.get("goal") if isinstance(task_cfg.get("goal"), dict) else {}
            goal = _scale_xyz(legacy_goal.get("position", (0, 0, 0)))

        if isinstance(goal, (list, tuple)) and len(goal) >= 3:
            goal_position = (float(goal[0]), float(goal[1]), float(goal[2]))
        else:
            goal_position = (0.0, 0.0, 0.0)

        # Extract start position and timeout from robot config
        robots_cfg = scenario.get("robots", {})
        entries = robots_cfg.get("entries", [])
        start_position = None
        expert_time_s: float = 0.0
        expert_frames: int = 0
        
        if not entries:
            raise ValueError(f"Scenario '{scenario_id}' has no robot entries in 'robots.entries'")
        
        first_robot = entries[0]
        initial_pose = first_robot.get("initial_pose", {})
        pos = initial_pose.get("position")
        if pos and len(pos) >= 3:
            scaled = _scale_xyz(pos)
            start_position = (float(scaled[0]), float(scaled[1]), float(scaled[2]))
        
        # Prefer canonical recording.gt_path; fall back to legacy rb_gt_waypoints for older envsets.
        expert_waypoints = resolve_recording_waypoints(scenario, envset_path=source_envset_path)
        if not expert_waypoints:
            raise ValueError(
                f"Scenario '{scenario_id}' is missing standard recording.gt_path and legacy rb_gt_waypoints. "
                "This field is required for computing timeout limits."
            )

        last_waypoint = expert_waypoints[-1]
        if not isinstance(last_waypoint, dict):
            raise ValueError(
                f"Scenario '{scenario_id}': last expert waypoint is not a valid dict"
            )
        
        expert_time_s = float(last_waypoint.get("time_s", 0.0))
        expert_frames = int(last_waypoint.get("frame", 0))
        
        if expert_time_s <= 0 and expert_frames <= 0:
            raise ValueError(
                f"Scenario '{scenario_id}': last waypoint has invalid time_s={expert_time_s} "
                f"and frame={expert_frames}. At least one must be positive."
            )
        
        multiplier = self.config.timeout_multiplier
        max_steps = int(expert_frames * multiplier) if expert_frames > 0 else 10000  # fallback

        extra: Dict[str, Any] = {}

        goal_radius_used = float(self.config.success_threshold)
        extra["goal_radius"] = goal_radius_used

        waypoints = [goal_position] if start_position is None else [start_position, goal_position]
        extra["shortest_path"] = self._estimate_shortest_path(waypoints)

        # Forward navigation fields from scenario to extra (consumed by _build_measure_setup).
        if isinstance(nav_cfg, dict):
            nav_objects = nav_cfg.get("objects")
            nav_room_zone = nav_cfg.get("room_zone")

            if nav_objects is not None:
                if isinstance(nav_objects, dict):
                    scaled_objects = {}
                    for name, value in nav_objects.items():
                        scaled_objects[name] = _scale_xyz(value) if isinstance(value, (list, tuple)) else value
                    extra["objects"] = scaled_objects
                else:
                    extra["objects"] = nav_objects

            if nav_room_zone:
                if isinstance(nav_room_zone, dict):
                    scaled_zones = {}
                    for zone_name, zone in nav_room_zone.items():
                        if not isinstance(zone, dict):
                            scaled_zones[zone_name] = zone
                            continue
                        scaled = dict(zone)
                        if "aabb_min" in scaled:
                            scaled["aabb_min"] = _scale_xyz(scaled.get("aabb_min"))
                        if "aabb_max" in scaled:
                            scaled["aabb_max"] = _scale_xyz(scaled.get("aabb_max"))
                        scaled_zones[zone_name] = scaled
                    extra["room_zone"] = scaled_zones
                else:
                    extra["room_zone"] = nav_room_zone

            # `answer` is required by EQA scoring.
            nav_answer = nav_cfg.get("answer")
            if nav_answer is not None:
                extra["answer"] = nav_answer

        # Forward subtasks for offline metric computation.
        subtasks = task_cfg.get("subtasks")
        if subtasks is not None:
            extra["subtasks"] = subtasks

        # Forward qa entries for EQA scoring.
        qa_info = task_cfg.get("qa")
        if qa_info is not None:
            extra["qa"] = qa_info

        success_threshold = goal_radius_used

        extra["expert_time_s"] = expert_time_s
        extra["expert_frames"] = expert_frames
        extra["timeout_multiplier"] = multiplier

        return EpisodeConfig(
            scenario_id=scenario_id,
            instruction=instruction,
            goal_position=goal_position,
            start_position=start_position,
            max_steps=max_steps,
            success_threshold=success_threshold,
            extra=extra,
        )

    def _build_termination_conditions(self, config: EpisodeConfig) -> List[TerminationCondition]:
        """Build termination conditions for episode."""
        extra = config.extra or {}
        expert_time_s = float(extra.get("expert_time_s", 0.0) or 0.0)
        multiplier = float(extra.get("timeout_multiplier", self.config.timeout_multiplier) or self.config.timeout_multiplier)
        wall_timeout_s = None
        if expert_time_s > 0:
            # Wall-clock guard is intentionally generous; step limit remains the benchmark timeout.
            wall_timeout_s = max(300.0, expert_time_s * multiplier * 5.0)

        return [
            StuckCondition(duration_s=60.0, move_threshold_m=0.1),
            TimeoutCondition(
                max_steps=config.max_steps,
                max_wall_time_s=wall_timeout_s,
            ),
        ]

    def _get_robot_name(self, scenario: Dict[str, Any]) -> str:
        """Get robot name from scenario."""
        robots_cfg = scenario.get("robots", {})
        entries = robots_cfg.get("entries", [])
        if entries:
            return entries[0].get("label", "robot")
        return "robot"

    def _set_camera_light(self) -> None:
        """Switch Kit viewport lighting mode to 'camera'.

        This uses the omni.kit.actions.core API to trigger the same action as
        the UI menu: Viewport → Lighting → Camera Light. It must be called
        after SimulationApp and the viewport have been initialized.

        Raises:
            RuntimeError: if the Kit actions API or the lighting action is not available.
        """
        try:
            import omni.kit.actions.core as actions  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Failed to import omni.kit.actions.core for camera light control") from exc

        registry = actions.get_action_registry()
        action = registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera")
        if action is None:
            raise RuntimeError(
                "Kit action 'omni.kit.viewport.menubar.lighting.set_lighting_mode_camera' not found. "
                "Ensure the viewport lighting extension is enabled."
            )
        action.execute()

    def _get_robot_execution_profile(self, runner: "SimulatorRunner", scenario: Dict[str, Any]):
        """Fetch execution profile from robot if provided."""
        robot_name = self._get_robot_name(scenario)
        for task in runner.current_tasks.values():
            robots = getattr(task, "robots", {}) or {}
            if robot_name in robots:
                robot = robots[robot_name]
                if hasattr(robot, "get_execution_profile"):
                    profile = robot.get_execution_profile()
                    module = self._import_policy_robot_config()
                    override = getattr(module, "get_execution_profile_override", None) if module else None
                    if callable(override):
                        try:
                            profile = override(profile)
                            print(f"[BenchRunner] Applied policy execution profile override for '{robot_name}'")
                            print(f"[BenchRunner] Final execution profile finish_rot_eps_deg: {profile.finish_rot_eps_deg}")
                        except Exception as exc:
                            print(f"[BenchRunner] Failed to apply execution profile override: {exc}")
                    return profile
        raise ValueError(f"Robot execution profile not found for {robot_name}")

    def _get_articulation_z(self, articulation) -> Optional[float]:
        """Return the robot's world-frame Z height (handles API variants)."""
        try:
            pos, _ = articulation.get_world_pose()
            return float(pos[2])
        except Exception:
            try:
                pos_arr, _ = articulation.get_world_poses()
                return float(pos_arr[0][2])
            except Exception:
                return None

    def _snap_robot_to_initial_pose(self, scenario: Dict[str, Any], robot) -> None:
        """Reset using initial_pose only; intentionally ignores waypoints fallback."""
        robots_cfg = scenario.get("robots", {})
        entries = robots_cfg.get("entries", [])
        if not entries:
            return

        first_robot = entries[0]
        initial_pose = first_robot.get("initial_pose", {})
        pos = initial_pose.get("position") or initial_pose.get("xyz")
        yaw_deg = initial_pose.get("orientation_deg")

        if pos is None or yaw_deg is None:
            return

        scene_cfg = scenario.get("scene", {}) if isinstance(scenario.get("scene"), dict) else {}
        units_in_meters = scene_cfg.get("units_in_meters")
        try:
            env_unit_scale = float(units_in_meters) if units_in_meters is not None else 1.0
        except (TypeError, ValueError):
            env_unit_scale = 1.0

        x0 = float(pos[0]) * env_unit_scale
        y0 = float(pos[1]) * env_unit_scale
        z0 = float(pos[2]) * env_unit_scale if len(pos) >= 3 else None

        yaw_rad = math.radians(float(yaw_deg))
        quat0 = (math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0))

        articulation = getattr(robot, "articulation", None)
        if articulation is None:
            return

        # Prefer the current physics-engine Z to avoid falling through or sinking into the floor.
        z_target = self._get_articulation_z(articulation)
        if z_target is None:
            z_target = z0 if z0 is not None else 0.0

        articulation.set_world_pose((x0, y0, z_target), quat0)

        # Zero velocities explicitly so a previous step's motion does not leak in.
        try:
            articulation.set_linear_velocity(np.zeros(3))
            articulation.set_angular_velocity(np.zeros(3))
            articulation.set_joint_velocities(np.zeros(articulation.num_dof))
        except Exception as e:
            print(f"[BenchRunner] Warning: failed to reset velocities: {e}")

    # =========================================================================
    # Scenario Grouping Methods (for scene reuse)
    # =========================================================================

    def _group_scenarios(self, scenario_refs: List[ScenarioRef]) -> List[Tuple[Path, List[ScenarioRef]]]:
        """Group scenarios.

        If isolate_episodes is True, creates a separate group for each scenario.
        Otherwise, groups them by parent folder to maximize scene reuse.
        """
        if self.config.isolate_episodes:
            # Force one scenario per group to trigger multiprocess isolation
            return [(ref.source_envset_path.parent, [ref]) for ref in scenario_refs]

        groups: Dict[Path, List[ScenarioRef]] = {}
        ordered_keys: List[Path] = []
        for ref in scenario_refs:
            key = ref.source_envset_path.parent
            if key not in groups:
                groups[key] = []
                ordered_keys.append(key)
            groups[key].append(ref)
        return [(key, groups[key]) for key in ordered_keys]

    def _iter_envset_files(self) -> List[Path]:
        """Resolve envset JSON files from envset_path (file or directory).

        Directory scanning: root/*.json + root/*/*.json (same as ReplayRunner).

        Raises:
            FileNotFoundError: If envset_path doesn't exist or contains no JSON files.
        """
        path = self.config.envset_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Envset not found: {path}")

        if path.is_file():
            if path.suffix.lower() != ".json":
                raise ValueError(f"Envset file must be JSON: {path}")
            return [path]

        if not path.is_dir():
            raise FileNotFoundError(f"Envset path is neither file nor directory: {path}")

        # Scan: root/*.json + root/*/*.json
        json_files: List[Path] = []
        json_files.extend(
            sorted(
                p
                for p in path.iterdir()
                if p.is_file() and p.suffix.lower() == ".json" and not p.name.endswith(".backup.json")
            )
        )
        for sub in sorted(p for p in path.iterdir() if p.is_dir()):
            json_files.extend(
                sorted(
                    p
                    for p in sub.iterdir()
                    if p.is_file() and p.suffix.lower() == ".json" and not p.name.endswith(".backup.json")
                )
            )

        if not json_files:
            raise FileNotFoundError(f"No JSON files found in: {path}")
        return json_files

    def _load_scenario_refs(self) -> List[ScenarioRef]:
        """Load all scenarios and build ScenarioRef objects.

        Returns:
            List of ScenarioRef objects for all scenarios to be evaluated.

        Raises:
            ValueError: If any scenario is missing required configuration.
        """
        from OmniNavExt.envset.config_loader import EnvsetConfigLoader
        from OmniNavExt.envset.core.scene_manager import is_matterport_scenario

        scenario_refs: List[ScenarioRef] = []
        id_filter = set(self.config.scenario_ids or [])

        for envset_file in self._iter_envset_files():
            with envset_file.open("r", encoding="utf-8") as f:
                envset = json.load(f)
            scenarios = envset.get("scenarios", [])

            # Filter by scenario IDs if specified
            if id_filter:
                scenarios = [s for s in scenarios if s.get("id") in id_filter]

            for scenario in scenarios:
                if not isinstance(scenario, dict):
                    raise ValueError(f"Invalid scenario entry in {envset_file}: expected dict, got {type(scenario)}")

                # Normalize paths before computing keys
                EnvsetConfigLoader.normalize_scenario_paths(scenario, self.config.scene_root)

                scenario_id = str(scenario.get("id") or "unknown")
                scene_cfg = scenario.get("scene", {}) if isinstance(scenario.get("scene"), dict) else {}
                is_mp = is_matterport_scenario(scene_cfg)

                scenario_refs.append(
                    ScenarioRef(
                        source_envset_path=envset_file,
                        scenario=scenario,
                        scenario_id=scenario_id,
                        is_matterport=is_mp,
                    )
                )

        return scenario_refs

    def _is_scenario_completed(self, scenario_id: str) -> bool:
        """Check if a scenario has already been completed.

        Returns:
            True if skip_completed is enabled and output JSON file exists.
        """
        if not self.config.skip_completed:
            return False
        if not self.config.save_per_episode:
            return False
        output_file = self.config.output_dir / f"{scenario_id}.json"
        return output_file.exists()

    @staticmethod
    def _episode_output_file(ref: ScenarioRef, output_root: Path) -> Path:
        """Per-episode result path under output_root (grouped by envset folder).

        Envset invariant: each episode lives in its own envset JSON file under a group folder,
        and scenario_id can repeat across those files (e.g., Matterport uses a constant id).
        """
        return output_root / ref.source_envset_path.parent.name / f"{ref.source_envset_path.stem}.json"

    def _is_episode_completed(self, ref: ScenarioRef) -> bool:
        """Check if an episode (envset JSON file) has already been completed."""
        if not self.config.skip_completed:
            return False
        if not self.config.save_per_episode:
            return False
        return self._episode_output_file(ref, self.config.output_dir).exists()

    @staticmethod
    def _log(message: str) -> None:
        """Log a message with flush."""
        print(message, flush=True)

    # =========================================================================
    # Scene Reuse Methods (run scenarios grouped by scene/navmesh/robot)
    # =========================================================================

    # Warmup and articulation constants
    _WARMUP_STEPS: int = 20
    _ARTICULATION_WAIT_MAX_FRAMES: int = 50
    _CAMERA_LIGHT_WARMUP_STEPS: int = 5

    def _build_group_config_model(self, group_refs: List[ScenarioRef]):
        """Build a merged Config model for all scenarios in a group.

        Args:
            group_refs: List of ScenarioRef in the same scene group.

        Returns:
            Config model with combined task_configs.

        Raises:
            ValueError: If no scenarios or invalid config structure.
        """
        from OmniNavExt.envset.config_loader import EnvsetConfigLoader
        from OmniNav.core.config import Config

        merged_configs: List[Dict[str, Any]] = []
        for ref in group_refs:
            loader = EnvsetConfigLoader(
                config_path=self.config.uninav_config,
                envset_path=ref.source_envset_path,
                scenario_id=ref.scenario_id,
                scene_root=self.config.scene_root,
            )
            bundle = loader.load()
            merged_config = bundle.config
            merged_config.setdefault("simulator", {})["headless"] = self.config.headless
            merged_configs.append(merged_config)

        if not merged_configs:
            raise ValueError("No scenarios provided for group config")

        base_config = dict(merged_configs[0])
        task_configs: List[Any] = []
        for merged in merged_configs:
            tasks = merged.get("task_configs")
            if not isinstance(tasks, list) or not tasks:
                raise ValueError("Benchmark expects task_configs in merged config")
            if len(tasks) != 1:
                raise ValueError(
                    f"Benchmark expects one task_config per scenario, got {len(tasks)}"
                )
            task_configs.extend(tasks)

        base_config["task_configs"] = task_configs
        base_config.setdefault("simulator", {})["headless"] = self.config.headless

        try:
            config_model = Config.model_validate(base_config)
        except AttributeError:
            config_model = Config.parse_obj(base_config)
        return config_model

    def _setup_first_scenario_navmesh(
        self,
        runner: "SimulatorRunner",
        scenario: Dict[str, Any],
        scenario_id: str,
        scene_cfg: Dict[str, Any],
        navmesh_cfg: Dict[str, Any],
        is_matterport: bool,
        simulation_app,
    ) -> None:
        """Setup NavMesh for the first scenario in a group.

        Args:
            runner: SimulatorRunner instance.
            scenario: Scenario dict.
            scenario_id: Scenario identifier for logging.
            scene_cfg: Scene configuration dict.
            navmesh_cfg: NavMesh configuration dict.
            is_matterport: Whether this is a Matterport scene.
            simulation_app: SimulationApp instance.

        Raises:
            RuntimeError: If NavMesh baking fails.
        """
        from OmniNavExt.envset.core.scene_manager import find_scene_root
        from OmniNavExt.envset.core import PhysicsManager, NavMeshManager
        from OmniNavExt.envset.runtime_hooks import EnvsetTaskRuntime

        self._log(f"[BenchRunner][{scenario_id}] Step: find_scene_root + fix_grscenes_physics")
        scene_root = find_scene_root(runner._stage, scenario)
        self._log(f"[BenchRunner][{scenario_id}] scene_root={scene_root}")
        PhysicsManager.fix_grscenes_physics(scene_cfg, scene_root)

        navmesh_success = True
        if not is_matterport:
            self._log(f"[BenchRunner][{scenario_id}] Step: exclude robots from NavMesh")
            for task_name, task in runner.current_tasks.items():
                robots = getattr(task, "robots", {}) or {}
                self._log(f"[BenchRunner][{scenario_id}] task={task_name} robots={list(robots.keys())}")
                for robot_name, robot in robots.items():
                    if hasattr(robot, "config") and hasattr(robot.config, "prim_path"):
                        prim_path = robot.config.prim_path
                        self._log(f"[BenchRunner][{scenario_id}]   - exclude {robot_name} prim={prim_path}")
                        EnvsetTaskRuntime._exclude_from_navmesh(prim_path)

            self._log(f"[BenchRunner][{scenario_id}] Step: bake NavMesh")
            navmesh_manager = NavMeshManager.from_scenario(navmesh_cfg, scene_cfg, scene_root)
            navmesh_success = navmesh_manager.bake_sync(simulation_app, envset_cfg=scenario)
            self._log(f"[BenchRunner][{scenario_id}] NavMesh success={navmesh_success}")

        if not navmesh_success:
            raise RuntimeError(f"NavMesh baking failed for scenario {scenario_id}")

        self._log(f"[BenchRunner][{scenario_id}] Step: disable NavMesh visualization")
        import omni.kit.commands
        omni.kit.commands.execute(
            "ChangeSetting",
            path="/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh",
            value=False,
        )

    @staticmethod
    def _apply_large_mesh_convex_hull_collision(stage, scene_root_path: str, *, log_prefix: str = "[FIX]") -> None:
        from pxr import PhysxSchema, UsdPhysics, UsdGeom, Usd

        def _has_points(mesh_prim):
            pts = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
            return pts is not None and len(pts) > 0

        scene_prim = stage.GetPrimAtPath(scene_root_path)
        if not scene_prim or not scene_prim.IsValid():
            print(f"{log_prefix} Scene root not found: {scene_root_path}")
            return

        target_roots = {}
        for prim in Usd.PrimRange(scene_prim):
            path_str = str(prim.GetPath())
            if "/robots/" in path_str:
                continue
            if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(PhysxSchema.PhysxMeshMergeCollisionAPI):
                target_roots[path_str] = prim

        converted_count = 0
        skipped_small = 0
        skipped_empty = 0
        seen_meshes = set()
        root_count = 0
        for root in target_roots.values():
            root_count += 1
            for child in Usd.PrimRange(root):
                if child == root:
                    continue
                child_path = str(child.GetPath())
                if "/robots/" in child_path or child_path in seen_meshes:
                    continue
                if not child.IsA(UsdGeom.Mesh):
                    continue
                if not _has_points(child):
                    skipped_empty += 1
                    continue
                mesh = UsdGeom.Mesh(child)
                face_counts = mesh.GetFaceVertexCountsAttr().Get()
                num_faces = len(face_counts) if face_counts else 0
                if num_faces < 100000:
                    skipped_small += 1
                    continue
                mesh_api = UsdPhysics.MeshCollisionAPI.Apply(child)
                mesh_api.CreateApproximationAttr().Set("convexHull")
                PhysxSchema.PhysxConvexHullCollisionAPI.Apply(child)
                converted_count += 1
                seen_meshes.add(child_path)
                print(f"{log_prefix} {child.GetPath()} faces={num_faces} -> convexHull")
        print(
            f"{log_prefix} Total: {converted_count} large mesh(es) converted from {root_count} collision root(s), "
            f"skipped {skipped_small} small mesh(es), skipped {skipped_empty} empty mesh(es) "
            f"under {scene_root_path}"
        )

    def _run_single_scenario_in_group(
        self,
        runner: "SimulatorRunner",
        ref: ScenarioRef,
        is_first_scenario: bool,
        is_matterport: bool,
        scene_cfg: Dict[str, Any],
        navmesh_cfg: Dict[str, Any],
        simulation_app,
    ) -> EpisodeResult:
        """Run a single scenario within a scene group.

        This method handles the episode lifecycle without reinitializing SimulationApp.

        Args:
            runner: SimulatorRunner instance (reused across scenarios).
            ref: ScenarioRef for this scenario.
            is_first_scenario: Whether this is the first scenario (needs NavMesh setup).
            is_matterport: Whether this is a Matterport scene.
            scene_cfg: Scene configuration dict.
            navmesh_cfg: NavMesh configuration dict.
            simulation_app: SimulationApp instance.

        Returns:
            EpisodeResult for this scenario.
        """
        from OmniNavExt.envset.runtime_hooks import EnvsetTaskRuntime
        from OmniNavExt.envset.core import PhysicsManager
        from bench.configs.execution import ExecutionConfig
        import omni.timeline

        scenario = ref.scenario
        scenario_id = ref.scenario_id
        robot_name = self._get_robot_name(scenario)

        self._log(f"[BenchRunner][{scenario_id}] Begin scenario in group")

        # 1. Timeline management
        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing():
            self._log(f"[BenchRunner][{scenario_id}] Step: timeline.pause before reset")
            timeline.pause()

        # 2. Reset episode state
        EnvsetTaskRuntime.reset_episode_state(stage=runner._stage)

        # 3. Runner reset
        scene_root_path = scenario.get("scene", {}).get("root_prim_path", "/World")

        def _collision_hook(stage):
            self._apply_large_mesh_convex_hull_collision(stage, scene_root_path, log_prefix="[FIX-Hook]")

        runner._pre_physics_hook = _collision_hook
        self._log(f"[BenchRunner][{scenario_id}] Step: runner.reset(start_timeline=False)")
        runner.reset(start_timeline=False)

        # Policy-specific post-reset hook
        self._run_policy_post_reset_hook(runner, robot_name)

        # 4. First scenario: NavMesh setup
        if is_first_scenario:
            self._setup_first_scenario_navmesh(
                runner, scenario, scenario_id, scene_cfg, navmesh_cfg, is_matterport, simulation_app
            )

        # 5. Start timeline
        self._log(f"[BenchRunner][{scenario_id}] Step: timeline.play")
        timeline.play()

        vh_cfg = scenario.get("virtual_humans", {}) if isinstance(scenario.get("virtual_humans"), dict) else {}
        defer_vh_routes = bool(vh_cfg.get("defer_routes", True))

        # 6. Virtual humans initialization
        EnvsetTaskRuntime.reconcile_virtual_humans(
            scenario,
            stage=runner._stage,
            defer_routes=defer_vh_routes,
        )

        # 7. Register robots as dynamic obstacles
        EnvsetTaskRuntime.register_robots_as_dynamic_obstacles(runner)

        # 8. Wait for articulations
        articulations_ready = PhysicsManager.wait_for_articulations(
            runner, max_frames=self._ARTICULATION_WAIT_MAX_FRAMES
        )
        self._log(f"[BenchRunner][{scenario_id}] articulations_ready={articulations_ready}")

        # Reset robot pose using the same method as runReplay for consistency.
        try:
            task = list(runner.current_tasks.values())[0]
            target_robot = task.robots.get(robot_name)
            if target_robot is not None:
                self._snap_robot_to_initial_pose(scenario, target_robot)
        except Exception as exc:
            self._log(f"[BenchRunner][{scenario_id}] Set initial pose failed: {exc}")

        # 9. Warmup
        self._log(f"[BenchRunner][{scenario_id}] Step: warmup ({self._WARMUP_STEPS} steps)")
        runner.warm_up(steps=self._WARMUP_STEPS, render=True, physics=True)

        # 10. Inject virtual human routes after warmup only when deferred
        if defer_vh_routes:
            EnvsetTaskRuntime._setup_virtual_routes(scenario)

        # 11. Set camera light when scene requests viewport camera lighting
        from OmniNavExt.envset.core.scene_manager import should_use_camera_light
        if should_use_camera_light(scene_cfg):
            self._set_camera_light()
            if self._CAMERA_LIGHT_WARMUP_STEPS > 0:
                runner.warm_up(steps=self._CAMERA_LIGHT_WARMUP_STEPS, render=True, physics=True)

        # Build episode config
        episode_config = self._build_episode_config(scenario, source_envset_path=self.config.envset_path)
        if episode_config is None:
            raise ValueError(f"Scenario '{scenario_id}' should be skipped (single task parser returned None)")
        self._log(f"[BenchRunner][{scenario_id}] Episode config:")
        self._log(f"  Start:  {episode_config.start_position}")
        self._log(f"  Goal:   {episode_config.goal_position}")
        self._log(f"  Max steps: {episode_config.max_steps}")
        self._log(f"  Success threshold: {episode_config.success_threshold}m")

        # Apply policy-required camera names for group runs (EpisodeRunner uses these).
        try:
            task = list(runner.current_tasks.values())[0]
            target_robot = task.robots.get(robot_name) if hasattr(task, "robots") else None
            if target_robot is not None and hasattr(target_robot, "config"):
                self._apply_policy_required_cameras(episode_config, target_robot.config)
        except Exception as exc:
            self._log(f"[BenchRunner][{scenario_id}] Apply required cameras failed: {exc}")

        # Build termination conditions
        termination_conditions = self._build_termination_conditions(episode_config)

        # 12. Initialize video recording (Visualizer) if enabled
        visualizer: Optional[Visualizer] = None
        if self.config.record_video or self.config.record_images:
            episode_output_file = self._episode_output_file(ref, self.config.output_dir)
            video_root, path_root = resolve_recording_dirs(episode_output_file)
            camera_dir = video_root / "front"
            camera_dir.mkdir(parents=True, exist_ok=True)
            path_root.mkdir(parents=True, exist_ok=True)

            rgb_path = camera_dir / "rgb.mp4" if self.config.record_video else None
            depth_path = camera_dir / "depth.mp4" if self.config.save_depth_video else None
            if not self.config.record_video:
                depth_path = None
            image_dir = camera_dir / "rgb_frames" if self.config.record_images else None

            try:
                # Initialize Visualizer (which includes AsyncVideoWriter)
                # Fail fast if map generation or dependencies are missing
                visualizer = Visualizer(
                    env=runner._world,  # Use OmniNav's world wrapper as env, or potentially runner
                    output_rgb_path=rgb_path,
                    output_depth_path=depth_path,
                    fps=self.config.video_fps,
                    image_output_dir=image_dir,
                    image_interval_s=self.config.image_interval_s,
                    map_resolution=0.05,
                    recording_json_path=path_root / "path.json",
                    recording_instruction=episode_config.instruction,
                )
                outputs: List[str] = []
                if rgb_path is not None:
                    outputs.append(str(rgb_path))
                if image_dir is not None:
                    outputs.append(str(image_dir))
                outputs.append(str(path_root / "path.json"))
                self._log(f"[BenchRunner][{scenario_id}] Visualizer started: {', '.join(outputs)}")
            except Exception as e:
                self._log(f"[BenchRunner][{scenario_id}] FATAL: Failed to initialize Visualizer: {e}")
                # We do not fallback to silent failure - we re-raise because the user explicitly requested video.
                raise

        try:
            # 13. Run EpisodeRunner
            self._log(f"[BenchRunner][{scenario_id}] Starting EpisodeRunner")
            execution_config = ExecutionConfig(policy_mode_map=default_policy_mode_map())
            robot_profile = self._get_robot_execution_profile(runner, scenario)
            episode_runner = EpisodeRunner(
                policy=self.policy,
                termination_conditions=termination_conditions,
                record_trajectory=self.config.record_trajectory,
                execution_config=execution_config,
                robot_profile=robot_profile,
                visualizer=visualizer,
            )

            result = episode_runner.run(runner, episode_config, robot_name)
        finally:
            if visualizer is not None:
                visualizer.close()
                self._log(f"[BenchRunner][{scenario_id}] Visualizer closed")

        self._log(f"[BenchRunner][{scenario_id}] Episode finished:")
        self._log(f"  Reason:  {result.termination_reason}")
        self._log(f"  Steps:   {result.steps}")
        self._log(f"  Time:    {result.time_s:.2f}s")

        if self.config.save_per_episode:
            self._save_episode_result(result, ref=ref)
            self._log(f"[BenchRunner][{scenario_id}] Saved episode result")

        return result

    def _run_scenario_group(
        self,
        group_refs: List[ScenarioRef],
    ) -> List[EpisodeResult]:
        """Run all scenarios in a scene group with shared SimulationApp.

        Args:
            group_refs: List of ScenarioRef in the same scene group.

        Returns:
            List of EpisodeResult for all scenarios in the group.

        Raises:
            RuntimeError: If SimulationApp initialization or NavMesh baking fails.
        """
        from OmniNav.core.task_config_manager.base import create_task_config_manager
        from OmniNavExt import import_extensions
        from OmniNavExt.envset.core import SimulationBootstrap, SimulationConfig
        from OmniNavExt.envset.core.scene_manager import SceneManager
        from OmniNavExt.envset.runtime_hooks import EnvsetTaskRuntime
        import traceback

        if not group_refs:
            return []

        results: List[EpisodeResult] = []

        # Filter out completed scenarios
        pending_refs: List[ScenarioRef] = []
        for ref in group_refs:
            if self._is_episode_completed(ref):
                out_file = self._episode_output_file(ref, self.config.output_dir)
                self._log(f"[BenchRunner][{ref.scenario_id}] SKIP: already completed ({out_file})")
                continue
            pending_refs.append(ref)

        if not pending_refs:
            self._log(f"[BenchRunner][Group] All {len(group_refs)} scenarios completed, skipping")
            return results

        self._log(
            f"[BenchRunner][Group] {len(pending_refs)}/{len(group_refs)} scenarios pending "
            f"({len(group_refs) - len(pending_refs)} skipped)"
        )

        first_ref = pending_refs[0]
        first_scenario = first_ref.scenario
        scene_cfg = first_scenario.get("scene", {}) if isinstance(first_scenario.get("scene"), dict) else {}
        navmesh_cfg = first_scenario.get("navmesh", {}) if isinstance(first_scenario.get("navmesh"), dict) else {}

        self._log(f"[BenchRunner][Group] Begin group with {len(pending_refs)} scenario(s)")
        sim_bootstrap = None

        try:
            config_model = self._build_group_config_model(pending_refs)

            self._log(f"[BenchRunner][Group] Init SimulationApp (headless={self.config.headless})")
            sim_bootstrap = SimulationBootstrap(SimulationConfig(headless=self.config.headless))
            simulation_app = sim_bootstrap.initialize()

            self._log("[BenchRunner][Group] import_extensions")
            import_extensions()

            from OmniNavExt.envset.world_utils import bootstrap_world_if_needed

            # Apply policy sensor configuration to all robots in the group
            # (must be done before creating SimulatorRunner, which triggers create_sensors)
            for ref in pending_refs:
                robot_name = self._get_robot_name(ref.scenario)
                self._apply_policy_robot_config_to_model(config_model, robot_name)

            self._log("[BenchRunner][Group] Create SimulatorRunner")
            task_manager = create_task_config_manager(config_model)
            runner = self._create_runner_with_app(config_model, task_manager, simulation_app)

            self._log("[BenchRunner][Group] bootstrap_world_if_needed")
            bootstrap_world_if_needed()
            EnvsetTaskRuntime.reset_navmesh_cache()

            is_mp = first_ref.is_matterport
            self._log(f"[BenchRunner][Group] is_matterport={is_mp}")
            if is_mp:
                self._log("[BenchRunner][Group] Import matterport scene")
                matterport_prim = SceneManager.import_matterport_scene(scene_cfg, self.config.scene_root)
                self._log(f"[BenchRunner][Group] matterport_prim={matterport_prim}")
                self._log("[BenchRunner][Group] Prepare matterport navmesh")
                SceneManager.prepare_matterport_navmesh(
                    matterport_prim, navmesh_cfg, scene_cfg, simulation_app, envset_cfg=first_scenario
                )

            # Run each pending scenario
            first_scenario_run = False
            for idx, ref in enumerate(pending_refs):
                self._log(f"\n[BenchRunner] Running scenario {idx + 1}/{len(pending_refs)}: {ref.scenario_id}")

                try:
                    result = self._run_single_scenario_in_group(
                        runner=runner,
                        ref=ref,
                        is_first_scenario=(not first_scenario_run),
                        is_matterport=is_mp,
                        scene_cfg=scene_cfg,
                        navmesh_cfg=navmesh_cfg,
                        simulation_app=simulation_app,
                    )
                    results.append(result)
                    first_scenario_run = True

                except Exception as e:
                    self._log(f"[BenchRunner][{ref.scenario_id}] ERROR: {e}")
                    self._log(traceback.format_exc())
                    # Continue to next scenario instead of failing the entire group
                    continue

        except Exception as e:
            self._log(f"[BenchRunner][Group] FATAL: {e}")
            self._log(traceback.format_exc())
            raise

        finally:
            if sim_bootstrap is not None:
                self._log("[BenchRunner][Group] Shutdown SimulationApp")
                sim_bootstrap.shutdown()

        return results

    def _run_groups_multiprocess(
        self,
        groups: List[Tuple[Path, List[ScenarioRef]]],
    ) -> None:
        """Run each scene group in a separate subprocess.

        Isaac Sim's SimulationApp can only be initialized once per Python process.
        When we have multiple folder groups, we spawn a new process for each group.

        Args:
            groups: List of (group_key, scenario_refs) tuples.
        """
        import subprocess
        import sys

        total_groups = len(groups)
        for idx, (group_key, group_refs) in enumerate(groups):
            group_dir = group_key
            scenario_ids = sorted(set(ref.scenario_id for ref in group_refs))

            # Collect unique envset files for this group
            envset_files = list(set(ref.source_envset_path for ref in group_refs))

            self._log(
                f"\n[BenchRunner] Group {idx + 1}/{total_groups}: "
                f"group_dir={group_dir} scenarios={len(group_refs)}"
            )

            # Check if all scenarios in this group are already completed
            all_completed = True
            for ref in group_refs:
                if not self._is_episode_completed(ref):
                    all_completed = False
                    break

            if all_completed:
                self._log(f"[BenchRunner] Group {idx + 1}/{total_groups}: all scenarios completed, skipping")
                continue

            # Build subprocess command
            cmd = self._build_subprocess_cmd(envset_files, scenario_ids)
            self._log(f"[BenchRunner] Subprocess cmd: {' '.join(cmd)}")

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                self._log(f"[BenchRunner] ERROR: Group {idx + 1}/{total_groups} failed: {e}")
                # Continue to next group instead of failing entirely
                continue
            except KeyboardInterrupt:
                self._log("\n[BenchRunner] Interrupted by user")
                raise

        self._log(f"\n[BenchRunner] All {total_groups} groups processed")

        # Load results from saved JSON files for aggregation
        self._load_results_from_output_dir()

    def _build_subprocess_cmd(
        self,
        envset_files: List[Path],
        scenario_ids: List[str],
    ) -> List[str]:
        """Build the subprocess command for running a single group.

        Args:
            envset_files: List of envset file paths for this group.
            scenario_ids: List of scenario IDs to run.

        Returns:
            Command line arguments as list of strings.
        """
        import sys

        # Find runBench.py relative to this file
        # bench_runner.py is at bench/evaluator/bench_runner.py
        # runBench.py is at the repo root
        run_bench_script = Path(__file__).resolve().parent.parent.parent / "runBench.py"

        cmd = [sys.executable, str(run_bench_script)]

        # Required args
        cmd.extend(["--config", str(self.config.uninav_config.resolve())])
        cmd.extend(["--output", str(self.config.output_dir.resolve())])

        # Pass envset files - if single file, pass directly; otherwise pass first one
        # and use --scenario to filter
        if len(envset_files) == 1:
            cmd.extend(["--envset", str(envset_files[0].resolve())])
        else:
            # Multiple envset files for same scene - pass parent directory
            # and rely on scenario filtering
            common_parent = envset_files[0].parent
            cmd.extend(["--envset", str(common_parent.resolve())])

        # Pass scenario IDs to filter
        for sid in scenario_ids:
            cmd.extend(["--scenario", sid])

        # Optional args
        if self.config.scene_root:
            cmd.extend(["--scene-root", str(self.config.scene_root.resolve())])
        if self.config.headless:
            cmd.append("--headless")

        cmd.extend(["--timeout-multiplier", str(self.config.timeout_multiplier)])
        cmd.extend(["--success-threshold", str(self.config.success_threshold)])

        if not self.config.record_trajectory:
            cmd.append("--no-trajectory")
        if not self.config.save_per_episode:
            cmd.append("--no-save-per-episode")
        if not self.config.skip_completed:
            cmd.append("--no-skip")

        # Video recording parameters
        if self.config.record_video:
            cmd.append("--record-video")
        cmd.extend(["--video-fps", str(self.config.video_fps)])
        if not self.config.save_depth_video:
            cmd.append("--no-depth-video")
        if self.config.record_images:
            cmd.append("--record-images")
        cmd.extend(["--image-interval-s", str(self.config.image_interval_s)])

        # Policy parameters (critical for subprocess mode)
        cmd.extend(["--policy", self.config.policy_name])
        
        # Pass policy-specific arguments
        policy_args = self.config.policy_args or {}
        for key, value in policy_args.items():
            if value is None:
                continue
            # Convert key from snake_case to --kebab-case
            arg_name = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    cmd.append(arg_name)
            else:
                cmd.extend([arg_name, str(value)])

        return cmd

    def _load_results_from_output_dir(self) -> None:
        """Load episode results from saved JSON files in output directory.

        This is used after multiprocess mode to aggregate results from all groups.
        """
        if not self.config.output_dir.exists():
            return

        for json_file in sorted(self.config.output_dir.rglob("*.json")):
            if json_file.name == "summary.json":
                continue
            if "videos" in json_file.parts:
                continue

            try:
                with json_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                scenario_id = data.get("scenario_id", "unknown")

                result = EpisodeResult(
                    scenario_id=scenario_id,
                    success=data.get("success", False),
                    termination_reason=data.get("termination_reason", "unknown"),
                    steps=data.get("steps", 0),
                    time_s=data.get("time_s", 0.0),
                    distance_to_goal=data.get("distance_to_goal", float("inf")),
                    path_length=data.get("path_length", 0.0),
                    stop_step=data.get("stop_step", -1),
                    metrics={},
                    trajectory=[],  # Don't load trajectory for aggregation
                )
                self._results.append(result)

            except Exception as e:
                self._log(f"[BenchRunner] Warning: failed to load {json_file}: {e}")

    @staticmethod
    def _estimate_shortest_path(
        waypoints: List[tuple[float, float, float]]
    ) -> float:
        """Estimate shortest path length from a sequence of waypoints."""
        if len(waypoints) < 2:
            return 0.0
        total = 0.0
        for i in range(len(waypoints) - 1):
            dx = waypoints[i + 1][0] - waypoints[i][0]
            dy = waypoints[i + 1][1] - waypoints[i][1]
            total += np.sqrt(dx * dx + dy * dy)
        return total

    def _compute_aggregates(self, total_time: float) -> BenchResult:
        """Aggregate timing and step counts. Scoring is offline (see
        :mod:`bench.evaluator.offline_test`)."""
        if not self._results:
            return BenchResult(total_time_s=total_time)

        n = len(self._results)
        return BenchResult(
            results=self._results,
            avg_steps=sum(r.steps for r in self._results) / n,
            avg_time_s=sum(r.time_s for r in self._results) / n,
            total_time_s=total_time,
        )

    def _save_episode_result(self, result: EpisodeResult, *, ref: Optional[ScenarioRef] = None):
        """Save single episode result to JSON."""
        if ref is None:
            output_file = self.config.output_dir / f"{result.scenario_id}.json"
        else:
            output_file = self._episode_output_file(ref, self.config.output_dir)
            output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and handle numpy types for JSON serialization
        def _to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
                return int(obj)
            if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: _to_json_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_json_serializable(i) for i in obj]
            if isinstance(obj, Path):
                return str(obj)
            return obj

        data = {
            "scenario_id": result.scenario_id,
            "source_envset": str(ref.source_envset_path) if ref is not None else None,
            "termination_reason": result.termination_reason,
            "steps": int(result.steps),
            "time_s": float(result.time_s),
            "path_length": float(result.path_length),
            "stop_step": int(result.stop_step),
        }
        instruction = ""
        initial_pose = None
        robot_type = None
        if ref is not None:
            task = ref.scenario.get("task") if isinstance(ref.scenario, dict) else None
            if isinstance(task, dict):
                navigation = task.get("navigation")
                if isinstance(navigation, dict):
                    instruction = str(navigation.get("instruction") or "")
                else:
                    instruction = str(task.get("instruction") or "")
            robots = ref.scenario.get("robots") if isinstance(ref.scenario, dict) else None
            entries = robots.get("entries") if isinstance(robots, dict) else None
            if isinstance(entries, list) and entries and isinstance(entries[0], dict):
                initial_pose = entries[0].get("initial_pose")
                robot_type = entries[0].get("type")

        data["instruction"] = instruction
        if initial_pose is not None:
            data["initial_pose"] = _to_json_serializable(initial_pose)
        if robot_type is not None:
            data["robot_type"] = str(robot_type)
        if result.trajectory:
            data["trajectory"] = [
                {
                    "step": int(pt.step),
                    "time_s": float(pt.time_s),
                    "position": [float(pt.position[0]), float(pt.position[1]), float(pt.position[2])],
                    "orientation": [float(pt.orientation[0]), float(pt.orientation[1]), float(pt.orientation[2]), float(pt.orientation[3])],
                }
                for pt in result.trajectory
            ]

        # Only whitelisted extra fields are written to public output
        extra = result.extra or {}
        if extra.get("human_paths"):
            data["human_paths"] = _to_json_serializable(extra["human_paths"])
        if extra.get("eqa_answer"):
            data["eqa_answer"] = str(extra["eqa_answer"])

        try:
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[BenchRunner] ERROR saving episode result to {output_file}: {e}")

    def _save_final_results(self, result: BenchResult):
        """Save aggregated results to JSON."""
        output_file = self.config.output_dir / "summary.json"

        # Convert to dict and handle numpy types for JSON serialization
        def _to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
                return int(obj)
            if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: _to_json_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_json_serializable(i) for i in obj]
            if isinstance(obj, Path):
                return str(obj)
            return obj

        data = {
            "avg_steps": float(result.avg_steps),
            "avg_time_s": float(result.avg_time_s),
            "total_time_s": float(result.total_time_s),
            "num_episodes": int(len(result.results)),
        }
        try:
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[BenchRunner] ERROR saving summary to {output_file}: {e}")

        print(f"\n[BenchRunner] === Summary ===")
        print(f"  Episodes: {len(result.results)}")
        print(f"  Avg Steps: {result.avg_steps:.1f}")
        print(f"  Total Time: {result.total_time_s:.1f}s")
        print(f"  (scoring is offline — see bench/evaluator/offline_test.py)")
        print(f"  Results saved to: {self.config.output_dir}")

