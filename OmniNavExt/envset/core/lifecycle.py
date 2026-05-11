"""Simulation lifecycle management - coordinates initialization sequence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from OmniNav.core.runner import SimulatorRunner
    from isaacsim import SimulationApp


class SimulationLifecycle:
    """Manages simulation lifecycle: initialization, timeline, shutdown.

    Coordinates the initialization sequence:
    1. Load scene (runner.reset with start_timeline=False)
    2. Fix physics properties (GRScenes)
    3. Bake NavMesh
    4. Start timeline
    5. Initialize virtual humans
    6. Wait for initialization (physics + render warmup)
    """

    def __init__(
        self,
        runner: "SimulatorRunner",
        scenario: Dict[str, Any],
        simulation_app: "SimulationApp",
    ):
        self._runner = runner
        self._scenario = scenario
        self._simulation_app = simulation_app

    def initialize(
        self,
        fix_physics: bool = True,
        bake_navmesh: bool = True,
        exclude_robots: bool = True,
        init_virtual_humans: bool = True,
    ) -> bool:
        """Execute full initialization sequence.

        Args:
            fix_physics: Whether to fix physics properties (GRScenes)
            bake_navmesh: Whether to bake NavMesh
            exclude_robots: Whether to exclude robot prims from NavMesh baking
            init_virtual_humans: Whether to initialize virtual humans

        Returns:
            True if initialization succeeded
        """
        from .navmesh_manager import NavMeshManager
        from .physics_manager import PhysicsManager
        from .scene_manager import SceneManager

        scene_cfg = self._scenario.get("scene") or {}
        navmesh_cfg = self._scenario.get("navmesh") or {}

        # Step 1: Load scene (timeline paused)
        print("[Lifecycle] Loading scene (timeline paused)...")
        self._runner.reset(start_timeline=False)

        # Step 2: Fix physics (GRScenes)
        if fix_physics:
            print("[Lifecycle] Fixing physics properties...")
            scene_manager = SceneManager(scene_cfg, self._scenario)
            scene_root = scene_manager.find_root(self._runner._stage)
            PhysicsManager.fix_grscenes_physics(scene_cfg, scene_root)

        # Step 3: Exclude robots from NavMesh baking
        if exclude_robots:
            try:
                from OmniNavExt.envset.runtime_hooks import EnvsetTaskRuntime

                for task_name, task in self._runner.current_tasks.items():
                    if hasattr(task, "robots") and task.robots:
                        for robot_name, robot in task.robots.items():
                            if hasattr(robot, "config") and hasattr(robot.config, "prim_path"):
                                robot_prim_path = robot.config.prim_path
                                EnvsetTaskRuntime._exclude_from_navmesh(robot_prim_path)
                                print(f"[Lifecycle] Excluded robot '{robot_name}' from NavMesh: {robot_prim_path}")
            except Exception as exc:
                print(f"[Lifecycle] Failed to exclude robots from NavMesh: {exc}")

        # Step 4: Bake NavMesh
        navmesh_success = False
        if bake_navmesh and navmesh_cfg:
            print("[Lifecycle] Baking NavMesh...")
            try:
                scene_root = SceneManager(scene_cfg, self._scenario).find_root(self._runner._stage)
                navmesh_manager = NavMeshManager.from_scenario(navmesh_cfg, scene_cfg, scene_root)
                navmesh_success = navmesh_manager.bake_sync(self._simulation_app, envset_cfg=self._scenario)
            except Exception as e:
                print(f"[Lifecycle] NavMesh baking failed: {e}")
                navmesh_success = False
        else:
            navmesh_success = True  # Skip navmesh

        # Step 5: Start timeline
        print("[Lifecycle] Starting timeline...")
        self.start_timeline()

        # Step 6: Initialize virtual humans
        if init_virtual_humans and navmesh_success:
            print("[Lifecycle] Initializing virtual humans...")
            self._initialize_virtual_humans()

        # Step 7: Wait for initialization
        print("[Lifecycle] Waiting for initialization...")
        self.wait_for_initialization()

        return navmesh_success

    def start_timeline(self):
        """Start the simulation timeline."""
        import omni.timeline  # type: ignore

        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            timeline.play()
            print("[Lifecycle] Timeline started")

    def stop_timeline(self):
        """Stop the simulation timeline."""
        import omni.timeline  # type: ignore

        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing():
            timeline.stop()
            print("[Lifecycle] Timeline stopped")

    def is_timeline_playing(self) -> bool:
        """Check if timeline is playing."""
        import omni.timeline  # type: ignore

        return omni.timeline.get_timeline_interface().is_playing()

    def wait_for_initialization(self, physics_steps: int = 2, render_steps: int = 12):
        """Wait for physics and render to stabilize.

        Args:
            physics_steps: Number of physics warmup steps
            render_steps: Number of render warmup steps
        """
        from omni.isaac.core.simulation_context import SimulationContext  # type: ignore

        world = getattr(self._runner, "_world", None)
        if not world:
            print("[Lifecycle] World not available, skipping initialization wait")
            return

        # Physics warmup
        print(f"[Lifecycle] Physics warmup ({physics_steps} steps)...")
        for i in range(physics_steps):
            try:
                world.step(render=False)
            except Exception as e:
                print(f"[Lifecycle] Physics step {i+1} failed: {e}")

        # Render warmup
        print(f"[Lifecycle] Render warmup ({render_steps} steps)...")
        for i in range(render_steps):
            try:
                SimulationContext.render(world)
            except Exception as e:
                print(f"[Lifecycle] Render step {i+1} failed: {e}")

        print("[Lifecycle] Initialization warmup completed")

    def _initialize_virtual_humans(self):
        """Initialize virtual human behaviors."""
        try:
            from OmniNavExt.envset.runtime_hooks import EnvsetTaskRuntime

            EnvsetTaskRuntime.initialize_virtual_humans(self._scenario)
        except Exception as e:
            print(f"[Lifecycle] Virtual human initialization failed: {e}")
