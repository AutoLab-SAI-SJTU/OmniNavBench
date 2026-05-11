import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from OmniNav.core.config import Config, DistributedConfig, TaskCfg
from OmniNav.core.robot.rigid_body import IRigidBody
from OmniNav.local_paths import apply_local_path_env, resolve_runtime_kit_path

# Init
from OmniNav.core.scene.scene import IScene
from OmniNav.core.task.task import BaseTask, create_task
from OmniNav.core.task_config_manager.base import BaseTaskConfigManager
from OmniNav.core.util import extensions_utils, log
from OmniNav.core.util.clear_task import clear_stage_by_prim_path


class SimulatorRunner:
    def __init__(self, config: Config, task_config_manager: BaseTaskConfigManager):
        self.config = config
        self.task_config_manager = task_config_manager
        self.env_num = self.config.env_num
        if isinstance(self.config, DistributedConfig):
            self.runner_id = self.config.distribution_config.runner_id
            extensions_utils.reload_extensions(self.config.distribution_config.extensions)
        self.setup_isaacsim()
        self.metrics_config = None
        self.metrics_save_path = self.config.metrics_save_path
        if self.metrics_save_path != 'console':
            try:
                with open(self.metrics_save_path, 'w'):
                    pass
            except Exception as e:
                log.error(f'Can not create result file at {self.metrics_save_path}.')
                raise e
        self.create_world()
        self._scene = IScene.create()
        self._stage = self._world.stage

        # Map task_name -> env_id:
        self.task_name_to_env_id_map = {}
        self.env_id_to_task_name_map = {}

        # finished_tasks contains all the finished tasks in current tasks dict
        self.finished_tasks = set()

        self.render_interval = (
            self.config.simulator.rendering_interval if self.config.simulator.rendering_interval is not None else 5
        )
        log.info(f'rendering interval: {self.render_interval}')
        self.render_trigger = 0
        self.loop = False
        self._render = False

    @property
    def current_tasks(self) -> dict[str, BaseTask]:
        return self._world._current_tasks

    def warm_up(self, steps: int = 10, render: bool = True, physics: bool = True):
        """
        Warm up the simulation by running a specified number of steps.

        Args:
            steps (int): The number of warm-up steps to perform. Defaults to 10.
            render (bool): Whether to render the scene during warm-up. Defaults to True.
            physics (bool): Whether to enable physics during warm-up. Defaults to True.

        Raises:
            ValueError: If both `render` and `physics` are set to False, or if `steps` is less than or equal to 0.
        """
        from omni.isaac.core.simulation_context import SimulationContext

        if not render and not physics:
            raise ValueError('both `render` and `physics` are set to False')
        if steps <= 0:
            raise ValueError('steps` is less than or equal to 0')
        if physics:
            for _ in range(steps):
                self._world.step(render=render)
        else:
            for _ in range(steps):
                SimulationContext.render(self._world)

    def step(
        self, actions: Union[List[Dict], None] = None, render: bool = True
    ) -> Tuple[List[Dict], List[bool], List[float]]:
        """
        Step function to advance the simulation environment by one time step.

        This method processes actions for active tasks, steps the simulation world,
        collects observations, updates metrics, and determines task terminations. It also
        handles rendering based on specified intervals.

        Args:
            actions (Union[List[Dict], None], optional): A dictionary mapping task names to
                another dictionary of robot names and their respective actions. If None,
                no actions are applied. Defaults to None.
            render (bool, optional): Flag indicating whether to render the simulation
                at this step. True triggers rendering if the render interval is met.
                Defaults to True.

        Returns:
            Tuple[Dict, Dict[str, bool], Dict[str, float]]:
                - obs (Dict): A dictionary containing observations for each task,
                  further divided by robot names and their observation data.
                - terminated_status (Dict[str, bool]): A dictionary mapping task names
                  to boolean values indicating whether the task has terminated.
                - reward (Dict[str, float]): A dictionary that would contain rewards
                  for each task or robot; however, the actual return and computation of
                  rewards is not shown in the provided code snippet.

        Raises:
            Exception: If an error occurs when applying an action to a robot, the
                exception is logged and re-raised, providing context about the task,
                robot, and current tasks state.

        Notes:
            - The `_world.step()` method advances the simulation, optionally rendering
              the environment based on the `render` flag and the render interval.
            - `get_obs()` is a method to collect observations from the simulation world,
              though its implementation details are not shown.
            - Metrics for each task are updated, and upon task completion, results are
              saved to a JSON file. This includes a flag 'normally_end' set to True,
              which seems to indicate normal termination of the task.
            - The function also manages a mechanism to prevent further action application
              and metric updates for tasks that have been marked as finished.
        """
        """ ================ TODO: Key optimization interval ================= """
        terminated_status = []
        reward = []

        # Handle None actions
        if actions is None:
            actions = []

        for env_id, action_dict in enumerate(actions):
            # terminated tasks will no longer apply action
            if env_id not in self.env_id_to_task_name_map:
                continue

            task_name = self.env_id_to_task_name_map[env_id]

            if task_name in self.finished_tasks:
                continue

            if task_name not in self.current_tasks:
                continue

            task = self.current_tasks.get(task_name)
            for name, action in action_dict.items():
                if name in task.robots:
                    try:
                        task.robots[name].apply_action(action)
                    except Exception:
                        log.error('task_name     : %s', task_name)
                        log.error('robot_name    : %s', name)
                        log.error('current_tasks : %s', [i for i in self.current_tasks.keys()])
                        try:
                            log.error('action        : %s', action)
                            robot = task.robots.get(name)
                            controllers = getattr(robot, 'controllers', None)
                            if controllers is not None:
                                log.error('controllers   : %s', list(controllers.keys()))
                        except Exception:
                            pass
                        log.exception('apply_action raised an exception')
                        raise

        self.render_trigger += 1
        self._render = render and self.render_trigger > self.render_interval
        if self.render_trigger > self.render_interval:
            self.render_trigger = 0

        # Step
        self._world.step(render=self._render)

        # Get obs
        obs = self.get_obs()

        # Record episode data for tasks that are not finished
        time_step_index = self.get_current_time_step_index()
        for env_id, action_dict in enumerate(actions):
            if env_id not in self.env_id_to_task_name_map:
                continue
            task_name = self.env_id_to_task_name_map[env_id]
            if task_name in self.finished_tasks or task_name not in self.current_tasks:
                continue
            task = self.current_tasks.get(task_name)
            if task.episode_logger:
                simulation_time = None
                try:
                    from omni.isaac.core.simulation_context import SimulationContext
                    simulation_time = SimulationContext.instance().get_physics_dt() * time_step_index
                except Exception:
                    pass
                task.record_episode_step(action_dict=action_dict, frame_idx=time_step_index, time_s=simulation_time)

        # update metrics
        for task in self.current_tasks.values():
            if task.is_done():
                self.finished_tasks.add(task.name)
                log.info(f'Task {task.name} finished.')
                metrics_results = task.calculate_metrics()
                if self.metrics_save_path == 'console':
                    print(json.dumps(metrics_results, indent=4))
                elif self.metrics_save_path == 'none' or self.metrics_save_path is None:
                    pass
                else:
                    with open(self.metrics_save_path, 'a') as f:
                        f.write(json.dumps(metrics_results))
                        f.write('\n')

            # finished tasks will no longer update metrics
            if task.name not in self.finished_tasks:
                for metric in task.metrics.values():
                    metric.update(obs[self.task_name_to_env_id_map[task.name]])

        # update terminated_status and rewards
        for env_id in range(self.env_num):
            # terminated tasks will no longer apply action

            if env_id not in self.env_id_to_task_name_map:
                terminated_status.append(True)
                reward.append(-1)
                continue

            task_name = self.env_id_to_task_name_map[env_id]

            if task_name in self.finished_tasks:
                terminated_status.append(True)
                reward.append(-1)
            else:
                terminated_status.append(False)
                r = self.current_tasks[task_name].reward
                reward.append(r.calc() if r is not None else -1)

        return obs, terminated_status, reward

    def get_obs(self) -> List[Dict | None]:
        """
        Get obs
        Returns:
            List[Dict]: obs from isaac sim.
        """
        obs = {}
        for task_name, task in self.current_tasks.items():
            obs[task_name] = task.get_observations()
        # Add render obs
        for task_name, task_obs in obs.items():
            for robot_name, robot_obs in task_obs.items():
                obs[task_name][robot_name]['render'] = self._render

        _obs = []
        for env_id in range(self.env_num):
            if env_id in self.env_id_to_task_name_map:
                _obs.append(obs[self.env_id_to_task_name_map[env_id]])
            else:
                _obs.append(None)
        return _obs

    def stop(self):
        """
        Stop all current operations and clean up the **World**
        """
        self._world.reset()
        self._world.clear()
        self._world.stop()

    def get_current_time_step_index(self) -> int:
        return self._world.current_time_step_index

    def reset(self, env_ids: Optional[List[int]] = None, start_timeline: bool = True) -> Tuple[List, List]:
        """
        Resets the environment for the given environment IDs or initializes it if no IDs are provided.
        This method handles resetting the simulation context, generating new task configs, and finalizing
        tasks when necessary. It supports partial resets for specific environments and ensures proper
        handling of task transitions.

        Args:
            env_ids (Optional[List[int]]): A list of environment IDs to reset. If None, all environments
                are reset or initialized based on the current state.
            start_timeline (bool): Whether to start the timeline after reset. If False, only loads the scene
                and initializes physics without starting the timeline. Default is True for backward compatibility.

        Returns:
            Tuple[List, List]: A tuple containing two lists. The first list contains observations for
                the reset environments, and the second list contains the new task configs.

        Raises:
            ValueError: If the provided `env_ids` are invalid or don't correspond to any existing tasks.
            RuntimeError: If the simulation context fails to reset or initialize properly.

        Notes:
            - If `env_ids` is None and there are current tasks, all environments are reset.
            - If `env_ids` is None and there are no current tasks, the simulation context is initialized.
            - Observations are collected only for environments that are reset or initialized.
            - Tasks corresponding to the reset environments are transitioned to new episodes.
            - When `start_timeline=False`, the scene is loaded and physics is initialized, but the timeline
              remains paused, allowing for physics property fixes before starting simulation.
        """
        from omni.isaac.core.simulation_context import SimulationContext
        from isaacsim.core.simulation_manager import SimulationManager

        new_task_configs = []

        if env_ids is None and self.current_tasks:
            # reset
            log.info('==================== reset all env ====================')
            env_ids = [i for i in range(self.env_num)]

        if env_ids is None:
            # init
            log.info('===================== init reset =====================')
            if start_timeline:
                # Original behavior: reset and start timeline
                SimulationContext.reset(self._world, soft=False)
            else:
                # New behavior: only initialize physics, don't start timeline
                # _next_episodes() will call SimulationManager._on_stop('reset') which stops timeline
                # and then initialize_physics() and _create_simulation_view()
                log.info('==================== init without timeline =====================')
            new_task_configs = self._next_episodes()
        else:
            # switch to next episodes
            env_to_reset = [env_id for env_id in env_ids if env_id in self.env_id_to_task_name_map]

            if not env_to_reset:
                log.warning(f'Not reset empty envs: {env_ids}.')
                return [None for _ in env_ids], [None for _ in env_ids]

            tasks = [self.env_id_to_task_name_map[env_id] for env_id in env_to_reset]
            _task_configs = self._next_episodes(tasks)
            for env_id in env_ids:
                if env_id in self.env_id_to_task_name_map:
                    new_task_configs.append(_task_configs.pop(0))
                else:
                    new_task_configs.append(None)
            [self.finished_tasks.discard(task) for task in tasks]

        all_obs = self.get_obs()
        obs = [all_obs[i] for i in env_ids] if env_ids else all_obs

        if not self.current_tasks:
            # finished
            self._finalize()

        return obs, new_task_configs

    def _finalize(self):
        """
        Finalize the tasks and do some post-processing.
        """
        pass

    def world_clear(self):
        self._world.clear()

    def _can_reuse_scene(self, current_cfg: TaskCfg, next_cfg: Optional[TaskCfg]) -> bool:
        """Check if scene can be reused between current and next task."""
        if next_cfg is None:
            return False
        # Reuse if same USD path and same position
        return (
            current_cfg.scene_asset_path == next_cfg.scene_asset_path and
            current_cfg.scene_position == next_cfg.scene_position
        )

    def _can_reuse_task(self, current_cfg: TaskCfg, next_cfg: Optional[TaskCfg]) -> bool:
        """Check if entire task (scene + robots) can be reused."""
        if not self._can_reuse_scene(current_cfg, next_cfg):
            return False
        # Check robots config
        if len(current_cfg.robots) != len(next_cfg.robots):
            return False
        for curr_robot, next_robot in zip(current_cfg.robots, next_cfg.robots):
            if curr_robot.type != next_robot.type or curr_robot.usd_path != next_robot.usd_path:
                return False
        return True

    def clear_single_task(self, task_name: str, keep_scene: bool = False):
        """
        Clear single task with task_name.

        Args:
            task_name (str): Task name to clear.
            keep_scene (bool): If True, keep scene prim and only clear robots/objects.
        """
        from omni.isaac.core.loggers import DataLogger

        if task_name not in self.current_tasks:
            log.warning(f'Clear task {task_name} fail. The task {task_name} is not in current_tasks.')
            return
        old_task = self.current_tasks[task_name]
        old_task.cleanup()
        del self.current_tasks[task_name]
        self._world._task_scene_built = False
        self._world._data_logger = DataLogger()

        env_id = self.task_name_to_env_id_map[task_name]
        env_path = f'/World/env_{env_id}'

        if keep_scene:
            # Keep scene, only clear robots and objects
            log.info(f'[SceneReuse] Keeping scene at {env_path}, clearing robots/objects')
            clear_stage_by_prim_path(f'{env_path}/robots')
            clear_stage_by_prim_path(f'{env_path}/objects')
        else:
            log.info(f'Clear stage: {env_path}')
            clear_stage_by_prim_path(env_path)

    def _next_episodes(self, reset_tasks: List[str] = None) -> List[TaskCfg]:
        """
        Switch tasks that need to be reset to the next episode.

        This method handles cleaning-up tasks, resetting sim backend, creating new tasks, and
        restoring states for non-reset environments.

        Args:
            reset_tasks (List[str]): a list of task names that need to be reset. If None, the method
                initializes all environments without resetting specific tasks.

        Returns:
            List[TaskCfg]: a list of TaskCfg.

        Raises:
            RuntimeError: If a task specified in `reset_tasks` isn't found in the current tasks.
        """
        from isaacsim.core.simulation_manager import SimulationManager

        env_id_list = []
        env_offset_list = []
        task_name_list = []
        task_configs_list = []
        task_configs_by_name = {}
        delete_task_configs = {}
        delete_task_name_list = []
        reuse_tasks = []
        delete_tasks = []

        if reset_tasks:
            # recycling env_id
            reset_env_ids = {}
            for task_name in reset_tasks:
                if task_name not in self.current_tasks:
                    raise RuntimeError(f'Task with task_name {task_name} not in `current_tasks`.')
                old_task = self.current_tasks[task_name]
                env_id_list.append(old_task.env_id)
                reset_env_ids[task_name] = old_task.env_id

            # clean up tasks that need to reset
            # Peek next config to decide if scene can be reused
            next_cfg = self.task_config_manager.peek_next()
            for task_name in reset_tasks:
                current_cfg = self.current_tasks[task_name].config
                if self._can_reuse_task(current_cfg, next_cfg):
                    reuse_tasks.append(task_name)
                else:
                    delete_tasks.append(task_name)

            if delete_tasks:
                # Stop physics before removing prims to avoid invalid tensor views.
                SimulationManager._on_stop('reset')

                # save the state of envs that need to be kept
                for _task_name, task in self.current_tasks.items():
                    if _task_name in reset_tasks:
                        continue
                    task.save_info()

            if reuse_tasks:
                log.info(f'[SoftReset] Reusing tasks: {reuse_tasks}')
                import numpy as np

                for task_name in reuse_tasks:
                    task = self.current_tasks[task_name]
                    if not isinstance(self.config, DistributedConfig):
                        next_task = self.task_config_manager.get_next(task.env_id)
                    else:
                        import ray

                        next_task = ray.get(
                            self.task_config_manager.get_next.remote(task.env_id, self.runner_id)
                        )
                    new_task_name, new_env_id, new_env_offset, new_cfg = next_task
                    if new_env_id is not None:
                        new_env_id = int(new_env_id)
                    task_configs_by_name[task_name] = new_cfg
                    if new_cfg:
                        task.config = new_cfg
                        task.set_up_runtime(new_task_name, new_env_id, new_env_offset)
                        if new_task_name != task_name:
                            del self.current_tasks[task_name]
                            self.current_tasks[new_task_name] = task
                            if task_name in self.finished_tasks:
                                self.finished_tasks.discard(task_name)
                                self.finished_tasks.add(new_task_name)
                        if task_name != new_task_name and task_name in self.task_name_to_env_id_map:
                            del self.task_name_to_env_id_map[task_name]
                        self.task_name_to_env_id_map[new_task_name] = new_env_id
                        self.env_id_to_task_name_map[new_env_id] = new_task_name
                        # Update robot positions from new config
                        for robot_name, robot in task.robots.items():
                            for new_robot_cfg in new_cfg.robots:
                                if new_robot_cfg.name == robot_name or robot_name.startswith(new_robot_cfg.name):
                                    robot.articulation.set_world_pose(
                                        position=new_robot_cfg.position,
                                        orientation=new_robot_cfg.orientation
                                    )
                                    robot.articulation.set_linear_velocity(np.zeros(3))
                                    robot.articulation.set_angular_velocity(np.zeros(3))
                                    robot.articulation.set_joint_velocities(
                                        np.zeros(robot.articulation.num_dof)
                                    )
                                    break
                        task.post_reset()

            for task_name in delete_tasks:
                current_cfg = self.current_tasks[task_name].config
                keep_scene = self._can_reuse_scene(current_cfg, next_cfg)
                self.clear_single_task(task_name, keep_scene=keep_scene)

            # clear all rigid bodies in scene register and physics backend
            for task_name, task in self.current_tasks.items():
                if task_name in reuse_tasks:
                    continue
                task.clear_rigid_bodies()

            if reuse_tasks and not delete_tasks:
                task_configs_list = [task_configs_by_name.get(name) for name in reset_tasks]
                return task_configs_list

            env_id_list = [reset_env_ids[name] for name in delete_tasks]
            delete_task_name_list = list(delete_tasks)
        else:
            # init
            SimulationManager._on_stop('reset')
            env_id_list = [None for _ in range(self.env_num)]

        # get next_task_configs
        for idx, env_id in enumerate(env_id_list):
            if not isinstance(self.config, DistributedConfig):
                next_task = self.task_config_manager.get_next(env_id)
            else:
                import ray

                next_task = ray.get(self.task_config_manager.get_next.remote(env_id, self.runner_id))
            new_task_name, new_env_id, new_env_offset, task_cfg = next_task
            if task_cfg is None and env_id in self.env_id_to_task_name_map:
                del self.env_id_to_task_name_map[env_id]
            env_id_list[idx] = new_env_id
            env_offset_list.append(new_env_offset)
            task_name_list.append(new_task_name)
            task_configs_list.append(task_cfg)
            if delete_task_name_list:
                delete_task_configs[delete_task_name_list[idx]] = task_cfg

        # create tasks with new task configs
        _new_tasks = []
        _new_tasks_names = []
        for idx, task_config in enumerate(task_configs_list):
            if task_config is None:
                continue
            task = create_task(task_config, self._scene)
            task.set_up_runtime(task_name_list[idx], env_id_list[idx], env_offset_list[idx])
            self._world.add_task(task)
            _new_tasks.append(task)
            _new_tasks_names.append(task.name)
            task.set_up_scene(self._scene)

            # map task_name and env_id of new tasks
            self.task_name_to_env_id_map[task.name] = task.env_id
            self.env_id_to_task_name_map[task.env_id] = task.name

        # Ensure a physics scene exists before initializing physics.
        try:
            import omni.usd
            from pxr import Sdf, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage and not stage.GetPrimAtPath("/World/physicsScene").IsValid():
                UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
        except Exception:
            raise RuntimeError("Failed to define physics scene before initializing physics.")

        # Wait for assets to finish loading before physics init.
        try:
            import omni.kit.app

            assets_loading = None
            for _ in range(120):
                assets_loading = SimulationManager.assets_loading()
                if not assets_loading:
                    break
                omni.kit.app.get_app().update()
            if assets_loading:
                log.warning(
                    "Assets still loading before physics init (assets_loading=%s)",
                    assets_loading,
                )
        except Exception:
            log.exception("Asset load wait failed before physics init")

        print(f"[DEBUG] _next_episodes: reset_tasks={reset_tasks}")

        # Diagnostic: inspect timeline and PhysX state before initialization.
        try:
            import omni.timeline
            import omni.physx
            timeline = omni.timeline.get_timeline_interface()
            print(f"[DEBUG] Before initialize_physics: timeline.is_playing={timeline.is_playing()}, timeline.is_stopped={timeline.is_stopped()}")

            physx_interface = omni.physx.get_physx_interface()
            if physx_interface:
                print(f"[DEBUG] PhysX interface available")
            else:
                print(f"[DEBUG] PhysX interface is None")
        except Exception as e:
            print(f"[DEBUG] Failed to get timeline/physx state: {e}")

        # Allow callers to adjust collision properties before PhysX cooks them.
        pre_physics_hook = getattr(self, "_pre_physics_hook", None)
        if callable(pre_physics_hook):
            pre_physics_hook(self._stage)

        print(f"[DEBUG] Calling SimulationManager.initialize_physics()...")
        import sys
        sys.stdout.flush()

        SimulationManager.initialize_physics()
        print(f"[DEBUG] After initialize_physics - checking robot prim...")
        # Warm up physics before creating the sim view so PhysX registers articulations.
        sim_context = None
        timeline = None
        was_playing = None
        try:
            from isaacsim.core.api.simulation_context import SimulationContext
            import omni.timeline

            sim_context = SimulationContext.instance()
            timeline = omni.timeline.get_timeline_interface()
            was_playing = timeline.is_playing()
            if not was_playing:
                sim_context.play()
            log.info("Warmup: timeline playing after SimulationContext.play=%s", timeline.is_playing())
        except Exception:
            log.exception("Warmup failed before creating sim_view")
            sim_context = None
            timeline = None
            was_playing = None

        # BenchRunner-style physics warmup to register articulation handles.
        try:
            if self._world is not None:
                for _ in range(20):
                    self._world.step(render=False)
        except Exception:
            log.exception("Physics warmup failed before create_simulation_view")

        # create sim_view after physics init and task prims are ready
        if timeline is not None and not timeline.is_playing():
            raise RuntimeError("Warmup failed: timeline is not playing before create_simulation_view")
        SimulationManager._create_simulation_view('reset')
        sim_view = SimulationManager.get_physics_sim_view()
        if sim_view is None:
            raise RuntimeError("Failed to initialize physics simulation view after create_simulation_view()")

        # restore the state of envs that haven't been reset
        if reset_tasks:
            for t in self.current_tasks.values():
                if t.name in _new_tasks_names or t.name in reuse_tasks:
                    continue
                t.restore_info()

        self._scene.unwrap()._finalize(sim_view)  # noqa
        if sim_context is not None and was_playing is False:
            try:
                sim_context.pause()
            except Exception:
                log.exception("Failed to pause timeline after create_simulation_view")
        # post_reset for new tasks
        for task in _new_tasks:
            task.post_reset()

        # log new episodes
        log.info('===================== episodes ========================')
        for task in _new_tasks:
            log.info(f'Next episode: {task.name} at {str(task.env_id)}')
        log.info('======================================================')
        if reset_tasks and reuse_tasks:
            task_configs_list = [
                task_configs_by_name.get(name, delete_task_configs.get(name)) for name in reset_tasks
            ]
        return task_configs_list

    def get_obj(self, name: str) -> IRigidBody:
        # Only supported in gym_env.
        # TODO: handle name by a more robust way and maybe support vec env
        return self._scene.get(name + '_0')

    def remove_collider(self, prim_path: str):
        from omni.physx.scripts import utils

        build = self._world.stage.GetPrimAtPath(prim_path)
        if build.IsValid():
            utils.removeCollider(build)

    def add_collider(self, prim_path: str):
        from omni.physx.scripts import utils

        build = self._world.stage.GetPrimAtPath(prim_path)
        if build.IsValid():
            utils.setCollider(build, approximationShape=None)

    def create_world(self):
        physics_dt = self.config.simulator.physics_dt
        rendering_dt = self.config.simulator.rendering_dt
        physics_dt = eval(physics_dt) if isinstance(physics_dt, str) else physics_dt
        self.dt = physics_dt
        rendering_dt = eval(rendering_dt) if isinstance(rendering_dt, str) else rendering_dt
        use_fabric = self.config.simulator.use_fabric
        log.info(f'simulator params: physics dt={physics_dt}, rendering dt={rendering_dt}, use_fabric={use_fabric}')
        from omni.isaac.core import World

        self._world: World = World(
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
            stage_units_in_meters=1.0,
            sim_params={'use_fabric': use_fabric},
        )
        
    def setup_isaacsim(self):
        # Init Isaac Sim
        from isaacsim import SimulationApp  # noqa
        import os

        apply_local_path_env()
        headless = self.config.simulator.headless
        native = self.config.simulator.native
        webrtc = self.config.simulator.webrtc

        # Build launch config with optional extension paths
        launch_config = {
            'headless': headless,
            'anti_aliasing': 0,
            'hide_ui': False,
            'multi_gpu': False,
            # Force the repo-local kit for consistent runtime behavior.
            'experience': str(resolve_runtime_kit_path()),
        }

        # Add custom extension paths if specified
        # Isaac Sim uses different parameter names in different versions:
        # - 'extension_folders' (list)
        # - 'extra_extension_folders' (list)
        # Try both for compatibility
        if hasattr(self.config.simulator, 'extension_folders') and self.config.simulator.extension_folders:
            ext_paths = self.config.simulator.extension_folders

            # Set environment variable as fallback (Isaac Sim checks this)
            if 'ISAAC_EXTRA_EXT_PATH' in os.environ:
                existing = os.environ['ISAAC_EXTRA_EXT_PATH']
                os.environ['ISAAC_EXTRA_EXT_PATH'] = os.pathsep.join([existing] + ext_paths)
            else:
                os.environ['ISAAC_EXTRA_EXT_PATH'] = os.pathsep.join(ext_paths)

            # Try different parameter names
            launch_config['extension_folders'] = ext_paths

            log.info(f'Custom extension paths configured: {ext_paths}')

        self._simulation_app = SimulationApp(launch_config)
        self._simulation_app._carb_settings.set('/physics/cooking/ujitsoCollisionCooking', False)
        log.debug('SimulationApp init done')

        # Configure streaming for Isaac Sim 5.0+
        if native:
            log.warning('native streaming is DEPRECATED, webrtc streaming is used instead')
        webrtc = native or webrtc
        self.setup_streaming_500(webrtc)

    def setup_streaming_500(self, webrtc: bool):
        """Configure streaming for Isaac Sim 5.0+."""
        if webrtc:
            from omni.isaac.core.utils.extensions import enable_extension

            self._simulation_app.set_setting('/app/window/drawMouse', True)
            enable_extension('omni.services.streamclient.webrtc')

    @property
    def simulation_app(self):
        return self._simulation_app
