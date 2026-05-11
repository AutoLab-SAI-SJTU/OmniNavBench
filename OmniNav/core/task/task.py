import traceback
from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from OmniNav.core.config import TaskCfg
from OmniNav.core.datahub import DataHub
from OmniNav.core.object import init_objects
from OmniNav.core.robot.rigid_body import IRigidBody
from OmniNav.core.robot.robot import BaseRobot, init_robots
from OmniNav.core.scene.scene import IScene
from OmniNav.core.task.episode_logger import EpisodeLogger
from OmniNav.core.task.metric import BaseMetric, create_metric
from OmniNav.core.util import log
from OmniNav.core.util.pose_mixin import PoseMixin

if TYPE_CHECKING:
    from OmniNav.core.util.ground_probe import GroundProbe


class BaseTask(ABC):
    """
    wrap of omniverse isaac sim's base task

    * enable register for auto register task
    * contains scene/robots/objects.
    """

    tasks = {}

    def __init__(self, config: TaskCfg, scene: IScene):
        self.name = None
        self.env_id = 0
        self.env_offset = []
        self.root_path = None

        self.scene_prim = None
        self.objects = None
        self.robots: Union[Dict[str, BaseRobot], None] = None
        self._scene = scene
        self.scene_rigid_bodies: Dict[str, IRigidBody] = {}
        self.config = config

        self.metrics: Dict[str, BaseMetric] = {}
        self.steps = 0
        self.work = True
        self.loaded = False
        for metric_config in config.metrics:
            self.metrics[metric_config.name] = create_metric(metric_config, self.config)

        from OmniNav.core.task.reward import BaseReward, create_reward  # noqa

        self.reward: Union[BaseReward, None] = create_reward(config.reward, self) if config.reward is not None else None

        # Episode logger is initialized below if envset.logging is configured.
        self.episode_logger: Optional[EpisodeLogger] = None
        self._last_action_dict: Optional[Dict[str, Any]] = None
        # Virtual-human name list, sourced from envset.virtual_humans.name_sequence.
        self._virtual_human_names: Optional[List[str]] = None
        self._episode_saved: bool = False
        self._ground_probe: Optional[GroundProbe] = None

    def set_up_runtime(self, task_name, env_id, env_offset):
        """
        Sets up info (task_name, env_id, env_offset) for this task.

        TODO: refactor and rename

        Args:
            task_name (str): The name of the task.
            env_id (int): The env ID of this task.
            env_offset (List[float]): The env offset for the task.
        """
        if env_id is not None:
            env_id = int(env_id)
        self.env_id: int = env_id
        self.env_offset: List[float] = env_offset
        self.root_path = f'/World/env_{str(self.env_id)}'
        if env_id not in PoseMixin.env_offset_map:
            PoseMixin.env_offset_map[str(env_id)] = env_offset
            log.info(f'env {env_id} at {env_offset}')
        self.name = task_name
        for metric in self.metrics.values():
            metric.set_up_runtime(task_name, env_id, env_offset)

    def load(self):
        """

        Loads the environment scene and initializes robots and objects.

        This method first checks if a scene asset path is defined in the task config.
        If so, it creates a scene using the provided path and specified parameters such as scaling and positioning.
        The scene is then populated with robots and objects based on the configurations stored within `self.config`.

        Raises:
        - Exceptions may be raised during file operations or USD scene creation, but specific exceptions are not documented here.

        Attributes Modified:
        - **self.robots**: A collection of initialized robots set up within the scene.
        - **self.objects**: A dictionary mapping object names to their respective initialized instances within the scene.
        - **self.loaded**: A boolean flag indicating whether the environment has been successfully loaded, set to `True` upon successful completion of this method.

        Logs:
        - Information about the initialized robots and objects is logged using the `log.info` method after successful setup.
        """
        from pxr import Usd

        if self.config.scene_asset_path is not None:
            self._scene.load(self.config, self.env_id, self.env_offset)
            try:
                from OmniNavExt.envset.core.physics_manager import PhysicsManager

                scene_cfg = getattr(self.config, "scene", None)
                scene_root = (
                    str(self._scene.scene_prim.GetPath())
                    if getattr(self._scene, "scene_prim", None) is not None
                    else None
                )
                if isinstance(scene_cfg, dict):
                    PhysicsManager.fix_grscenes_physics(scene_cfg, scene_root)
            except ModuleNotFoundError:
                pass
            except Exception as e:
                log.warn(f"[BaseTask.load] GRScenes physics fix failed: {e}")
            # Scene objects don't need RigidBody - they only need CollisionAPI for collision detection.
            # RigidBody is only needed for robots and virtual humans (created in init_robots/init_objects).
            # Removed the code that creates IRigidBody for all scene prims to prevent:
            # 1. Objects falling through the ground due to invalid physics properties
            # 2. Unnecessary physics computation for static scene objects

        print(f"[DEBUG] Task {self.name}: Before init_robots")
        self.robots = init_robots(self.config, self._scene)
        print(f"[DEBUG] Task {self.name}: After init_robots, created {len(self.robots)} robots")
        self.objects = init_objects(self.config, self._scene)
        self.loaded = True
        self._apply_envset_runtime_hooks()
        self._init_episode_logger()

    def clear_rigid_bodies(self):
        for rigid_body_name in self.scene_rigid_bodies.keys():
            if self._scene.object_exists(rigid_body_name):
                self._scene.remove(target=rigid_body_name)

    def save_info(self):
        """
        Saves the robot information and rigidbody statuses.
        """
        self.save_robot_info()
        self._save_rigidbody_statuses()

    def _save_rigidbody_statuses(self):
        """
        Saves the current status of all rigid bodies in the scene by querying their physics properties excluding
        those in the robot.

        Note:
            rigid prims within articulations aren't included since those RigidBody' physical
            status (transform, velocity, etc) can't be set individually.
        """
        for rigid_body_name, rigid_body in self.scene_rigid_bodies.items():
            if not self._scene.object_exists(rigid_body_name):
                log.error(f'[cache_info] {rigid_body_name} does not exist.')
                continue
            rigid_body.save_status()

    def _restore_rigidbody_statuses(self):
        """
        Restores the statuses of all rigid bodies in the scene based on their stored status data excluding
        those in the robot.
        """
        for rigid_body_name, rigid_body in self.scene_rigid_bodies.items():
            if rigid_body.status is None or not self._scene.object_exists(rigid_body_name):
                continue
            rigid_body.restore_status()

    def set_up_scene(self, scene: IScene) -> None:
        """
        Adding assets to the stage as well as adding the encapsulated objects such as XFormPrim..etc
        to the task_objects happens here.

        Args:
            scene (Scene): [description]
        """
        self._scene = scene
        if not self.loaded:
            self.load()

    def _apply_envset_runtime_hooks(self):
        try:
            from OmniNavExt.envset.runtime_hooks import EnvsetTaskRuntime

            EnvsetTaskRuntime.configure_task(self)
        except ModuleNotFoundError:
            return
        except Exception as exc:
            log.warn(f"[EnvsetRuntime] hook failed: {exc}")

    def _init_episode_logger(self):
        """Initialize the episode logger from config.envset.logging."""
        try:
            log.info(f"[BaseTask] _init_episode_logger() called for task {self.name}")
            envset = getattr(self.config, 'envset', None)
            if not envset:
                log.info(f"[BaseTask] No envset found")
                return

            logging_cfg = envset.get('logging') if isinstance(envset, dict) else getattr(envset, 'logging', None)
            if not logging_cfg:
                log.info(f"[BaseTask] No envset.logging found")
                return

            log_path = logging_cfg.get('path') if isinstance(logging_cfg, dict) else getattr(logging_cfg, 'path', None)
            if not log_path:
                log.info(f"[BaseTask] No log_path in envset.logging")
                return
            
            log.info(f"[BaseTask] Initializing EpisodeLogger with path: {log_path}")

            instruction = logging_cfg.get('instruction') if isinstance(logging_cfg, dict) else getattr(logging_cfg, 'instruction', '')
            distance_threshold = logging_cfg.get('distance_threshold') if isinstance(logging_cfg, dict) else getattr(logging_cfg, 'distance_threshold', 0.0)

            robot_path = None
            if self.robots and len(self.robots) > 0:
                first_robot = list(self.robots.values())[0]
                robot_path = getattr(first_robot.config, 'prim_path', None)

            objects = envset.get('objects') if isinstance(envset, dict) else getattr(envset, 'objects', None)
            room_zone = envset.get('room_zone') if isinstance(envset, dict) else getattr(envset, 'room_zone', None)
            answer = envset.get('answer') if isinstance(envset, dict) else getattr(envset, 'answer', None)

            # Resolve virtual-human names for trajectory logging.
            vh_cfg = envset.get('virtual_humans') if isinstance(envset, dict) else getattr(envset, 'virtual_humans', None)
            if isinstance(vh_cfg, dict):
                seq = vh_cfg.get('name_sequence') or []
                self._virtual_human_names = [str(name) for name in seq if name]

            # Automatically extract all unknown fields from envset
            # Known standard fields that should NOT be copied:
            known_fields = {
                "logging", "scene", "navmesh", "virtual_humans", "robots",
                "task", "goals", "gt_locations", "path_waypoints",
                "objects", "room_zone", "answer"  # These are handled explicitly above
            }
            
            extra_fields: Dict[str, Any] = {}
            if isinstance(envset, dict):
                for key, value in envset.items():
                    if key not in known_fields:
                        extra_fields[key] = value
            else:
                # If envset is an object, try to get all attributes
                for attr_name in dir(envset):
                    if not attr_name.startswith('_') and attr_name not in known_fields:
                        try:
                            attr_value = getattr(envset, attr_name)
                            if not callable(attr_value):
                                extra_fields[attr_name] = attr_value
                        except Exception:
                            pass

            meters_per_env_unit = self._resolve_meters_per_env_unit(envset)
            meters_per_stage_unit = self._get_meters_per_stage_unit()

            self.episode_logger = EpisodeLogger(
                log_path=log_path,
                instruction=instruction,
                distance_threshold=distance_threshold,
                robot_path=robot_path,
                objects=objects,
                room_zone=room_zone,
                answer=answer,
                meters_per_env_unit=meters_per_env_unit,
                meters_per_stage_unit=meters_per_stage_unit,
                extra_fields=extra_fields,
            )
            
            # Initialize GroundProbe when the robot is available.
            if robot_path and self.robots and len(self.robots) > 0:
                try:
                    from OmniNav.core.util.ground_probe import GroundProbe
                    self._ground_probe = GroundProbe(
                        robot_path=robot_path,
                        prefix=f"[BaseTask.{self.name}] ",
                    )
                    log.info(f"[BaseTask] GroundProbe initialized via NavMesh for task {self.name}")
                except Exception as exc:
                    log.error(f"[BaseTask] Failed to initialize GroundProbe: {exc}")
                    log.error("[BaseTask] GroundProbe disabled; continuing without ground projection.")
                    raise
            
            log.info(f"[BaseTask] EpisodeLogger initialized successfully")
        except Exception as exc:
            log.error(f"[BaseTask] Failed to init episode logger: {exc}")
            import traceback
            log.error(traceback.format_exc())

    def _resolve_meters_per_env_unit(self, envset_cfg: Dict[str, Any]) -> float:
        value = getattr(self.config, "scene_units_in_meters", None)
        if value is None and isinstance(envset_cfg, dict):
            value = envset_cfg.get("scene_units_in_meters")
        if value is None:
            raise ValueError("[BaseTask] scene_units_in_meters missing; EnvsetTaskAugmentor should inject it.")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"[BaseTask] scene_units_in_meters must be numeric, got {value}") from exc
        if value <= 0.0:
            raise ValueError(f"[BaseTask] scene_units_in_meters must be > 0, got {value}")
        return value

    @staticmethod
    def _get_meters_per_stage_unit() -> float:
        try:
            from OmniNavExt.envset.stage_util import UnitScaleService  # type: ignore
        except ImportError:
            print("[BaseTask] UnitScaleService unavailable; assume stage units already meters.")
            return 1.0

        try:
            meters_per_unit = float(UnitScaleService.get_meters_per_unit())
        except Exception as exc:
            print(f"[BaseTask] Failed to query UnitScaleService.get_meters_per_unit(): {exc}; using 1.0")
            return 1.0
        if meters_per_unit <= 0.0:
            print(f"[BaseTask] Invalid meters_per_unit={meters_per_unit}; using 1.0")
            return 1.0
        return meters_per_unit

    def _record_virtual_humans_step(self, frame_idx: int, time_s: Optional[float]) -> None:
        """Record virtual-human trajectories to the episode logger."""
        if not self.episode_logger:
            return
        if not self._virtual_human_names:
            return

        try:
            from OmniNavExt.envset.agent_manager import AgentManager  # type: ignore
        except ImportError as exc:
            raise RuntimeError(f"[BaseTask] AgentManager unavailable for virtual human logging: {exc}") from exc

        if not AgentManager.has_instance():
            raise RuntimeError("[BaseTask] AgentManager instance not initialized while virtual humans are configured")

        mgr = AgentManager.get_instance()

        for name in self._virtual_human_names:
            # Skip safely during early frames before the agent has been registered.
            if not mgr.agent_registered(name):
                continue
            position = mgr.get_agent_position(name)
            if position is None:
                continue

            # Only positions are recorded for now; yaw is left empty (could later be
            # extracted from a behaviour script or the USD stage).
            self.episode_logger.record_virtual_human_step(
                agent_name=name,
                position=(position[0], position[1], position[2]),
                yaw_deg=None,
                frame_idx=frame_idx,
                time_s=time_s,
                force=False,
            )

    def record_episode_step(self, action_dict: Optional[Dict[str, Any]] = None, frame_idx: Optional[int] = None, time_s: Optional[float] = None, force: bool = False):
        """Record one episode step."""
        if not self.episode_logger:
            return
        if not self.robots:
            log.debug(f"[BaseTask] record_episode_step: no robots available")
            return
        # Skip logging once episode has been marked finished or already saved
        if DataHub.get_episode_finished(self.name) or self._episode_saved:
            return

        if action_dict:
            self._last_action_dict = action_dict

        # Use the first robot's pose.
        first_robot = list(self.robots.values())[0]
        try:
            obs = first_robot.get_obs()
            position = obs.get('position')
            orientation = obs.get('orientation')

            if position is None or orientation is None:
                # Fall back to get_pose().
                position, orientation = first_robot.get_pose()

            if position is None or orientation is None:
                return

            # Project z onto the ground via GroundProbe when available.
            x, y, z = float(position[0]), float(position[1]), float(position[2])
            if self._ground_probe:
                try:
                    ground_z, _ = self._ground_probe.project(x, y, z)
                    position = (x, y, ground_z)
                except Exception as exc:
                    log.debug(f"[BaseTask] GroundProbe.project failed: {exc}, using original z")
            # Without GroundProbe, the original z is kept.

            command = EpisodeLogger._extract_velocity_command(
                self._last_action_dict or {},
                first_robot
            )

            self.episode_logger.record_step(
                position=tuple(position),
                orientation=tuple(orientation),
                command=command,
                frame_idx=frame_idx if frame_idx is not None else self.steps,
                time_s=time_s,
                force=force,
            )
            # Record virtual-human trajectories in the same time step when configured.
            if not force:
                self._record_virtual_humans_step(
                    frame_idx=frame_idx if frame_idx is not None else self.steps,
                    time_s=time_s,
                )
        except Exception as exc:
            log.debug(f"[BaseTask] Failed to record episode step: {exc}")

    def get_observations(self) -> Dict[str, Any]:
        """
        Returns current observations from the objects needed for the behavioral layer.

        Return:
            Dict[str, Any]: observation of robots in this task
        """
        if not self.work:
            return {}
        obs = {}
        for robot_name, robot in self.robots.items():
            try:
                _obs = robot.get_obs()
                if _obs:
                    obs[robot_name] = _obs
            except Exception as e:
                log.error(self.name)
                log.error(e)
                traceback.print_exc()
                return {}
        return obs

    def update_metrics(self):
        """

        Updates all metrics stored within the instance.

        Scans through the dictionary of metrics kept by the current instance,
        invoking the 'update' method on each one. This facilitates the aggregation
        or recalculation of metric values as needed.

        Note:
        This method does not return any value; its purpose is to modify the state
        of the metric objects internally.
        """
        for _, metric in self.metrics.items():
            metric.update()

    def calculate_metrics(self) -> dict:
        """

        Calculates and aggregates the results of all metrics registered within the instance.

        This method iterates over the stored metrics, calling their respective `calc` methods to compute
        the metric values. The computed values are then compiled into a dictionary, where each key corresponds
        to the metrics' name, and each value is the result of the metric calculation.

        Returns:
            dict: A dictionary containing the calculated results of all metrics, with metric names as keys.

        Note:
            Ensure that all metrics added to the instance have a `calc` method implemented.

        Example Usage:
        ```python
        # Assuming `self.metrics` is populated with metric instances.
        results = calculate_metrics()
        print(results)
        # Output: {'metric1': 0.85, 'metric2': 0.92, 'metric3': 0.78}
        ```
        """
        # Save the episode here if it has finished and not yet been saved.
        log.info(f"[BaseTask] calculate_metrics() called: episode_logger={self.episode_logger is not None}, saved={self._episode_saved}")
        if self.episode_logger and not self._episode_saved:
            try:
                log.info(f"[BaseTask] Saving episode...")
                # Force-record the final pose; use self.steps as frame_idx because
                # time_step_index may not be available at this point.
                self.record_episode_step(frame_idx=self.steps, force=True)
                self.episode_logger.save_episode()
                self._episode_saved = True
                log.info(f"[BaseTask] Episode saved successfully")
            except Exception as exc:
                log.error(f"[BaseTask] Failed to save episode: {exc}")
                import traceback
                log.error(traceback.format_exc())

        metrics_res = {}
        for name, metric in self.metrics.items():
            metrics_res[name] = metric.calc()

        return metrics_res

    def is_done(self) -> bool:
        """
        Returns True of the task is done. The result should be decided by the state of the task.
        """
        raise NotImplementedError

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """
        Called before stepping the physics simulation.

        Args:
            time_step_index (int): [description]
            simulation_time (float): [description]
        """
        self.steps += 1
        self.record_episode_step(frame_idx=time_step_index, time_s=simulation_time)
        return

    def save_robot_info(self):
        """
        Saves information of all robots in the task instance.
        """
        for robot in self.robots.values():
            robot.save_robot_info()

    def restore_info(self):
        """
        Restores the information and statuses of rigid bodies and robots.
        """
        self._restore_rigidbody_statuses()
        for robot in self.robots.values():
            robot.restore_robot_info()

    def post_reset(self) -> None:
        """Calls while doing a .reset() on the world."""
        self.steps = 0
        self._episode_saved = False
        self._last_action_dict = None
        if self.episode_logger:
            self.episode_logger.reset()
        for robot in self.robots.values():
            robot.post_reset()
        # TODO: Verify whether RigidPrims' post_reset need to be called
        return

    def cleanup(self) -> None:
        """
        Used to clean up the resources loaded in the task.
        """
        for obj in self.objects.values():
            # Using try here because we want to ignore all exceptions
            try:
                self._scene.remove(obj.name)
            finally:
                log.info('[cleanup] objs cleaned.')
        for robot in self.robots.values():
            # Using try here because we want to ignore all exceptions
            log.info(f'[cleanup] cleanup robot {robot.articulation.name}')
            try:
                robot.cleanup()
                self._scene.remove(robot.articulation.name, registry_only=True)
            finally:
                log.info('[cleanup] robots cleaned.')

    @classmethod
    def register(cls, name: str):
        """
        Register an task class with the given name(decorator).

        Args:
            name(str): name of the task
        """

        def wrapper(task_class):
            """
            Register the task class.
            """
            cls.tasks[name] = task_class
            return task_class

        return wrapper


def create_task(config: TaskCfg, scene: IScene):
    task_cls: BaseTask = BaseTask.tasks[config.type](config, scene)
    return task_cls
