from collections import OrderedDict

import numpy as np

from OmniNav.core.robot.articulation import IArticulation
from OmniNav.core.robot.articulation_subset import ArticulationSubset
from OmniNav.core.robot.rigid_body import IRigidBody
from OmniNav.core.robot.robot import BaseRobot
from OmniNav.core.scene.scene import IScene
from OmniNav.core.util import log
from OmniNavExt.configs.robots.aliengo import AliengoRobotCfg
from bench.configs.execution import RobotExecutionProfile


@BaseRobot.register('AliengoRobot')
class AliengoRobot(BaseRobot):
    def __init__(self, config: AliengoRobotCfg, scene: IScene):
        if not isinstance(config, AliengoRobotCfg):
            config_dict = {}
            fields_to_copy = getattr(config, "__fields_set__", set())
            if not fields_to_copy:
                fields_to_copy = {
                    "name",
                    "type",
                    "prim_path",
                    "usd_path",
                    "position",
                    "orientation",
                    "scale",
                    "controllers",
                    "sensors",
                }
            for field in fields_to_copy:
                if not hasattr(config, field):
                    continue
                val = getattr(config, field)
                if field == "sensors" and val is None:
                    # Keep default AliengoRobotCfg sensors when envset provides None.
                    continue
                if field in ["position", "orientation", "scale"] and isinstance(val, (list, np.ndarray)):
                    val = tuple(val)
                config_dict[field] = val
            config = AliengoRobotCfg(**config_dict)
        super().__init__(config, scene)
        self._start_position = np.array(config.position) if config.position is not None else None
        self._start_orientation = np.array(config.orientation) if config.orientation is not None else None

        log.debug(f'aliengo {config.name}: position    : ' + str(self._start_position))
        log.debug(f'aliengo {config.name}: orientation : ' + str(self._start_orientation))

        usd_path = config.usd_path

        log.debug(f'aliengo {config.name}: usd_path         : ' + str(usd_path))
        log.debug(f'aliengo {config.name}: config.prim_path : ' + str(config.prim_path))

        self._robot_scale = np.array([1.0, 1.0, 1.0])
        if config.scale is not None:
            self._robot_scale = np.array(config.scale)
        print(f"[DEBUG] AliengoRobot {config.name}: Creating articulation")
        print(f"[DEBUG] prim_path: {config.prim_path}")
        print(f"[DEBUG] usd_path: {usd_path}")
        self.articulation = IArticulation.create(
            prim_path=config.prim_path,
            name=config.name,
            position=self._start_position,
            orientation=self._start_orientation,
            usd_path=usd_path,
            scale=self._robot_scale,
        )
        print(f"[DEBUG] AliengoRobot {config.name}: Articulation created: {self.articulation is not None}")

        if self.articulation:
            isaac_art = self.articulation.unwrap()
            print(f"[DEBUG] Articulation prim path: {isaac_art.prim.GetPath()}")

            from pxr import UsdPhysics
            has_articulation_api = isaac_art.prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            print(f"[DEBUG] Has ArticulationRootAPI: {has_articulation_api}")

            try:
                dof_count = isaac_art.num_dof
                print(f"[DEBUG] Articulation DOF count: {dof_count}")
            except Exception as e:
                print(f"[DEBUG] Error getting DOF count: {e}")

    def post_reset(self):
        super().post_reset()
        self._robot_base = self._rigid_body_map[self.config.prim_path + '/base']
        self.set_gains()

    def get_robot_scale(self):
        return self._robot_scale

    def get_robot_base(self) -> IRigidBody:
        return self._robot_base

    def get_pose(self):
        return self._robot_base.get_pose()

    def apply_action(self, action: dict):
        """
        Args:
            action (dict): inputs for controllers.
        """
        for controller_name, controller_action in action.items():
            if controller_name not in self.controllers:
                raise KeyError(
                    f'unknown controller {controller_name} in action; '
                    f'available={list(self.controllers.keys())}'
                )
            controller = self.controllers[controller_name]
            control = controller.action_to_control(controller_action)
            self.articulation.apply_action(control)

    def get_obs(self) -> OrderedDict:
        position, orientation = self._robot_base.get_pose()

        # custom
        obs = {
            'position': position,
            'orientation': orientation,
            'controllers': {},
            'sensors': {},
        }

        # common
        for c_obs_name, controller_obs in self.controllers.items():
            obs['controllers'][c_obs_name] = controller_obs.get_obs()
        for sensor_name, sensor_obs in self.sensors.items():
            obs['sensors'][sensor_name] = sensor_obs.get_data()
        return self._make_ordered(obs)

    def set_gains(self):
        """
        Set default stiffness (kps) and damping (kds) for joints.
        """
        num_leg_joints = 12
        kps = np.array([40.0] * num_leg_joints)
        kds = np.array([2.0] * num_leg_joints)
        joint_names = [
            'FL_hip_joint',
            'FR_hip_joint',
            'RL_hip_joint',
            'RR_hip_joint',
            'FL_thigh_joint',
            'FR_thigh_joint',
            'RL_thigh_joint',
            'RR_thigh_joint',
            'FL_calf_joint',
            'FR_calf_joint',
            'RL_calf_joint',
            'RR_calf_joint',
        ]

        joint_subset = ArticulationSubset(self.articulation, joint_names)

        self.articulation.set_gains(kps=kps, kds=kds, joint_indices=joint_subset.joint_indices)
        # VERY important!!! additional physics parameter
        self.articulation.set_solver_position_iteration_count(8)
        self.articulation.set_solver_velocity_iteration_count(0)
        self.articulation.set_enabled_self_collisions(True)

    @staticmethod
    def get_execution_profile() -> RobotExecutionProfile:
        """Execution tuning for Aliengo."""
        return RobotExecutionProfile(
            max_lin_vel=1.0,
            max_ang_vel=1.5,
            step_lin_dist=0.25,
            step_ang_deg=30.0,
            finish_pos_eps=0.05,
            finish_rot_eps_deg=3.0,
        )
