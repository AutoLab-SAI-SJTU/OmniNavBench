# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from collections import OrderedDict
from typing import Optional

import numpy as np

from OmniNav.core.robot.articulation_action import ArticulationAction
from OmniNav.core.robot.isaacsim.articulation import IsaacsimArticulation
from OmniNav.core.robot.robot import BaseRobot
from OmniNav.core.scene.scene import IScene
from OmniNav.core.util import log
from OmniNavExt.configs.robots.carter_v1 import CarterV1RobotCfg
from OmniNavExt.robots.jetbot import WheeledRobot
from bench.configs.execution import RobotExecutionProfile


@BaseRobot.register('CarterV1Robot')
class CarterV1Robot(BaseRobot):
    def __init__(self, config: CarterV1RobotCfg, scene: IScene):
        if not isinstance(config, CarterV1RobotCfg):
            # Avoid config.dict() to bypass Pydantic serializer warnings about list vs tuple
            config_dict = {}
            # Copy explicitly set fields
            fields_to_copy = getattr(config, "__fields_set__", set())
            # If __fields_set__ is empty (could happen if constructed loosely), copy known non-None fields
            if not fields_to_copy:
                 fields_to_copy = {"name", "type", "prim_path", "usd_path", "position", "orientation", "scale", "controllers", "sensors"}

            for field in fields_to_copy:
                if not hasattr(config, field): continue
                val = getattr(config, field)
                if field == "sensors" and val is None:
                    # Keep default CarterV1RobotCfg sensors when envset provides None.
                    continue
                
                # Fix list -> tuple for geometric properties
                if field in ["position", "orientation", "scale"] and isinstance(val, (list, np.ndarray)):
                    val = tuple(val)
                
                config_dict[field] = val
            
            config = CarterV1RobotCfg(**config_dict)

        super().__init__(config, scene)
        self._start_position = np.array(config.position) if config.position is not None else None
        self._start_orientation = np.array(config.orientation) if config.orientation is not None else None

        log.debug(f'carter_v1 {config.name} position    : ' + str(self._start_position))
        log.debug(f'carter_v1 {config.name} orientation : ' + str(self._start_orientation))

        usd_path = config.usd_path

        log.debug(f'carter_v1 {config.name} usd_path         : ' + str(usd_path))
        log.debug(f'carter_v1 {config.name} config.prim_path : ' + str(config.prim_path))
        self.prim_path = str(config.prim_path)
        self._robot_scale = np.array([1.0, 1.0, 1.0])
        if config.scale is not None:
            self._robot_scale = np.array(config.scale)
        # Carter V1 uses 'left_wheel' and 'right_wheel' as joint names (not 'left_wheel_joint')
        self.articulation = WheeledRobot(
            prim_path=config.prim_path,
            name=config.name,
            wheel_dof_names=['left_wheel', 'right_wheel'],
            position=self._start_position,
            orientation=self._start_orientation,
            usd_path=usd_path,
            scale=self._robot_scale,
        )

    def get_robot_scale(self):
        return self._robot_scale

    def get_pose(self):
        return self.articulation.get_pose()

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
            try:
                control = controller.action_to_control(controller_action)
            except Exception:
                log.error(
                    '[CarterV1Robot] controller.action_to_control failed: controller=%s type=%s action=%s',
                    controller_name,
                    controller.__class__.__name__,
                    controller_action,
                )
                log.exception('[CarterV1Robot] action_to_control raised an exception')
                raise
            try:
                self.articulation.apply_wheel_actions(control)
            except Exception:
                articulation = getattr(self, 'articulation', None)
                try:
                    log.error(
                        '[CarterV1Robot] articulation.apply_action failed: controller=%s type=%s control=%s',
                        controller_name,
                        controller.__class__.__name__,
                        getattr(control, 'get_dict', lambda: control)(),
                    )
                    if articulation is not None:
                        log.error(
                            '[CarterV1Robot] articulation state: num_dof=%s dof_names=%s wheel_dof_indices=%s',
                            getattr(articulation, 'num_dof', None),
                            getattr(articulation, 'dof_names', None),
                            getattr(articulation, 'wheel_dof_indices', None),
                        )
                        log.error(
                            '[CarterV1Robot] control shape: len=%s joint_indices=%s',
                            getattr(control, 'get_length', lambda: None)(),
                            getattr(control, 'joint_indices', None),
                        )
                except Exception:
                    pass
                log.exception('[CarterV1Robot] articulation.apply_action raised an exception')
                raise

    def get_obs(self) -> OrderedDict:
        position, orientation = self.articulation.get_pose()

        # custom
        obs = {
            'position': position,
            'orientation': orientation,
            'joint_positions': self.articulation.get_joint_positions(),
            'joint_velocities': self.articulation.get_joint_velocities(),
            'controllers': {},
            'sensors': {},
        }

        # common
        for c_obs_name, controller_obs in self.controllers.items():
            obs['controllers'][c_obs_name] = controller_obs.get_obs()
        for sensor_name, sensor_obs in self.sensors.items():
            obs['sensors'][sensor_name] = sensor_obs.get_data()
        return self._make_ordered(obs)

    @staticmethod
    def get_execution_profile() -> RobotExecutionProfile:
        """Execution tuning for Carter v1."""
        return RobotExecutionProfile(
            max_lin_vel=0.8,
            max_ang_vel=1.2,
            step_lin_dist=0.25,
            step_ang_deg=20.0,
            finish_pos_eps=0.05,
            finish_rot_eps_deg=3.0,
        )
