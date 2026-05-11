"""
KITT15 wheeled mobile manipulator robot.

Position mode, gains set via articulation API (same as H1).
Wheels: target = current_pos + vel * dt.
"""
from collections import OrderedDict
from typing import Optional

import numpy as np

from OmniNav.core.robot.articulation import IArticulation
from OmniNav.core.robot.articulation_action import ArticulationAction
from OmniNav.core.robot.articulation_subset import ArticulationSubset
from OmniNav.core.robot.robot import BaseRobot
from OmniNav.core.scene.scene import IScene
from OmniNav.core.util import log
from OmniNavExt.configs.robots.kitt15 import KITT15RobotCfg
from bench.configs.execution import RobotExecutionProfile

# Drive gains set in USD BEFORE world.reset() — same as visual test
WHEEL_JOINT_SET = {
    'joint1_left_front_scroll', 'joint1_right_front_scroll',
    'joint1_left_rear_scroll', 'joint1_right_rear_scroll',
}
BODY_DRIVE_GAINS = {
    'Trunk1': (5000.0, 500.0), 'Trunk2': (5000.0, 500.0),
    'Trunk3': (5000.0, 500.0), 'Trunk4': (5000.0, 500.0),
    'Head1': (2000.0, 200.0), 'Head2': (2000.0, 200.0),
    'joint1_left_front_steering': (3000.0, 300.0),
    'joint1_right_front_steering': (3000.0, 300.0),
    'joint1_left_rear_steering': (3000.0, 300.0),
    'joint1_right_rear_steering': (3000.0, 300.0),
}
for _i in range(1, 8):
    BODY_DRIVE_GAINS[f'Joint{_i}_L'] = (2000.0, 200.0)
    BODY_DRIVE_GAINS[f'Joint{_i}_R'] = (2000.0, 200.0)


def _setup_drives_on_stage(prim_path: str):
    """Same as visual test's setup_drives(). Set drive gains + remove wheel limits on stage.
    No ArticulationRoot modification. Not saved to file."""
    from pxr import UsdPhysics
    from omni.usd import get_context
    stage = get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return
    def _walk(prim):
        for child in prim.GetAllChildren():
            name = child.GetName()
            drive = UsdPhysics.DriveAPI.Get(child, 'angular')
            if drive and drive.GetDampingAttr().Get() is not None:
                if name in WHEEL_JOINT_SET:
                    drive.GetStiffnessAttr().Set(1000.0)
                    drive.GetDampingAttr().Set(100.0)
                    drive.GetMaxForceAttr().Set(1e5)
                    for a in ('physics:lowerLimit', 'physics:upperLimit'):
                        attr = child.GetAttribute(a)
                        if attr and attr.Get() is not None:
                            attr.Set(-1e9 if 'lower' in a else 1e9)
                elif name in BODY_DRIVE_GAINS:
                    kp, kd = BODY_DRIVE_GAINS[name]
                    drive.GetStiffnessAttr().Set(kp)
                    drive.GetDampingAttr().Set(kd)
            _walk(child)
    _walk(root)
    log.info('[KITT15] Stage drives configured (no ArticulationRoot change)')

LEFT_WHEEL_JOINTS = ['joint1_left_front_scroll', 'joint1_left_rear_scroll']
RIGHT_WHEEL_JOINTS = ['joint1_right_front_scroll', 'joint1_right_rear_scroll']
TRUNK_JOINTS = ['Trunk1', 'Trunk2', 'Trunk3', 'Trunk4']


@BaseRobot.register('KITT15Robot')
class KITT15Robot(BaseRobot):

    _DT = 1.0 / 30.0

    def __init__(self, config: KITT15RobotCfg, scene: IScene):
        if not isinstance(config, KITT15RobotCfg):
            config_dict = {}
            fields_to_copy = getattr(config, '__fields_set__', set())
            if not fields_to_copy:
                fields_to_copy = {
                    'name', 'type', 'prim_path', 'usd_path',
                    'position', 'orientation', 'scale', 'controllers', 'sensors',
                }
            for field in fields_to_copy:
                if not hasattr(config, field):
                    continue
                val = getattr(config, field)
                if field == 'sensors' and val is None:
                    continue
                if field in ['position', 'orientation', 'scale'] and isinstance(val, (list, np.ndarray)):
                    val = tuple(val)
                config_dict[field] = val
            config = KITT15RobotCfg(**config_dict)

        super().__init__(config, scene)
        self._start_position = np.array(config.position) if config.position is not None else None
        self._start_orientation = np.array(config.orientation) if config.orientation is not None else None
        self.prim_path = str(config.prim_path)
        self._robot_scale = np.array([1.0, 1.0, 1.0])
        if config.scale is not None:
            self._robot_scale = np.array(config.scale)

        # Load USD at config.prim_path, but create articulation at base_link
        # (where ArticulationRoot actually is — matching the URDF structure)
        from omni.isaac.core.utils.stage import add_reference_to_stage
        add_reference_to_stage(prim_path=config.prim_path, usd_path=str(config.usd_path))

        self.articulation = IArticulation.create(
            prim_path=config.prim_path + '/base_link',
            name=config.name,
            position=self._start_position,
            orientation=self._start_orientation,
            usd_path=None,  # already loaded above
            scale=self._robot_scale,
        )

        self._left_wheel_indices = []
        self._right_wheel_indices = []
        self._trunk_indices = []

    def post_reset(self):
        super().post_reset()

        # Build DOF index cache
        dof_names = list(self.articulation.dof_names) if hasattr(self.articulation, 'dof_names') else []
        dof_map = {}
        for i, name in enumerate(dof_names):
            short = name.split('/')[-1].split(':')[-1]
            dof_map[short] = i
        self._left_wheel_indices = [dof_map[n] for n in LEFT_WHEEL_JOINTS if n in dof_map]
        self._right_wheel_indices = [dof_map[n] for n in RIGHT_WHEEL_JOINTS if n in dof_map]
        self._trunk_indices = [dof_map[n] for n in TRUNK_JOINTS if n in dof_map]

        # Set gains via articulation API — same pattern as H1Robot.set_gains()
        self._set_gains()

        log.info(f'[KITT15] DOFs={len(dof_names)}, wheels=L{self._left_wheel_indices}/R{self._right_wheel_indices}')

    def _set_gains(self):
        """Set PD gains via articulation API, same approach as H1Robot."""
        # Trunk
        trunk_names = np.array(['Trunk1', 'Trunk2', 'Trunk3', 'Trunk4'])
        trunk_sub = ArticulationSubset(self.articulation, trunk_names)
        self.articulation.set_gains(
            kps=np.array([300.0, 300.0, 300.0, 300.0]),
            kds=np.array([30.0, 30.0, 30.0, 30.0]),
            joint_indices=trunk_sub.joint_indices,
        )

        # Head
        head_names = np.array(['Head1', 'Head2'])
        head_sub = ArticulationSubset(self.articulation, head_names)
        self.articulation.set_gains(
            kps=np.array([100.0, 100.0]),
            kds=np.array([10.0, 10.0]),
            joint_indices=head_sub.joint_indices,
        )

        # Arms
        arm_names = np.array([f'Joint{i}_{s}' for s in ('L', 'R') for i in range(1, 8)])
        arm_sub = ArticulationSubset(self.articulation, arm_names)
        self.articulation.set_gains(
            kps=np.full(14, 100.0),
            kds=np.full(14, 10.0),
            joint_indices=arm_sub.joint_indices,
        )

        # Steering
        steer_names = np.array([
            'joint1_left_front_steering', 'joint1_right_front_steering',
            'joint1_left_rear_steering', 'joint1_right_rear_steering',
        ])
        steer_sub = ArticulationSubset(self.articulation, steer_names)
        self.articulation.set_gains(
            kps=np.full(4, 200.0),
            kds=np.full(4, 20.0),
            joint_indices=steer_sub.joint_indices,
        )

        # VERY important — same as H1
        self.articulation.set_solver_position_iteration_count(4)
        self.articulation.set_solver_velocity_iteration_count(0)

    def get_robot_scale(self):
        return self._robot_scale

    def get_pose(self):
        try:
            return self.articulation.get_pose()
        except Exception:
            # Physics view not ready — read from USD xform
            from pxr import UsdGeom
            from omni.usd import get_context
            stage = get_context().get_stage()
            prim = stage.GetPrimAtPath(str(self.articulation.prim.GetPath()))
            if prim.IsValid():
                xf = UsdGeom.Xformable(prim)
                mat = xf.ComputeLocalToWorldTransform(0)
                t = mat.ExtractTranslation()
                r = mat.ExtractRotationQuat()
                return (
                    np.array([t[0], t[1], t[2]]),
                    np.array([r.GetReal(), r.GetImaginary()[0], r.GetImaginary()[1], r.GetImaginary()[2]])
                )
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    def apply_action(self, action: dict):
        for controller_name, controller_action in action.items():
            if controller_name not in self.controllers:
                raise KeyError(f'unknown controller {controller_name}')
            controller = self.controllers[controller_name]
            control = controller.action_to_control(controller_action)

            if control.joint_velocities is not None:
                vels = control.joint_velocities
                left_vel = float(vels[0]) if len(vels) > 0 else 0.0
                right_vel = float(vels[1]) if len(vels) > 1 else 0.0
                try:
                    current_pos = self.articulation.get_joint_positions()
                except Exception:
                    return
                if current_pos is None:
                    return
                positions = np.array(current_pos, dtype=np.float64)
                for idx in self._trunk_indices:
                    positions[idx] = 0.0
                for idx in self._left_wheel_indices:
                    positions[idx] = current_pos[idx] + left_vel * self._DT
                for idx in self._right_wheel_indices:
                    positions[idx] = current_pos[idx] + (-right_vel) * self._DT
                self.articulation.apply_action(ArticulationAction(joint_positions=positions))
            elif control.joint_positions is not None:
                self.articulation.apply_action(control)

    def get_obs(self) -> OrderedDict:
        position, orientation = self.get_pose()
        controllers_obs, sensors_obs = super()._get_controllers_and_sensors_obs()
        obs = {
            'position': position,
            'orientation': orientation,
            'controllers': controllers_obs,
            'sensors': sensors_obs,
        }
        return self._make_ordered(obs)

    @staticmethod
    def get_execution_profile() -> RobotExecutionProfile:
        return RobotExecutionProfile(
            max_lin_vel=0.5,
            max_ang_vel=1.0,
            step_lin_dist=0.25,
            step_ang_deg=15.0,
            finish_pos_eps=0.05,
            finish_rot_eps_deg=3.0,
        )
