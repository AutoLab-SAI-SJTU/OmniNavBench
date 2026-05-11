from typing import List, Dict
from OmniNavExt.envset.settings import CommandSetting
from OmniNavExt.envset.stage_util import RobotUtil

from .randomizer import (
    CommandTransitionMap,
    Randomizer,
    TimingCommand,
    GoToCommand,
    Tuple,
)


"""  # noqa
Command supported by the Robot Randomizer.
2 build in command: Idle, GoTo.
"""


class Idle(TimingCommand):
    def __init__(self):
        super().__init__(name="Idle", min_time=5, max_time=10)


class GoTo(GoToCommand):
    def __init__(self):
        super().__init__(name="GoTo", min_distance=-1, max_distance=-1, random_rotation=False)

    def randomize(
        self, agent, agent_speed, agent_pos_dict, navmesh, interactable_objects, command_seed, navigation_area
    ) -> Tuple[str, float]:
        # Get latest setting for GoTo distance
        self.min_distance = CommandSetting.get_robot_goto_min_distance()
        self.max_distance = CommandSetting.get_robot_goto_max_distance()
        return super().randomize(
            agent, agent_speed, agent_pos_dict, navmesh, interactable_objects, command_seed, navigation_area
        )


"""  # noqa
Class for the Robot Randomizer
    Initialize special attributes for the robots
"""


class RobotRandomizer(Randomizer):
    def __init__(self, global_seed):
        super().__init__(global_seed)
        # Command settings (hardcode for now)
        self.commands_dict: Dict[str : List[float]] = {"GoTo": GoTo(), "Idle": Idle()}
        self.transition_map = CommandTransitionMap(
            {
                "GoTo": {"weight": 1.0, "transitions": {"GoTo": 0.8, "Idle": 0.2}},
                "Idle": {"weight": 1.0, "transitions": {"GoTo": 0.8, "Idle": 0.2}},
            }
        )
        self.transition_matrix = {"GoTo": [0.8, 0.2], "Idle": [0.8, 0.2]}
        self.fallback_command = Idle()  # Will be used whenever an invalid command is randomly generated
        # Robot speed
        self.agent_speed = 0.6

    async def generate_robot_commands(self, global_seed, duration, robot_type, agent_count, navigation_area):
        robot_list = RobotUtil.get_robots_in_stage(count=agent_count, robot_type_name=robot_type)
        robot_dict = {}
        for c in robot_list:
            name = RobotUtil.get_robot_name(c)
            pos = RobotUtil.get_robot_pos(c)
            robot_dict[name] = pos
        return await self.generate_commands(global_seed, duration, robot_dict, navigation_area)
