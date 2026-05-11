import re
import asyncio
from dataclasses import dataclass, field
import random
import hashlib
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
import numpy as np
import carb
import omni.usd
import omni.kit.app
import omni.anim.navigation.core as nav
from omni.anim.people.scripts.interactable_object_helper import InteractableObjectHelper
from omni.kit.notification_manager import post_notification, NotificationStatus
from omni.metropolis.utils.carb_util import CarbUtil
from omni.metropolis.utils.simulation_util import SimulationUtil
from omni.metropolis.utils.semantics_util import SemanticsUtils
from OmniNavExt.envset.settings import CommandSetting
from OmniNavExt.envset.stage_util import UnitScaleService
from .randomizer_util import RandomizerUtil

MAX_RANDOM_CNT = 1000
ONE_AGENT_RANDOM_COMMAND_TIMEOUT = 5


class Command(ABC):
    """
    Command:
        Base class for randomization of one type of command.
        Each subclass is to implement randomiztion rules in randomize().
    """

    def __init__(self, name=""):
        self.name = name if name else self.__class__.__name__
        self.num_precision = 2  # How many digits to keep in the command parameter's text form

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Command):
            return self.name == other.name
        return False

    # Generate one random command.
    # Return the command text form and estimated duration.
    @abstractmethod
    def randomize(
        self, agent, agent_speed, agent_pos_dict, navmesh, interactable_objects, command_seed, navigation_area
    ) -> Tuple[str, float]:
        raise NotImplementedError()


class TimingCommand(Command):
    """
    TimingCommand:
        Base class to randoimze duration parameter.
    """

    def __init__(self, name, min_time, max_time):
        super().__init__()
        self.name = name
        self.min_time = min_time  # Time range to be randomized
        self.max_time = max_time

    def randomize(
        self, agent, agent_speed, agent_pos_dict, navmesh, interactable_objects, command_seed, navigation_area
    ) -> Tuple[str, float]:
        duration = self._randomize_duration()
        text = f"{agent} {self.name} {round(duration, self.num_precision)}"
        return (text, duration)

    def _randomize_duration(self):
        """Get a randomize duration"""
        if self.min_time < 0 or self.max_time < 0 or self.min_time > self.max_time:
            carb.log_warn(f"Command class {self.name} has invalid time range. 1.0 will be used for duration.")
            return 1.0
        return np.random.uniform(self.min_time, self.max_time)


class TimingToObjectCommand(TimingCommand):
    """
    TimingToObjectCommand:
        Base class to randomly pick object to interact as parameter.
        Inherient TimingCommand for randomizaing duration.
    """

    def __init__(self, name, min_time, max_time, object_filter_str):
        super().__init__(name, min_time, max_time)
        self.object_filter_str = object_filter_str  # Filter for interactable objects

    def randomize(
        self, agent, agent_speed, agent_pos_dict, navmesh, interactable_objects, command_seed, navigation_area
    ) -> Tuple[str, float]:
        # Pick object to interact
        agent_pos = agent_pos_dict[agent]
        object_prim = self._random_pick_object(interactable_objects)
        if not object_prim:
            carb.log_info(f"No object to interact with. Creating {self.name} command fails.")
            return None, 0
        object_path = object_prim.GetPrimPath()
        stage = omni.usd.get_context().get_stage()
        target_point, _, _, _ = InteractableObjectHelper.get_interact_prim_offsets(stage, object_prim)
        closest_point = navmesh.query_closest_point(target_point, navigation_area)
        if not navmesh.query_shortest_path(agent_pos, closest_point):
            carb.log_info(f"{object_path} is not reachable.")
            return None, 0
        distance_stage = CarbUtil.dist3(closest_point, agent_pos)
        distance_m = UnitScaleService.stage_to_meters(distance_stage)
        try:
            speed_mps = float(agent_speed)
        except (TypeError, ValueError):
            speed_mps = 0.0
        walk_to_duration = distance_m / speed_mps if speed_mps > 0.0 else 0.0
        # Randomize interact duration
        interact_duration = self._randomize_duration()
        text = f"{agent} {self.name} {object_path} {round(interact_duration, self.num_precision)}"
        duration = walk_to_duration + interact_duration
        return (text, duration)

    def _random_pick_object(self, interactable_objects):
        """Pick a randomize object to interact with"""
        p = re.compile(self.object_filter_str)
        filter_objects = []
        for obj in interactable_objects:
            sem_list = SemanticsUtils.get_prim_semantics(obj)
            for t, d in sem_list:
                if t == "class" and p.match(d):
                    filter_objects.append(obj)
        if filter_objects:
            idx = np.random.randint(0, len(filter_objects))
            return filter_objects[idx]
        else:
            return None


class GoToCommand(Command):
    """
    GoToCommand:
        Base class to randomly pick a goto location.
    """

    def __init__(self, name, min_distance, max_distance, random_rotation):
        super().__init__(name)
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.random_rotation = random_rotation

    def randomize(
        self, agent, agent_speed, agent_pos_dict, navmesh, interactable_objects, command_seed, navigation_area
    ) -> Tuple[str, float]:
        # Need to reset the random seed for the randomization
        inav = nav.acquire_interface()
        inav.set_random_seed(self.name, command_seed)
        # Find a valid and reachable point on the navmesh
        agent_pos = agent_pos_dict[agent]
        agent_pos = carb.Float3(agent_pos[0], agent_pos[1], agent_pos[2])
        min_distance_stage = UnitScaleService.meters_to_stage_threshold(self.min_distance)
        max_distance_stage = UnitScaleService.meters_to_stage_threshold(self.max_distance)
        valid_point = carb.Float3(0, 0, 0)
        valid = False  # Whether the target point is a point on the navmesh
        reachable = None  # Whether there exists a path from the starting point to the target point
        num_attempts = 0  # Number of attempts to find a valid path

        navigation_area_idx = RandomizerUtil.area_name_to_index(navmesh, navigation_area)
        closest_navmesh_point = navmesh.query_closest_point(agent_pos, navigation_area_idx)
        agent_in_area = SimulationUtil.is_the_same_point(agent_pos, closest_navmesh_point)
        area_probabilities = RandomizerUtil.area_idx_to_probability(navmesh, navigation_area_idx)

        while not valid or reachable is None:
            if num_attempts > 1000:
                # If failed to find a valid destination after 1000 attempts, terminate and reset to the default command
                carb.log_info(f"Can't find a valid random point for {agent}.")
                return None, 0
            if agent_in_area:  # agent is in the area, query some random point
                valid_point = navmesh.query_random_point(self.name, area_probabilities)
                distance = CarbUtil.dist3(agent_pos, valid_point)
                valid = (min_distance_stage < distance < max_distance_stage)
                reachable = navmesh.query_shortest_path(agent_pos, valid_point)
            else:  # agent is not in the area, move agent to the area first
                valid_point = navmesh.query_random_point(self.name, area_probabilities)
                valid = True
                reachable = navmesh.query_shortest_path(agent_pos, valid_point)
            num_attempts += 1

        x = str(round(valid_point.x, self.num_precision))
        y = str(round(valid_point.y, self.num_precision))
        z = str(round(valid_point.z, self.num_precision))
        parameter = f"{x} {y} {z}"
        # Update the starting position of this agent for the next GoTo command to work
        agent_pos_dict[agent] = [valid_point.x, valid_point.y, valid_point.z]
        text = f"{agent} {self.name} {parameter}"
        if self.random_rotation:
            text += " _"
        travel_stage = CarbUtil.dist3(valid_point, agent_pos)
        travel_meters = UnitScaleService.stage_to_meters(travel_stage)
        try:
            speed_mps = float(agent_speed)
        except (TypeError, ValueError):
            speed_mps = 0.0
        duration = travel_meters / speed_mps if speed_mps > 0.0 else 0.0
        return (text, duration)


class CommandTransitionMap:
    """
    CommandTransitionMap
        Define command transition to be used in Randomizer.
    """

    @dataclass
    class Command:
        name: str
        weight: float
        transitions: Dict[str, float] = field(default=lambda: {})

    def __init__(self, json_data):
        self._commands: List[CommandTransitionMap.Command] = []
        for name, cmd in json_data.items():
            self._commands.append(
                CommandTransitionMap.Command(name=name, weight=cmd["weight"], transitions=cmd["transitions"])
            )

    def get_all_commands(self):
        return self._commands

    def get_command_by_name(self, command_name):
        for cmd in self._commands:
            if cmd.name == command_name:
                return cmd
        return None

    def add_command(self, command_name: str, weight: float, transitions: Dict[str, float]) -> bool:
        if any(cmd.name == command_name for cmd in self._commands):
            carb.log_warn(f"Command [{command_name}] exists. Adding command to transition map fails.")
            return False
        self._commands.append(CommandTransitionMap.Command(command_name, weight, transitions))
        return True

    def remove_command(self, command_name) -> bool:
        cmd = self.get_command_by_name(command_name)
        if cmd is None:
            carb.log_warn(f"Command [{command_name}] does not exist. Removing command from transition map fails.")
            return False
        self._commands.remove(cmd)
        return True

    def add_command_transition(self, command_name: str, to_command_name: str, weight: float) -> bool:
        if not any(cmd.name == command_name for cmd in self._commands):
            carb.log_warn(f"Command [{command_name}] does not exist. Adding command transiton fails.")
            return False
        transitions = self._commands[command_name].transitions
        if any(cmd.name == to_command_name for cmd in transitions.keys()):
            carb.log_warn(
                f"Command [{command_name}] has transition for {to_command_name} already. Adding transition fails."
            )
            return False
        transitions[to_command_name] = weight
        return True

    def remove_command_transition(self, command_name: str, to_command_name: str) -> bool:
        if not any(cmd.name == command_name for cmd in self._commands):
            carb.log_warn(f"Command [{command_name}] does not exist. Removing transition fails.")
            return False
        transitions = self._commands[command_name].transitions
        if any(cmd.name == to_command_name for cmd in transitions.keys()):
            carb.log_warn(
                f"Command [{command_name}] has transition for {to_command_name} already. Removing transition fails."
            )
            return False
        transitions.pop(to_command_name)
        return True

    def to_dict(self) -> dict:
        result = {}
        for cmd in self._commands:
            result[cmd.name] = {"weight": cmd.weight, "transitions": cmd.transitions}
        return result


class Randomizer:
    """
    Base Randomizer Class
        It supports generating random position and get the random commands for agents.
    """

    ID_counter = -1

    def __init__(self, global_seed):
        self._global_seed = global_seed
        self._existing_pos = []  # cache the random position already generated
        self.name = self.__class__.__name__.replace("Randomizer", "")  # child class name
        self.agent_positions = []  # Positions for all agents in the scene (avoid overlaps among different agents)
        self.agent_id = Randomizer.ID_counter  # an offset for differentiating spawn locations for different agents.
        Randomizer.ID_counter += 1
        # Command settings. To be initialized in the children randomizers.
        self.commands_dict: Dict[str, Command] = {}  # All avaliable commands to be randomized.
        self.fallback_command: Command = None  # Fallback command when one command randomization fails.
        # Agent speed for command randomization
        self.agent_speed = -1
        # Default AABB min and MAX
        self.extent = [(0, 0), (0, 0)]

        # Get the navmesh in the stage
        self.inav = nav.acquire_interface()
        spawn_seed = RandomizerUtil.handle_overflow(self._global_seed + self.agent_id)
        self.inav.set_random_seed(self.name, spawn_seed)

        self.transition_map: CommandTransitionMap = None

    # Every time global seed is changed, the randomized state needs to be reset
    def update_seed(self, new_seed):
        self._global_seed = new_seed
        self.reset()

    # Need to be called after another agent called spawn()
    def update_agent_positions(self, pos):
        self.agent_positions = pos

    # Reset the randomization state and position cache
    # Should be called when a new environment is loaded
    def reset(self):
        self._existing_pos = []

        spawn_seed = RandomizerUtil.handle_overflow(self._global_seed + self.agent_id)
        self.inav.set_random_seed(self.name, spawn_seed)

    # Generate random commands for the given agent list
    async def generate_commands(self, global_seed, duration, agent_pos_dict, navigation_area):
        # Clean up last result
        self.commands = []
        # Get the navmesh in the stage
        navmesh = self.inav.get_navmesh()
        # Get all interactable objects in the stage
        interactable_objects = InteractableObjectHelper.get_all_interactable_objects_in_stage(
            CommandSetting.get_character_interact_object_root_path()
        )

        for area in navigation_area:
            if area.strip() == "":
                continue  # empty string is used for default navigation area
            if self.inav.find_area(area) == -1:
                post_notification(
                    f"Unknown navigation area: '{area}', please check your configuration.",
                    status=NotificationStatus.WARNING,
                )
                return None

        async def one_agent_generate_commands(agent, one_agent_commands, one_agent_duration, navigation_area):
            # This helps each agent generate unique seed
            # Note: Hash collision may still happen, although
            agent_name_hash = int(hashlib.sha256(agent.encode()).hexdigest(), 16) % 10000
            command_seed = RandomizerUtil.handle_overflow(global_seed + agent_name_hash)
            random.seed(command_seed)
            np.random.seed(command_seed)
            command_name = ""
            cnt = 0
            while one_agent_duration < duration:
                if not command_name:
                    # Pick first command
                    t_commands = self.transition_map.get_all_commands()
                    name_list = [cmd.name for cmd in t_commands]
                    weight_list = [cmd.weight for cmd in t_commands]
                    command_name = random.choices(population=name_list, weights=weight_list)[0]
                else:
                    # Pick next command
                    t_command = self.transition_map.get_command_by_name(command_name)
                    if not t_command:
                        carb.log_info(
                            f"{command_name} command is not in transition map. Default command will be used instead."
                        )
                        command_name = self.fallback_command.name
                    else:
                        name_list = list(t_command.transitions.keys())
                        weight_list = list(t_command.transitions.values())
                        command_name = random.choices(population=name_list, weights=weight_list)[0]

                # Check if command exists
                command = None
                if command_name not in self.commands_dict:
                    carb.log_info(f"{command_name} command is not registered. Default command will be used instead.")
                    command = self.fallback_command
                else:
                    command = self.commands_dict[command_name]
                # Run randomization on command
                text, cmd_duration = command.randomize(
                    agent,
                    self.agent_speed,
                    agent_pos_dict,
                    navmesh,
                    interactable_objects,
                    command_seed,
                    navigation_area,
                )
                if not text or not cmd_duration:
                    carb.log_info(
                        f"{command_name} can not be propery randomized. Default command will be used instead."
                    )
                    text, cmd_duration = self.fallback_command.randomize(
                        agent,
                        self.agent_speed,
                        agent_pos_dict,
                        navmesh,
                        interactable_objects,
                        command_seed,
                        navigation_area,
                    )
                one_agent_commands.append(text)
                one_agent_duration += cmd_duration

                cnt += 1
                if cnt >= MAX_RANDOM_CNT:
                    carb.log_warn(
                        f"Reach random command generation maxinum attempts ({MAX_RANDOM_CNT}) for {agent}. "
                        f"Generated commands duration: {one_agent_duration}."
                    )
                    break

        # Generate commands for the agents one by one
        # agent is a the name of the agent in stage
        for agent in agent_pos_dict:
            one_agent_commands = []
            one_agent_duration = 0
            try:
                await asyncio.wait_for(
                    one_agent_generate_commands(
                        agent=agent,
                        one_agent_commands=one_agent_commands,
                        one_agent_duration=one_agent_duration,
                        navigation_area=navigation_area,
                    ),
                    timeout=ONE_AGENT_RANDOM_COMMAND_TIMEOUT,
                )
            except TimeoutError:
                carb.log_warn(
                    f"Reach random command generation timeout ({ONE_AGENT_RANDOM_COMMAND_TIMEOUT}) for {agent}. "
                    f"Generated commands duration: {one_agent_duration}."
                )
            finally:
                for cmd in one_agent_commands:
                    self.commands.append(cmd)
                carb.log_info(f"Generate commands for {agent} done.")
                await omni.kit.app.get_app().next_update_async()

        return self.commands

    # Randomly generate a valid agent position
    # Along with the global seed, each idx gives a deterministic result
    def get_random_position(self, idx: int, spawn_area=None) -> Optional[carb.Float3]:
        # Pos has been spawned, no need to re-compute
        # TODO: fix position calculation cache
        # if idx < len(self._existing_pos):
        #     return self._existing_pos[idx]

        spawn_location = carb.Float3(0, 0, 0)

        navmesh = self.inav.get_navmesh()
        if navmesh is None:
            carb.log_error("Navmesh not found when trying to get random positions")
            return None

        valid = False
        num_attempts = 0
        has_overlap = False

        if spawn_area is None:
            spawn_area = []

        for area in spawn_area:
            if area.strip() == "":
                continue  # empty string is used for default spawn area
            if self.inav.find_area(area) == -1:
                post_notification(
                    f"Unknown spawn area: '{area}', please check your configuration.", status=NotificationStatus.WARNING
                )
                return None

        # determian area probabilities based on spawn_area_idx
        spawn_area_indices = RandomizerUtil.area_name_to_index(navmesh, spawn_area)
        area_probabilities = RandomizerUtil.area_idx_to_probability(navmesh, spawn_area_indices)

        # Determine spawn separation in stage units (defaults to 0.5 meters)
        try:
            import carb.settings as _cs
            sep_m = float(_cs.get_settings().get(
                "/persistent/exts/isaacsim.replicator.agent/spawn_min_separation_m"
            ) or 2.0)
        except Exception:
            sep_m = 2.0
        # metersPerUnit -> stage units per meter
        try:
            from pxr import Usd, UsdGeom
            stage = omni.usd.get_context().get_stage()
            mpu = None
            try:
                mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
            except Exception:
                pass
            if not mpu or mpu <= 0:
                mpu = float(stage.GetMetadata("metersPerUnit") or 1.0)
            if not mpu or mpu <= 0:
                mpu = 1.0
        except Exception:
            mpu = 1.0
        sep_units = sep_m / max(mpu, 1e-9)

        while not valid:
            if num_attempts > 1000:
                has_overlap = True
                break

            spawn_location = navmesh.query_random_point(self.name, area_probabilities)
            valid = True

            for pos in self._existing_pos + self.agent_positions:
                if CarbUtil.dist3(carb.Float3(pos), spawn_location) < sep_units:
                    valid = False
                    break

            if valid and self.extent != [(0, 0), (0, 0)]:
                for point in RandomizerUtil.get_2d_bounding_points(self.extent[0], self.extent[1], spawn_location):
                    closest_point = navmesh.query_closest_point(point, spawn_area_indices)
                    if not SimulationUtil.is_the_same_point(point, closest_point):
                        valid = False
                        break

            num_attempts += 1

        if has_overlap:
            carb.log_warn("With the current number of agents and the scene asset, agent overlapping may not be avoided")
        # Store the random number state so next time it continues the sequence
        # TODO: fix position calculation cache
        # self._existing_pos.append(spawn_location)
        return spawn_location

    # Command transition map operations

    def get_command_transition_map(self):
        return self.transition_map
