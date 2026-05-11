import carb
import carb.events
from pathlib import Path
import omni.kit.app
from omni.metropolis.utils.file_util import JSONFileUtil
from omni.metropolis.utils.carb_util import CarbSettingUtil
from omni.anim.people.scripts.custom_command.command_manager import CustomCommandManager
from omni.anim.people.scripts.custom_command.defines import CustomCommandTemplate
from OmniNavExt.envset.stage_util import CharacterUtil
from OmniNavExt.envset.settings import CommandSetting, Infos

from .randomizer import (
    Command,
    CommandTransitionMap,
    Randomizer,
    TimingCommand,
    GoToCommand,
    TimingToObjectCommand,
    Tuple,
    Dict,
)

"""  # noqa
    This list determines what commands are available for the character randomizer and their probability of
    being picked from the last command
    Each time a new command is added, add it in this list and create a class that inherits the Command class for it
"""


"""  # noqa
Command supported by the Character Randomizer, including character build in commands and OAP custom commands.
4 build in commands: Idle, LookAround, GoTo, Sit.
"""


class Idle(TimingCommand):
    def __init__(self):
        # A reasonble idle duration is about 2 to 6 seconds
        super().__init__(name="Idle", min_time=2, max_time=6)


class LookAround(TimingCommand):
    def __init__(self):
        # A reasonble look-around duration is about 2 to 4 seconds
        super().__init__(name="LookAround", min_time=2, max_time=4)


class GoTo(GoToCommand):
    def __init__(self):
        super().__init__(name="GoTo", min_distance=-1, max_distance=-1, random_rotation=True)

    def randomize(
        self, agent, agent_speed, agent_pos_dict, navmesh, interactable_objects, command_seed, navigation_area
    ) -> Tuple[str, float]:
        # Get latest setting for GoTo distance
        self.min_distance = CommandSetting.get_character_goto_min_distance()
        self.max_distance = CommandSetting.get_character_goto_max_distance()
        return super().randomize(
            agent, agent_speed, agent_pos_dict, navmesh, interactable_objects, command_seed, navigation_area
        )


class Sit(TimingToObjectCommand):
    def __init__(self):
        # A reasonable sit duration is about 4 to 8 seconds, and interact with "Sittable" object
        super().__init__(name="Sit", min_time=4, max_time=8, object_filter_str="Sittable")


class CustomTiming(TimingCommand):
    def __init__(self, oap_command):
        super().__init__(
            name=oap_command.name, min_time=oap_command.min_random_time, max_time=oap_command.max_random_time
        )


class CustomTimingToObject(TimingToObjectCommand):
    def __init__(self, oap_command):
        super().__init__(
            name=oap_command.name,
            min_time=oap_command.min_random_time,
            max_time=oap_command.max_random_time,
            object_filter_str=oap_command.interact_object_filter,
        )


class CustomGoTo(GoToCommand):
    def __init__(self, oap_command):
        super().__init__(name=oap_command.name, min_distance=-1, max_distance=-1, random_rotation=True)

    def randomize(
        self, agent, agent_speed, agent_pos_dict, navmesh, interactable_objects, command_seed, navigation_area
    ) -> Tuple[str, float]:
        # Get latest setting for GoTo distance
        self.min_distance = CommandSetting.get_character_goto_min_distance()
        self.max_distance = CommandSetting.get_character_goto_max_distance()
        return super().randomize(
            agent, agent_speed, agent_pos_dict, navmesh, interactable_objects, command_seed, navigation_area
        )


"""  # noqa
Class for the Character Randomizer
    Initialize special attributes for the characters
"""

CHARACTER_BUILD_IN_COMMANDS = [Idle(), LookAround(), GoTo(), Sit()]


class CharacterRandomizer(Randomizer):

    COMMAND_TRANSITION_MAP = "/persistent/exts/isaacsim.replicator.agent/character_transition_map_file_path"

    def __init__(self, global_seed):
        super().__init__(global_seed)
        # Define character build_in commands
        # Command settings
        self.commands_dict: Dict[Command] = {}
        for cmd in CHARACTER_BUILD_IN_COMMANDS:
            self.commands_dict[cmd.name] = cmd
        self.fallback_command: Command = Idle()
        # Character walking speed is ~1.0 m/s, assume it 1.1 m/s to underestimate the duration
        self.agent_speed = 1.1
        ext_path = Infos.ext_path
        # OAP custom command manager
        self.custom_command_manager = CustomCommandManager.get_instance()
        # Transition Map
        self.transition_map = None
        self.transition_map_path = ""
        self.default_transition_map_path = f"{ext_path}/data/character_command_transition_map.json"
        self.load_entry_command_transition_map()

    async def generate_character_commands(self, global_seed, duration, agent_count, navigation_area):
        character_list = CharacterUtil.get_characters_in_stage(count=agent_count)
        character_dict = {}  # <name, pos>
        for c in character_list:
            name = CharacterUtil.get_character_name(c)
            pos = CharacterUtil.get_character_pos(c)
            character_dict[name] = pos
        self.sync_commands_dict()
        return await self.generate_commands(global_seed, duration, character_dict, navigation_area)

    # Transition map operations

    def get_command_transition_map_path(self):
        return self.transition_map_path

    def load_entry_command_transition_map(self):
        file_path = CarbSettingUtil.get_value_by_key(
            key=CharacterRandomizer.COMMAND_TRANSITION_MAP, fallback_value=self.default_transition_map_path
        )
        self.load_command_transition_map(file_path)

    def load_command_transition_map(self, file_path):
        self.transition_map_path = file_path
        carb.settings.get_settings().set(CharacterRandomizer.COMMAND_TRANSITION_MAP, file_path)
        json_data = JSONFileUtil.load_from_file(file_path)
        self.transition_map = CommandTransitionMap(json_data)

    def save_command_transition_map(self):
        dict_data = self.transition_map.to_dict()
        if JSONFileUtil.write_to_file(self.transition_map_path, dict_data):
            carb.log_info("Character Command Transition Map is saved.")

    def sync_commands_dict(self):
        """Sync commands_dict with transition_map"""
        # - Clear last load
        self.commands_dict.clear()
        # - Get built-in commands
        for cmd in CHARACTER_BUILD_IN_COMMANDS:
            self.commands_dict[cmd.name] = cmd
        # - Register each command from transition_map
        for t_cmd in self.transition_map.get_all_commands():
            if t_cmd.name in self.commands_dict:
                continue
            oap_cmd = self.custom_command_manager.get_custom_command_by_name(t_cmd.name)
            if not oap_cmd:
                carb.log_warn(f"'{t_cmd.name}' is not a valid custom command in omin.anim.people.")
                continue
            if oap_cmd.template == CustomCommandTemplate.TIMING:
                self.commands_dict[t_cmd.name] = CustomTiming(oap_cmd)
            elif oap_cmd.template == CustomCommandTemplate.TIMING_TO_OBJECT:
                self.commands_dict[t_cmd.name] = CustomTimingToObject(oap_cmd)
            elif oap_cmd.template == CustomCommandTemplate.GOTO_BLEND:
                self.commands_dict[t_cmd.name] = CustomGoTo(oap_cmd)


