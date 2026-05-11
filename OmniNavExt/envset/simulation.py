# flake8: noqa
from omni.metropolis.utils.debug_util import DebugPrint

# Removed redundant direct imports; collision/ground handled in importer
from OmniNavExt.envset.settings import Settings

FRAME_RATE = 30

OMNI_ANIM_PEOPLE_COMMAND_PATH = "/exts/omni.anim.people/command_settings/command_file_path"
ANIM_ROBOT_COMMAND_PATH = "/exts/isaacsim.anim.robot/command_settings/command_file_path"
ENVSET_PATH_SETTING = "/exts/isaacsim.replicator.agent/envset/path"
ENVSET_SCENARIO_SETTING = "/exts/isaacsim.replicator.agent/envset/scenario_id"
ENVSET_AUTOSTART_SETTING = "/exts/isaacsim.replicator.agent/envset/autostart"

dp = DebugPrint(Settings.DEBUG_PRINT, "SimulationManager")