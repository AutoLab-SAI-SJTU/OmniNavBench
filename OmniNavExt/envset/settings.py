"""
    Setting will first read from Carb setting if it is not empty.
    Otherwise it will use default value or fallback value and update it to Carb.
    This way Carb setting is always the actual value in use.
"""

import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, ClassVar

try:
    import carb
    import carb.settings
    import omni.kit
    from isaacsim.storage.native.nucleus import get_assets_root_path_async
    from omni.metropolis.utils.carb_util import CarbSettingUtil
    ISAAC_AVAILABLE = True
except ImportError:
    carb = None
    omni = None
    ISAAC_AVAILABLE = False


@dataclass
class Settings:
    """
    Manager class to handle general settings
    """

    EXTEND_DATA_GENERATION_LENGTH: ClassVar[str] = "/exts/isaacsim.replicator.agent/extend_data_generation_length"
    SKIP_BIPED_SETUP: ClassVar[str] = "/exts/isaacsim.replicator.agent/skip_biped_setup"
    DEBUG_PRINT: ClassVar[str] = "/exts/isaacsim.replicator.agent/debug_print"

    @classmethod
    def extend_data_generation_length(cls):
        return CarbSettingUtil.get_value_by_key(
            key=cls.EXTEND_DATA_GENERATION_LENGTH, fallback_value=0, override_setting=True
        )

    @classmethod
    def skip_biped_setup(cls):
        return CarbSettingUtil.get_value_by_key(key=cls.SKIP_BIPED_SETUP, fallback_value=False, override_setting=True)

    @classmethod
    def debug_print(cls):
        return CarbSettingUtil.get_value_by_key(key=cls.DEBUG_PRINT, fallback_value=False, override_setting=True)


class Infos:
    """
    Information that to be shared across the extension
    """

    ext_version = ""
    ext_path = ""


class GlobalValues:
    config_file_format = None


class AssetPaths:
    """
    Manager class to handle all asset paths
    """

    cached_asset_default_root_path = None
    USE_ISAAC_SIM_ASSET_ROOT_SETTING = "/exts/isaacsim.replicator.agent/asset_settings/use_isaac_sim_asset_root"
    EXCLUSIVE_CHARACTER_FOLDERS = "/exts/isaacsim.replicator.agent/asset_settings/exclusive_character_assets_folders"

    DEFAULT_BIPED_ASSET_PATH = "/exts/isaacsim.replicator.agent/asset_settings/default_biped_assets_path"
    DEFAULT_SCENE_PATH = "/exts/isaacsim.replicator.agent/asset_settings/default_scene_path"
    DEFAULT_CHARACTER_PATH = "/exts/isaacsim.replicator.agent/asset_settings/default_character_asset_path"

    FALLBACK_BIPED_ASSET_PATH = "/exts/isaacsim.replicator.agent/asset_settings/fallback_biped_assets_path"
    FALLBACK_SCENE_PATH = "/exts/isaacsim.replicator.agent/asset_settings/fallback_scene_path"
    FALLBACK_CHARACTER_PATH = "/exts/isaacsim.replicator.agent/asset_settings/fallback_character_asset_path"

    @classmethod
    async def cache_paths_async(cls):
        try:
            cls.cached_asset_default_root_path = await get_assets_root_path_async()
            carb.log_info(f"Cache Isaac Sim assets root path: {cls.cached_asset_default_root_path}.")
        except RuntimeError:
            carb.log_warn(
                f"Finding Isaac Sim assets root path has encountered runtime error. Will use fallback asset paths instead."
            )
        except asyncio.CancelledError:
            carb.log_warn(
                f"Finding Isaac Sim assets root path has been cancelled unexpectedly. Will use fallback asset paths instead."
            )
        except TimeoutError:
            carb.log_warn(f"Timeout for finding Isaac Sim assets root path. Will use fallback asset paths instead.")

    @classmethod
    def exclusive_character_folders(cls):
        return CarbSettingUtil.get_value_by_key(
            key=cls.EXCLUSIVE_CHARACTER_FOLDERS, fallback_value=["biped_demo"], override_setting=True
        )

    @classmethod
    def default_biped_asset_path(cls):
        fallback_path = carb.settings.get_settings().get(cls.FALLBACK_BIPED_ASSET_PATH)
        return cls._get_asset_carb_value(
            cls.DEFAULT_BIPED_ASSET_PATH, "/Isaac/People/Characters/Biped_Setup.usd", fallback_path
        )

    @classmethod
    def default_biped_asset_name(cls):
        full_path = cls.default_biped_asset_path()
        if not full_path:
            return None
        return Path(full_path).stem

    @classmethod
    def default_scene_path(cls):
        fallback_path = carb.settings.get_settings().get(cls.FALLBACK_SCENE_PATH)
        return cls._get_asset_carb_value(
            cls.DEFAULT_SCENE_PATH, "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd", fallback_path
        )

    @classmethod
    def default_character_path(cls):
        fallback_path = carb.settings.get_settings().get(cls.FALLBACK_CHARACTER_PATH)
        return cls._get_asset_carb_value(cls.DEFAULT_CHARACTER_PATH, "/Isaac/People/Characters/", fallback_path)

    @classmethod
    def _get_asset_carb_value(cls, carb_setting_key, default_path_from_root, fallback_path):
        # First try the path in carb settings
        path = carb.settings.get_settings().get(carb_setting_key)
        if path:
            return path
        # Then try Isaac Sim root assets path (and update to carb)
        if cls.cached_asset_default_root_path:
            path = cls.cached_asset_default_root_path + default_path_from_root
            carb.settings.get_settings().set(carb_setting_key, path)
            return path
        # Finally we will use fallback path (and update to carb)
        if fallback_path:
            path = fallback_path
            carb.settings.get_settings().set(carb_setting_key, path)
            return path
        return None


class BehaviorScriptPaths:
    DEFAULT_BEHAVIOR_SCRIPT_PATH = "/exts/isaacsim.replicator.agent/behavior_script_settings/behavior_script_path"
    DEFAULT_ROBOT_BEHAVIOR_SCRIPT_PATH_PREFIX = "/exts/isaacsim.replicator.agent/behavior_script_settings"

    @classmethod
    def behavior_script_path(cls):
        fallback_path = (
            omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module("omni.anim.people")
            + "/omni/anim/people/scripts/character_behavior.py"
        )
        return CarbSettingUtil.get_value_by_key(
            key=cls.DEFAULT_BEHAVIOR_SCRIPT_PATH, fallback_value=fallback_path, override_setting=True
        )

    @classmethod
    def robot_behavior_script_path(cls, robot_type):
        fallback_path = (
            omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module("isaacsim.anim.robot")
            + "/isaacsim/anim/robot/agent/"
            + robot_type.lower()
            + ".py"
        )
        carb_key = f"{cls.DEFAULT_ROBOT_BEHAVIOR_SCRIPT_PATH_PREFIX}/{robot_type.lower()}_behavior_script_path"
        return CarbSettingUtil.get_value_by_key(key=carb_key, fallback_value=fallback_path, override_setting=True)


class PrimPaths:
    """
    Manager class to handle all prim paths
    """

    CHARACTERS_PARENT_PATH = "/exts/isaacsim.replicator.agent/characters_parent_prim_path"
    ROBOTS_PARENT_PATH = "/exts/isaacsim.replicator.agent/robots_parent_prim_path"
    CAMERAS_PARENT_PATH = "/exts/isaacsim.replicator.agent/cameras_parent_prim_path"
    LIDAR_CAMERAS_PARENT_PATH = "/exts/isaacsim.replicator.agent/lidar_cameras_parent_prim_path"

    @classmethod
    def biped_prim_path(cls):
        biped_name = AssetPaths.default_biped_asset_name()
        return f"{cls.characters_parent_path()}/{biped_name}"

    @classmethod
    def characters_parent_path(cls):
        return CarbSettingUtil.get_value_by_key(
            key=cls.CHARACTERS_PARENT_PATH, fallback_value="/World/Characters", override_setting=True
        )

    @classmethod
    def robots_parent_path(cls):
        return CarbSettingUtil.get_value_by_key(
            key=cls.ROBOTS_PARENT_PATH, fallback_value="/World/Robots", override_setting=True
        )

    @classmethod
    def cameras_parent_path(cls):
        return CarbSettingUtil.get_value_by_key(
            key=cls.CAMERAS_PARENT_PATH, fallback_value="/World/Cameras", override_setting=True
        )

    @classmethod
    def lidar_cameras_parent_path(cls):
        return CarbSettingUtil.get_value_by_key(
            key=cls.LIDAR_CAMERAS_PARENT_PATH, fallback_value="/World/Lidars", override_setting=True
        )


class WriterSetting:

    DEFAULT_OUTPUT_PATH = "/exts/isaacsim.replicator.agent/default_replicator_output_path"

    class DefaultWriterConstant:
        SHOULDER_OCCLUSION_THRESHOLD = (0.5,)
        WIDTH_THRESHOLD = (0.6,)
        HEIGHT_THRESHOLD = (0.6,)
        SHOULDER_HEIGHT_RATIO = (0.25,)

    class SensorType:
        Lidar = "Lidar"
        Camera = "Camera"
        Unknown = "Unkown"

    class AnnotatorPrefix:
        class ObjectDetection:
            GENERIC = "object_info"
            AGENT_SPECIFIC = "agent_info"

        class Others:
            CUSTOMIZED = "customized"

    class AgentStatus:
        INSIDE = 0
        TRUNCATED = 1
        OUTSIDE = 2

    @classmethod
    def get_writer_default_output_path(cls):
        fallback_path = str(Path.home().joinpath("ReplicatorResult"))
        return CarbSettingUtil.get_value_by_key(
            key=cls.DEFAULT_OUTPUT_PATH, fallback_value=fallback_path, override_setting=True
        )


class CommandSetting:

    CHARACTER_GOTO_MIN_DISTANCE = "/persistent/exts/isaacsim.replicator.agent/character_goto_min_distance"
    CHARACTER_GOTO_MAX_DISTANCE = "/persistent/exts/isaacsim.replicator.agent/character_goto_max_distance"
    CHARACTER_INTERACT_OBJECT_ROOT_PATH = (
        "/persistent/exts/isaacsim.replicator.agent/character_interact_object_root_path"
    )

    @classmethod
    def get_character_goto_min_distance(cls):
        return CarbSettingUtil.get_value_by_key(key=cls.CHARACTER_GOTO_MIN_DISTANCE)

    @classmethod
    def get_character_goto_max_distance(cls):
        return CarbSettingUtil.get_value_by_key(key=cls.CHARACTER_GOTO_MAX_DISTANCE)

    @classmethod
    def set_character_goto_min_distance(cls, value: float):
        return CarbSettingUtil.set_value_by_key(key=cls.CHARACTER_GOTO_MIN_DISTANCE, new_value=value)

    @classmethod
    def set_character_goto_max_distance(cls, value: float):
        return CarbSettingUtil.set_value_by_key(key=cls.CHARACTER_GOTO_MAX_DISTANCE, new_value=value)

    @classmethod
    def get_character_interact_object_root_path(cls):
        return CarbSettingUtil.get_value_by_key(key=cls.CHARACTER_INTERACT_OBJECT_ROOT_PATH)

    @classmethod
    def set_character_interact_object_root_path(cls, value: str):
        return CarbSettingUtil.set_value_by_key(key=cls.CHARACTER_INTERACT_OBJECT_ROOT_PATH, new_value=value)

    # Robot Command

    ROBOT_GOTO_MIN_DISTANCE = "/persistent/exts/isaacsim.replicator.agent/robot_goto_min_distance"
    ROBOT_GOTO_MAX_DISTANCE = "/persistent/exts/isaacsim.replicator.agent/robot_goto_max_distance"

    @classmethod
    def get_robot_goto_min_distance(cls):
        return CarbSettingUtil.get_value_by_key(key=cls.ROBOT_GOTO_MIN_DISTANCE)

    @classmethod
    def get_robot_goto_max_distance(cls):
        return CarbSettingUtil.get_value_by_key(key=cls.ROBOT_GOTO_MAX_DISTANCE)

    @classmethod
    def set_robot_goto_min_distance(cls, value: float):
        return CarbSettingUtil.set_value_by_key(key=cls.ROBOT_GOTO_MIN_DISTANCE, new_value=value)

    @classmethod
    def set_robot_goto_max_distance(cls, value: float):
        return CarbSettingUtil.set_value_by_key(key=cls.ROBOT_GOTO_MAX_DISTANCE, new_value=value)
