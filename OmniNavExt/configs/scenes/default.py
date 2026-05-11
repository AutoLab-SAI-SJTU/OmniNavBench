import random
from datetime import datetime
import carb
from omni.metropolis.utils.file_util import YamlFileUtil
from OmniNavExt.envset.settings import AssetPaths, Infos


class ConfigFileDefault:
    """
    Helper class to read default values for config file.
    """

    DEFAULT_CONFIG_VALUES_RELATIVE_PATH = "data/config_file/config_file_default_values.yaml"
    DEFAULT_CONFIG_FILE_RELATIVE_PATH = "config/default_config.yaml"

    _default_values = {}

    @classmethod
    def get_all_default_values(cls):
        if not cls._default_values:
            ext_path = Infos.ext_path
            path = f"{ext_path}/{cls.DEFAULT_CONFIG_VALUES_RELATIVE_PATH}"
            default_yaml = YamlFileUtil.load_yaml(path)
            if not default_yaml:
                raise FileNotFoundError(f"Unable to load config file default values from {path}.")
            # Default values obtained at runtime
            default_yaml["global"]["seed"] = cls.get_random_seed()
            default_yaml["scene"]["asset_path"] = AssetPaths.default_scene_path()
            default_yaml["character"]["asset_path"] = AssetPaths.default_character_path()
            cls._default_values = default_yaml
        return cls._default_values

    @classmethod
    def get_default_value_by_name(cls, section_name, property_name):
        default_values = cls.get_all_default_values()
        if section_name not in default_values:
            carb.log_warn(f"Unable to get default values for {section_name} section.")
            return None
        section_values = default_values[section_name]
        if property_name not in section_values:
            carb.log_warn(f"Unable to get default values for {property_name} in {section_name} section.")
            return None
        return section_values[property_name]

    @staticmethod
    def get_random_seed():
        """
        Generate a random seed based on current timestamp
        Python 3 int has no max limits. Here we use limit in most popular languages (2147483647) to avoid potential problems
        """
        random.seed(datetime.now().timestamp())
        return random.randrange(0, 2147483647)

    @classmethod
    def get_default_config_file_path(cls):
        ext_path = Infos.ext_path
        return f"{ext_path}/{cls.DEFAULT_CONFIG_FILE_RELATIVE_PATH}"
