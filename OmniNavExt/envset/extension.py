import asyncio
from typing import Optional, Callable

try:
    import carb
    import carb.settings
    import omni.ext
    import omni.kit.app
    ISAAC_AVAILABLE = True
except ImportError:
    carb = None
    omni = None
    ISAAC_AVAILABLE = False

from omni.metropolis.utils.config_file.core import ConfigFileFormat
from isaacsim.replicator.incident.config_file_defines import IncidentEventSection
from isaacsim.replicator.incident.extension import get_instance as get_incident_ext_instance
from .agent_manager import AgentManager
from .settings import AssetPaths, Infos, GlobalValues
from .config_file import get_all_section_cls
from .patches import install_safe_simtimes_guard
from .world_utils import bootstrap_world_if_needed

_extension_instance = None
_ext_id = None
_ext_path = None
_ext_version = None


def get_instance():
    return _extension_instance


def get_ext_id():
    return _ext_id


def get_ext_path():
    return _ext_path


def get_ext_version():
    return _ext_version


class Main(omni.ext.IExt):

    def on_startup(self, ext_id):
        import warp

        warp.init()

        ext_manager = omni.kit.app.get_app().get_extension_manager()

        # Set up global variables
        global _extension_instance
        _extension_instance = self
        global _ext_id
        _ext_id = ext_id
        global _ext_path
        _ext_path = ext_manager.get_extension_path(ext_id)
        global _ext_version
        _ext_version = str(ext_id).split("-")[-1]
        # Init Infos
        Infos.ext_version = _ext_version
        Infos.ext_path = _ext_path
        # Ensure Isaac World exists so that downstream controllers can register callbacks.
        try:
            bootstrap_world_if_needed()
        except Exception as exc:
            carb.log_error(f"[World] Unable to ensure Isaac World during startup: {exc}")
            raise
        # Handle async startup tasks
        self._is_startup_async_done = False
        self._startup_task = asyncio.ensure_future(self.startup_async())
        self._startup_task.add_done_callback(self.startup_async_done)
        # Ensure the global agent manager instance is initialized
        self._agent_manager = AgentManager.get_instance()
        # Install defensive patches for upstream components
        install_safe_simtimes_guard()
        # Create config file format
        GlobalValues.config_file_format = ConfigFileFormat(
            name="IRA config file format", required_header="isaacsim.replicator.agent", required_version=_ext_version
        )
        # Register self-defined sections
        GlobalValues.config_file_format.register_section(get_all_section_cls())
        # TODO:: decouple with IRI
        GlobalValues.config_file_format.register_section([IncidentEventSection])

        # To avoid IRI UI shows on top of IRA UI
        incident_ext = get_incident_ext_instance()
        if incident_ext:
            incident_ext.hide_windows()

    def on_shutdown(self):
        global _extension_instance
        _extension_instance = None
        global _ext_id
        _ext_id = None
        global _ext_path
        _ext_path = None

        if self._agent_manager:
            self._agent_manager = None

        # Config file format
        self._config_file_format = None

    async def startup_async(self):
        """
        Async startup tasks for this extension.
        """
        # Get asset paths from nucleus server
        if carb.settings.get_settings().get_as_bool(AssetPaths.USE_ISAAC_SIM_ASSET_ROOT_SETTING):
            await AssetPaths.cache_paths_async()

    def startup_async_done(self, context):
        self._is_startup_async_done = True
        self._startup_task = None  # Release handle

    def check_startup_async_done(self):
        return self._is_startup_async_done

    def add_startup_async_done_callback(self, callback: Callable):
        if self._startup_task:
            self._startup_task.add_done_callback(callback)

    def remove_startup_async_done_callback(self, callback: Callable):
        if self._startup_task:
            self._startup_task.remove_done_callback(callback)
