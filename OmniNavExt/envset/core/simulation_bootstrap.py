"""SimulationApp initialization and streaming configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from OmniNav.local_paths import apply_local_path_env, resolve_runtime_kit_path

if TYPE_CHECKING:
    from isaacsim import SimulationApp


@dataclass
class SimulationConfig:
    """SimulationApp configuration."""
    headless: bool = False
    native_streaming: bool = False
    webrtc_streaming: bool = False


class SimulationBootstrap:
    """Handles SimulationApp initialization and streaming configuration."""

    def __init__(self, config: SimulationConfig):
        self._config = config
        self._app: Optional[SimulationApp] = None

    @property
    def app(self) -> Optional[SimulationApp]:
        return self._app

    def initialize(self) -> "SimulationApp":
        """Initialize SimulationApp with configured settings."""
        from isaacsim import SimulationApp

        apply_local_path_env()
        launch_config = {
            "headless": self._config.headless,
            "anti_aliasing": 0,
            "hide_ui": False,
            "multi_gpu": False,
            # Force the repo-local kit for consistent runtime behavior.
            "experience": str(resolve_runtime_kit_path()),
        }

        self._app = SimulationApp(launch_config)
        # Disable collision cooking for better performance
        self._app._carb_settings.set("/physics/cooking/ujitsoCollisionCooking", False)

        self._configure_streaming()
        return self._app

    def _configure_streaming(self):
        """Configure streaming for Isaac Sim 5.0+."""
        if self._app is None:
            raise RuntimeError("SimulationApp not initialized")

        native = self._config.native_streaming
        webrtc = self._config.webrtc_streaming

        if native:
            print("[SimulationBootstrap] native streaming is deprecated, enabling webrtc instead.")

        _configure_streaming_500(self._app, native or webrtc)

    def shutdown(self):
        """Shutdown SimulationApp."""
        if self._app is not None:
            print("[SimulationBootstrap] shutting down Start")
            try:
                self._app.close()
            except Exception as e:
                print(f"[SimulationBootstrap] ERROR shutting down SimulationApp: {e}")
                pass
            self._app = None
            print("[SimulationBootstrap] shutting down End")


def _configure_streaming_500(sim_app, enable_webrtc: bool):
    """Configure streaming for Isaac Sim 5.0+."""
    if not enable_webrtc:
        return

    from omni.isaac.core.utils.extensions import enable_extension

    sim_app.set_setting("/app/window/drawMouse", True)
    try:
        enable_extension("omni.kit.livestream.webrtc")
    except Exception:
        enable_extension("omni.services.streamclient.webrtc")
