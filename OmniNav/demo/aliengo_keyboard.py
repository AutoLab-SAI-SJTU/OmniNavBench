import sys
from pathlib import Path

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from OmniNav.core.config import Config, SimConfig
from OmniNav.core.gym_env import Env
from OmniNav.core.util import has_display
from OmniNav.macros import gm
from OmniNav.local_paths import resolve_scene_root
from OmniNavExt import import_extensions
from OmniNavExt.configs.robots.aliengo import (
    AliengoRobotCfg,
    move_by_speed_cfg,
)
from OmniNavExt.configs.tasks import SingleInferenceTaskCfg
from OmniNavExt.interactions.keyboard import KeyboardInteraction


def main():
    # On Windows, prefer GUI (keyboard needs a window); on UNIX decide by DISPLAY
    headless = False if sys.platform.startswith('win') else not has_display()
    scene_root = resolve_scene_root()
    assert scene_root is not None
    warehouse_scene = scene_root / "IsaacAssets" / "Environments" / "Simple_Warehouse" / "full_warehouse.usd"

    # Avoid WebRTC to keep compatibility across Isaac Sim 4.5/5.0 by default.
    config = Config(
        simulator=SimConfig(
            physics_dt=1 / 240,
            rendering_dt=1 / 240,
            use_fabric=False,
            headless=headless,
            webrtc=False,
        ),
        task_configs=[
            SingleInferenceTaskCfg(
                scene_asset_path=str(warehouse_scene),
                robots=[
                    AliengoRobotCfg(
                        position=(0.0, 0.0, 1.05),
                        controllers=[move_by_speed_cfg],
                    )
                ],
            ),
        ],
    )

    import_extensions()

    env = Env(config)
    env.reset()

    keyboard = KeyboardInteraction()
    i = 0
    while env.simulation_app.is_running():
        i += 1
        # map I/K, J/L, U/O into (x, y, yaw) velocities
        command = keyboard.get_input()
        x_speed = float(command[0] - command[1])
        y_speed = float(command[2] - command[3])
        z_speed = float(command[4] - command[5])
        env_action = {move_by_speed_cfg.name: (x_speed, y_speed, z_speed)}
        env.step(action=env_action)

    env.close()


if __name__ == '__main__':
    main()
