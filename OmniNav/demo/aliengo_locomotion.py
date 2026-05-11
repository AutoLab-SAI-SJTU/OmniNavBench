import sys
from pathlib import Path

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from OmniNav.core.config import Config, SimConfig
from OmniNav.core.gym_env import Env
from OmniNav.core.util import has_display
from OmniNav.local_paths import resolve_scene_root
from OmniNav.macros import gm
from OmniNavExt import import_extensions
from OmniNavExt.configs.robots.aliengo import (
    AliengoRobotCfg,
    move_to_point_cfg,
)
from OmniNavExt.configs.tasks import SingleInferenceTaskCfg

def main():
    headless = not has_display()
    scene_root = resolve_scene_root()
    assert scene_root is not None
    warehouse_scene = scene_root / "IsaacAssets" / "Environments" / "Simple_Warehouse" / "full_warehouse.usd"

    config = Config(
        simulator=SimConfig(
            physics_dt=1 / 240,
            rendering_dt=1 / 240,
            use_fabric=False,
            headless=headless,
            webrtc=headless,
        ),
        task_configs=[
            SingleInferenceTaskCfg(
                scene_asset_path=str(warehouse_scene),
                robots=[
                    AliengoRobotCfg(
                        position=(0.0, 0.0, 1.05),
                        controllers=[move_to_point_cfg],
                    )
                ],
            ),
        ],
    )

    import_extensions()

    env = Env(config)
    obs, _ = env.reset()

    i = 0
    env_action = {
        move_to_point_cfg.name: [(3.0, 3.0, 0.0)],
    }

    print(f'actions: {env_action}')

    while env.simulation_app.is_running():
        i += 1
        obs, _, terminated, _, _ = env.step(action=env_action)

        if i % 1000 == 0:
            print(i)

    env.close()


if __name__ == '__main__':
    main()
