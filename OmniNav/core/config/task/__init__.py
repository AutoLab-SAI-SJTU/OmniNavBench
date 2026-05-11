from typing import List, Optional, Tuple

from pydantic import BaseModel, Extra

from OmniNav.core.config import ObjectCfg, RobotCfg
from OmniNav.core.config.metric import MetricCfg
from OmniNav.core.config.task.reward import RewardCfg


class TaskCfg(BaseModel, extra=Extra.allow):
    type: Optional[str] = None

    # inherit
    metrics: Optional[List[MetricCfg]] = []
    reward: Optional[RewardCfg] = None

    # path
    scene_root_path: Optional[str] = '/scene'
    robots_root_path: Optional[str] = '/robots'
    objects_root_path: Optional[str] = '/objects'

    # scene
    scene_asset_path: Optional[str] = None
    scene_units_in_meters: Optional[float] = None  # Scene units: 1 unit = X meters.
    scene_scale: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0)
    scene_position: Optional[Tuple[float, float, float]] = (0, 0, 0)
    scene_orientation: Optional[Tuple[float, float, float, float]] = (1.0, 0, 0, 0)

    # inherit
    robots: Optional[List[RobotCfg]] = []
    objects: Optional[List[ObjectCfg]] = []
