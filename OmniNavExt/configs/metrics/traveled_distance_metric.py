from typing import Optional

from OmniNav.core.config.metric import MetricCfg


class TraveledDistanceMetricCfg(MetricCfg):
    name: Optional[str] = 'traveled_distance_metric'
    type: Optional[str] = 'TraveledDistanceMetric'
    robot_name: str
