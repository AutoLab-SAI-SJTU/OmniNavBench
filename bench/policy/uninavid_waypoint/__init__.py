"""UniNaVid Waypoint policy package for OmniNavBench.

This package provides HTTP client-server integration for the Uni-NaVid
waypoint prediction model.

================================================================================
How to Run
================================================================================

1. Start the server (terminal 1, ``uni-navid`` env):

   conda activate uni-navid
   cd /path/to/Uni-NaVid_waypoints
   python -m bench.policy.uninavid_waypoint.uninavid_waypoint_server \
       --uninavid_path /path/to/Uni-NaVid_waypoints \
       --model_path /path/to/Uni-NaVid_waypoints/model_zoo/uninavid-7b-omninav-waypoint \
       --model_base /path/to/Uni-NaVid_waypoints/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2 \
       --port 8001 \
       --debug

2. Run the benchmark (terminal 2, ``isaaclab`` env):

   conda activate isaaclab
   cd $OMNINAV_REPO_ROOT

   # Single JSON file
   python runBench.py \
       --config configs/aliengoh1_test.yaml \
       --scene-root $OMNINAV_SCENE_ROOT \
       --envset $OMNINAV_REPO_ROOT/model_test_episode_aliengo.json \
       --output results/uninavid_waypoint_test/ \
       --policy uninavid_waypoint \
       --uninavid-waypoint-server-url http://localhost:8001 \
       --headless

   # Or pass a directory (all JSON files inside are iterated)
   python runBench.py \
       --config configs/aliengoh1_test.yaml \
       --scene-root $OMNINAV_SCENE_ROOT \
       --envset /path/to/dataset/dog \
       --output results/uninavid_waypoint_test/ \
       --policy uninavid_waypoint \
       --uninavid-waypoint-server-url http://localhost:8001 \
       --headless

3. Or use it directly from code:

   from bench.policy.uninavid_waypoint import UniNaVidWaypointHTTPPolicy
   policy = UniNaVidWaypointHTTPPolicy(server_url="http://localhost:8001")

================================================================================
"""

from .uninavid_waypoint_http_policy_action import UniNaVidWaypointHTTPPolicy
from .uninavid_waypoint_http_policy_points import UniNaVidWaypointPointsHTTPPolicy
from .robot_config import (
    configure_robot_sensors,
    get_required_cameras,
    get_execution_profile_override,
)

__all__ = [
    "UniNaVidWaypointHTTPPolicy",
    "UniNaVidWaypointPointsHTTPPolicy",
    "configure_robot_sensors",
    "get_required_cameras",
    "get_execution_profile_override",
]
