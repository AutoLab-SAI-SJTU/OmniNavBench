"""Tests for Offline Waypoint Comparison Framework.

Run with:
    pytest bench/policy/uninavid_waypoint/test_offline_waypoint.py -v
"""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest

from bench.policy.uninavid_waypoint.offline_waypoint_tester import (
    RobotWaypoint,
    EpisodeData,
    ComparisonResult,
    EpisodeMetrics,
    WaypointDataLoader,
    VideoLoader,
    CoordinateConverter,
    WaypointMetrics,
    OfflineWaypointTester,
    discover_episode_data,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_episode_json():
    """Build a sample episode JSON payload."""
    return {
        "scenarios": [{
            "id": "test_scenario",
            "robots": {
                "entries": [{
                    "type": "aliengo",
                    "initial_pose": {
                        "position": [0.0, 0.0, 0.0],
                        "orientation_deg": 0.0
                    }
                }]
            },
            "task": {
                "instruction": "Go forward"
            },
            "recording": {
                "instruction": "Go forward",
                "gt_path": [
                    {
                        "frame": 0,
                        "sim_step": 1,
                        "xyz": [0.0, 0.0, 0.0],
                        "yaw_deg": 0.0,
                        "time_s": 0.0,
                        "distance_xy": 0.0,
                        "distance_total_xy": 0.0,
                        "command": {"v": 0.0, "w": 0.0, "lateral": 0.0}
                    },
                    {
                        "frame": 10,
                        "sim_step": 11,
                        "xyz": [1.0, 0.0, 0.0],
                        "yaw_deg": 0.0,
                        "time_s": 0.1,
                        "distance_xy": 1.0,
                        "distance_total_xy": 1.0,
                        "command": {"v": 1.0, "w": 0.0, "lateral": 0.0}
                    },
                    {
                        "frame": 20,
                        "sim_step": 21,
                        "xyz": [2.0, 0.0, 0.0],
                        "yaw_deg": 0.0,
                        "time_s": 0.2,
                        "distance_xy": 1.0,
                        "distance_total_xy": 2.0,
                        "command": {"v": 1.0, "w": 0.0, "lateral": 0.0}
                    }
                ],
                "metadata": {"sample_count": 3, "distance_total_xy": 2.0}
            },
            "virtual_humans": {
                "vh_gt_waypoints": {}
            }
        }]
    }


@pytest.fixture
def temp_episode_file(sample_episode_json):
    """Write the sample episode JSON to a temp file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_episode_json, f)
        return f.name


@pytest.fixture
def bench_result_json(tmp_path):
    payload = {
        "scenario_id": "bench_episode",
        "instruction": "Walk to the table",
        "robot_type": "aliengo",
        "initial_pose": {
            "position": [0.0, 0.0, 0.0],
            "orientation_deg": 90.0,
        },
        "recording": {
            "instruction": "Walk to the table",
            "gt_path": [
                {
                    "frame": 0,
                    "sim_step": 1,
                    "xyz": [0.0, 0.0, 0.0],
                    "yaw_deg": 90.0,
                    "time_s": 0.1,
                    "distance_xy": 0.0,
                    "distance_total_xy": 0.0,
                },
                {
                    "frame": 1,
                    "sim_step": 2,
                    "xyz": [0.0, 1.0, 0.0],
                    "yaw_deg": 90.0,
                    "time_s": 0.2,
                    "distance_xy": 1.0,
                    "distance_total_xy": 1.0,
                },
            ],
            "metadata": {"sample_count": 2, "distance_total_xy": 1.0},
        },
    }
    scene_dir = tmp_path / "scene_01"
    scene_dir.mkdir(parents=True, exist_ok=True)
    json_path = scene_dir / "final_episode_1.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    video_dir = scene_dir / "video" / "final_episode_1" / "front"
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "rgb.mp4").write_bytes(b"")
    return json_path


@pytest.fixture
def temp_image_dir():
    """Build a temp directory of synthetic video frames."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate a 30-frame stand-in video.
        from PIL import Image
        for frame in range(30):
            img = Image.new('RGB', (640, 480), color='red')
            img.save(os.path.join(tmpdir, f"{frame:06d}.png"))
        yield tmpdir


# =============================================================================
# Unit Tests: Data Classes
# =============================================================================

class TestDataClasses:
    """Tests for the data classes."""

    def test_robot_waypoint_creation(self):
        wp = RobotWaypoint(
            frame=10,
            xyz=(1.0, 2.0, 0.0),
            yaw_deg=45.0,
            time_s=0.1
        )
        assert wp.frame == 10
        assert wp.xyz == (1.0, 2.0, 0.0)
        assert wp.yaw_deg == 45.0

    def test_episode_data_creation(self):
        episode = EpisodeData(
            scenario_id="test",
            instruction="Go forward",
            robot_type="aliengo",
            initial_position=(0.0, 0.0, 0.0),
            initial_yaw_deg=0.0,
            gt_path=[]
        )
        assert episode.scenario_id == "test"
        assert episode.instruction == "Go forward"


# =============================================================================
# Unit Tests: WaypointDataLoader
# =============================================================================

class TestWaypointDataLoader:
    """Tests for the data loader."""

    def test_load_episode(self, temp_episode_file):
        loader = WaypointDataLoader()
        episode = loader.load_episode(temp_episode_file)

        assert episode.scenario_id == "test_scenario"
        assert episode.instruction == "Go forward"
        assert episode.robot_type == "aliengo"
        assert len(episode.gt_path) == 3

        wp0 = episode.gt_path[0]
        assert wp0.frame == 0
        assert wp0.sim_step == 1
        assert wp0.xyz == (0.0, 0.0, 0.0)
        assert wp0.yaw_deg == 0.0

    def test_get_waypoint_at_frame(self, temp_episode_file):
        loader = WaypointDataLoader()
        episode = loader.load_episode(temp_episode_file)

        # Exact match.
        wp = loader.get_waypoint_at_frame(episode, 10)
        assert wp.frame == 10

        # Nearest match.
        wp = loader.get_waypoint_at_frame(episode, 15)
        assert wp.frame in [10, 20]  # Either nearest neighbor is acceptable.

    def test_load_bench_result_episode(self, bench_result_json):
        loader = WaypointDataLoader()
        episode = loader.load_episode(str(bench_result_json))

        assert episode.scenario_id == "bench_episode"
        assert episode.instruction == "Walk to the table"
        assert episode.initial_yaw_deg == 90.0
        assert [wp.frame for wp in episode.gt_path] == [0, 1]
        assert [wp.sim_step for wp in episode.gt_path] == [1, 2]

    def test_discover_episode_data_prefers_new_bench_layout(self, bench_result_json):
        episodes = discover_episode_data(str(bench_result_json.parent.parent))

        assert episodes == [
            (
                str(bench_result_json),
                str(bench_result_json.parent / "video" / bench_result_json.stem / "front" / "rgb.mp4"),
            )
        ]


# =============================================================================
# Unit Tests: VideoLoader
# =============================================================================

class TestVideoLoader:
    """Tests for the video/image loader."""

    def test_load_images_from_directory(self, temp_image_dir):
        loader = VideoLoader(temp_image_dir)
        frame_count = loader.load()

        assert frame_count == 30
        assert loader.frame_count == 30

    def test_get_frame(self, temp_image_dir):
        loader = VideoLoader(temp_image_dir)
        loader.load()

        frame = loader.get_frame(0)
        assert frame is not None
        assert frame.shape == (480, 640, 3)

    def test_get_frame_out_of_range(self, temp_image_dir):
        loader = VideoLoader(temp_image_dir)
        loader.load()

        frame = loader.get_frame(999)
        assert frame is None

    def test_get_frames_range(self, temp_image_dir):
        loader = VideoLoader(temp_image_dir)
        loader.load()

        frames = loader.get_frames_range(0, 5)
        assert len(frames) == 5


# =============================================================================
# Unit Tests: CoordinateConverter
# =============================================================================

class TestCoordinateConverter:
    """Tests for coordinate conversion."""

    def test_local_to_world_forward(self):
        """Robot moving forward."""
        local_waypoints = [[1.0, 0.0, 0.0]]  # 1 m forward.
        robot_pos = np.array([0.0, 0.0, 0.0])
        robot_yaw = 0.0  # Facing +X.

        world_points = CoordinateConverter.local_to_world(
            local_waypoints, robot_pos, robot_yaw
        )

        assert len(world_points) == 1
        assert abs(world_points[0][0] - 1.0) < 1e-6
        assert abs(world_points[0][1] - 0.0) < 1e-6

    def test_local_to_world_left(self):
        """Robot moving left."""
        local_waypoints = [[0.0, 1.0, 0.0]]  # 1 m to the left.
        robot_pos = np.array([0.0, 0.0, 0.0])
        robot_yaw = 0.0  # Facing +X.

        world_points = CoordinateConverter.local_to_world(
            local_waypoints, robot_pos, robot_yaw
        )

        assert len(world_points) == 1
        assert abs(world_points[0][0] - 0.0) < 1e-6
        assert abs(world_points[0][1] - 1.0) < 1e-6

    def test_local_to_world_rotated(self):
        """Conversion when the robot is rotated."""
        local_waypoints = [[1.0, 0.0, 0.0]]  # 1 m forward.
        robot_pos = np.array([0.0, 0.0, 0.0])
        robot_yaw = np.pi / 2  # Facing +Y (90 deg).

        world_points = CoordinateConverter.local_to_world(
            local_waypoints, robot_pos, robot_yaw
        )

        assert len(world_points) == 1
        assert abs(world_points[0][0] - 0.0) < 1e-6
        assert abs(world_points[0][1] - 1.0) < 1e-6

    def test_local_to_world_accumulated(self):
        """Accumulated displacement across waypoints."""
        local_waypoints = [
            [1.0, 0.0, 0.0],  # 1 m forward.
            [1.0, 0.0, 0.0],  # Another 1 m forward.
        ]
        robot_pos = np.array([0.0, 0.0, 0.0])
        robot_yaw = 0.0

        world_points = CoordinateConverter.local_to_world(
            local_waypoints, robot_pos, robot_yaw
        )

        assert len(world_points) == 2
        assert abs(world_points[0][0] - 1.0) < 1e-6
        assert abs(world_points[1][0] - 2.0) < 1e-6


# =============================================================================
# Unit Tests: WaypointMetrics
# =============================================================================

class TestWaypointMetrics:
    """Tests for metric computations."""

    def test_position_error(self):
        pred = (1.0, 0.0, 0.0)
        gt = (0.0, 0.0, 0.0)
        error = WaypointMetrics.position_error(pred, gt)
        assert abs(error - 1.0) < 1e-6

    def test_position_error_diagonal(self):
        pred = (1.0, 1.0, 0.0)
        gt = (0.0, 0.0, 0.0)
        error = WaypointMetrics.position_error(pred, gt)
        assert abs(error - np.sqrt(2)) < 1e-6

    def test_yaw_error(self):
        assert WaypointMetrics.yaw_error(45.0, 0.0) == 45.0
        assert WaypointMetrics.yaw_error(0.0, 45.0) == 45.0

    def test_yaw_error_wraparound(self):
        # Wrap-around cases.
        assert abs(WaypointMetrics.yaw_error(350.0, 10.0) - 20.0) < 1e-6
        assert abs(WaypointMetrics.yaw_error(-170.0, 170.0) - 20.0) < 1e-6

    def test_compute_episode_metrics_empty(self):
        metrics = WaypointMetrics.compute_episode_metrics([], "test")
        assert metrics.num_steps == 0
        assert metrics.mean_position_error_m == 0.0

    def test_compute_episode_metrics(self):
        results = [
            ComparisonResult(
                frame=0, time_s=0.0,
                gt_position=(0, 0, 0), gt_yaw_deg=0,
                pred_waypoints_local=[], pred_waypoints_world=[],
                position_error_m=1.0, yaw_error_deg=10.0,
                arrive_probs=[]
            ),
            ComparisonResult(
                frame=10, time_s=0.1,
                gt_position=(1, 0, 0), gt_yaw_deg=0,
                pred_waypoints_local=[], pred_waypoints_world=[],
                position_error_m=2.0, yaw_error_deg=20.0,
                arrive_probs=[]
            ),
        ]

        metrics = WaypointMetrics.compute_episode_metrics(results, "test")

        assert metrics.num_steps == 2
        assert metrics.mean_position_error_m == 1.5
        assert metrics.max_position_error_m == 2.0
        assert metrics.mean_yaw_error_deg == 15.0


# =============================================================================
# Integration Tests: OfflineWaypointTester
# =============================================================================

class TestOfflineWaypointTester:
    """Tests for the offline tester."""

    def test_init(self, temp_image_dir):
        tester = OfflineWaypointTester(
            server_url="http://localhost:8001",
            video_source=temp_image_dir
        )
        assert tester.server_url == "http://localhost:8001"
        assert tester.video_loader is not None

    @patch('requests.Session.get')
    def test_check_server_health_success(self, mock_get, temp_image_dir):
        mock_get.return_value = Mock(status_code=200)

        tester = OfflineWaypointTester(
            server_url="http://localhost:8001",
            video_source=temp_image_dir
        )
        assert tester.check_server_health() is True

    @patch('requests.Session.get')
    def test_check_server_health_failure(self, mock_get, temp_image_dir):
        mock_get.side_effect = Exception("Connection refused")

        tester = OfflineWaypointTester(
            server_url="http://localhost:8001",
            video_source=temp_image_dir
        )
        assert tester.check_server_health() is False

    @patch('requests.Session.post')
    def test_reset_server(self, mock_post, temp_image_dir):
        mock_post.return_value = Mock(status_code=200)

        tester = OfflineWaypointTester(
            server_url="http://localhost:8001",
            video_source=temp_image_dir
        )
        assert tester.reset_server("Go forward") is True

    @patch('requests.Session.post')
    def test_send_frame(self, mock_post, temp_image_dir):
        mock_response = Mock()
        mock_response.json.return_value = {
            "waypoints": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            "arrive_probs": [0.1, 0.2],
            "step": 1
        }
        mock_post.return_value = mock_response

        tester = OfflineWaypointTester(
            server_url="http://localhost:8001",
            video_source=temp_image_dir
        )

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = tester.send_frame(image, "Go forward")

        assert "waypoints" in result
        assert len(result["waypoints"]) == 2

    def test_save_report(self, temp_image_dir):
        tester = OfflineWaypointTester(
            server_url="http://localhost:8001",
            video_source=temp_image_dir
        )

        results = [
            EpisodeMetrics(
                scenario_id="test",
                num_steps=2,
                mean_position_error_m=1.0,
                max_position_error_m=2.0,
                std_position_error_m=0.5,
                mean_yaw_error_deg=10.0,
                max_yaw_error_deg=20.0,
                step_results=[]
            )
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name

        tester.save_report(results, output_path)

        with open(output_path, 'r') as f:
            report = json.load(f)

        assert report["num_episodes"] == 1
        assert report["summary"]["mean_position_error_m"] == 1.0
        assert len(report["episodes"]) == 1

        os.unlink(output_path)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
