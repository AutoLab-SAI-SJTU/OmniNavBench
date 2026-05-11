from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass
class NavMeshConfig:
    """Configuration for NavMesh validation and usage."""
    agent_radius: float = 0.5
    agent_height: float = 1.0
    strict_check: bool = True  # If True, raise error on missing NavMesh
    min_clearance_m: Optional[float] = None  # If set, reject samples below this clearance (meters)

@dataclass
class RobotConfig:
    """Configuration for the robot agent."""
    name: str = "robot"
    # Defined min/max height for object filtering relative to camera
    min_interaction_height: float = 0.0
    max_interaction_height: float = 2.0

@dataclass
class CaptureConfig:
    """Configuration for execution and sensor capture."""

    fps: float = 30.0
    keyframe_interval_s: float = 0.5
    track_kp_pos: float = 2.0
    track_kp_yaw: float = 3.0
    track_lookahead_steps: int = 10
    track_smoothing_alpha: float = 0.25
    track_snap_distance_m: Optional[float] = 0.5
    track_snap_duration_s: float = 0.5
    track_snap_kp_scale: float = 0.2
    nominal_speed_ratio: float = 0.5

@dataclass
class FollowConfig:
    """Configuration for FOLLOW_HUMAN / social-style constraints.

    All distances/speeds are expressed in meters and converted to stage/env units
    using scenario.scene.units_in_meters at runtime.
    """

    # Nominal human speed used for kinematic route simulation (m/s)
    human_speed_mps: float = 1.0
    # Route generation for virtual humans (GoTo chain + optional Idle)
    route_points_min: int = 2
    route_points_max: int = 4
    route_min_step_m: float = 2.0
    route_max_step_m: float = 8.0
    route_idle_probability: float = 0.5
    route_idle_duration_s: float = 1.0
    # Robot should stay within [band_min_m, band_max_m] from the human (meters)
    band_min_m: float = 1.0
    band_max_m: float = 2.5
    # Desired following distance behind the human (meters)
    target_distance_m: float = 1.5
    # Allow at most this fraction of steps violating the band
    max_violation_ratio: float = 0.2
    # Waypoint downsampling for replay (physics frames)
    waypoint_stride_frames: int = 3
    # Virtual human waypoint compression threshold (meters)
    vh_distance_threshold_m: float = 0.05
    # Idle detection: minimum idle duration (seconds) to count as a stop_event
    min_stop_event_s: float = 0.5
    # Personal space ellipse (front half-plane) (meters)
    personal_space_a_m: float = 1.2
    personal_space_b_m: float = 0.6
    max_personal_space_violation_ratio: float = 0.05


@dataclass
class EQAConfig:
    """Configuration for EQA question/answer generation."""

    # For MVP: answer is the target category; evidence frames are those with visibility >= min_pixels_visible.
    max_evidence_frames: int = 5


@dataclass
class TaskConfig:
    """Configuration for task sampling."""
    types: List[str]  # e.g., ["VLN", "Follow"]
    chain_length: int = 1
    num_episodes: int = 10
    final_goal_threshold_m: float = 0.1
    pointnav_steps: int = 4
    pointnav_step_min_m: float = 0.5
    pointnav_step_max_m: float = 2.0
    pointnav_step_attempts: int = 30
    grid_size_m: float = 0.5
    object_goal_min_m: float = 0.5
    object_goal_max_m: float = 1.5
    object_start_min_dist_m: float = 0.0  # min XY distance from random ObjectNav start to target center
    random_start: bool = False      # randomise start position on navmesh each episode
    start_min_dist_m: float = 3.0  # min Euclidean distance between any two episode starts

@dataclass
class PipelineConfig:
    """Global configuration for the data generation pipeline."""
    output_dir: Path
    navmesh: NavMeshConfig
    robot: RobotConfig
    task: TaskConfig
    object_annotations_path: Optional[Path] = None
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    follow: FollowConfig = field(default_factory=FollowConfig)
    eqa: EQAConfig = field(default_factory=EQAConfig)
    random_seed: int = 0

    # Instruction generation backend:
    # - "template": deterministic, offline.
    # - "vlm": requires a configured VLM client and network access (fail-fast on any error).
    instruction_backend: str = "template"

    # Validation thresholds
    min_pixels_visible: int = 100

    # Strictness toggles (default: fail-fast)
    strict_instance_id_prim_path: bool = True
    
