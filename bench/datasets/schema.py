"""Unified episode schema for dataset adapters.

All dataset adapters convert their native format into UnifiedEpisode,
which can then be serialised to the envset JSON format consumed by BenchRunner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from OmniNavExt.envset.recording import build_recording_payload

if TYPE_CHECKING:
    from .base import RobotDefaults


# ---------------------------------------------------------------------------
# Sub-structures
# ---------------------------------------------------------------------------

@dataclass
class SubtaskSpec:
    """Represents a single navigation subtask.

    Attributes:
        type: One of GOTO_OBJECT | GOTO_ROOM | GOTO_LANDMARK | FOLLOW_HUMAN | RETURN_TO
        target_id: Object / room / landmark ID (for GOTO_* and RETURN_TO)
        human_id: Virtual-human ID (for FOLLOW_HUMAN)
    """
    type: str
    target_id: Optional[str] = None
    human_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        st_type = self.type.upper()
        d: Dict[str, Any] = {"type": st_type}
        if st_type == "FOLLOW_HUMAN":
            if self.human_id:
                d["human_id"] = self.human_id
        elif st_type in {"GOTO_OBJECT", "GOTO_LANDMARK", "RETURN_TO"}:
            if self.target_id:
                d["object_id"] = self.target_id
        elif st_type == "GOTO_ROOM":
            if self.target_id:
                d["room_id"] = self.target_id
        return d


# ---------------------------------------------------------------------------
# Core unified episode
# ---------------------------------------------------------------------------

@dataclass
class UnifiedEpisode:
    """Canonical representation of one navigation episode.

    This is the intermediate layer between raw dataset formats and the
    envset JSON format that BenchRunner consumes.

    Attributes:
        episode_id: Unique identifier for this episode.
        scene_id: Scene identifier (used for grouping / lookup).
        instruction: Natural language navigation instruction.
        task_type: High-level task category: "vln" | "objectnav" | "follow" | "eqa".
        scene_usd_path: Absolute path to the Isaac Sim USD/USDA file.
        scene_category: "MP3D" for Matterport, "CustomUSD" for custom scenes.
        units_in_meters: Scene unit scale (1.0 = meters).
        start_position: Robot spawn position (x, y, z).
        start_rotation_deg: Robot spawn yaw in degrees.
        goal_position: Target navigation goal (x, y, z).
        goal_object_id: Primary goal object identifier (used in GOTO_OBJECT subtask).
        goal_objects: Dict mapping object name → [x, y, z] for task.navigation.objects.
        subtasks: Ordered list of SubtaskSpec.
        sub_instructions: List of sub-instruction dicts for task.sub_instructions.
        success_radius: Success threshold stored in task.navigation.success_radius.
        reference_path: Ground-truth trajectory waypoints [(x, y, z), ...].
        recording_payload: Optional canonical recording payload to emit verbatim.
        extra: Adapter-specific pass-through data (e.g. raw_scenario, qa).
    """

    episode_id: str
    scene_id: str
    instruction: str
    task_type: str

    # Scene
    scene_usd_path: Optional[str]
    scene_category: str = "MP3D"
    units_in_meters: float = 1.0

    # Robot pose
    start_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    start_rotation_deg: float = 0.0

    # Goal
    goal_position: Optional[Tuple[float, float, float]] = None
    goal_object_id: Optional[str] = None
    goal_objects: Dict[str, List[float]] = field(default_factory=dict)
    success_radius: float = 1.0

    # Task structure
    subtasks: List[SubtaskSpec] = field(default_factory=list)
    sub_instructions: List[Dict[str, Any]] = field(default_factory=list)

    # Reference trajectory (for SPL / NDTW computation)
    reference_path: List[Tuple[float, float, float]] = field(default_factory=list)
    recording_payload: Optional[Dict[str, Any]] = None

    # Pass-through / adapter-specific data
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_envset_scenario(self, robot_defaults: "RobotDefaults") -> Dict[str, Any]:
        """Convert to an envset JSON scenario dict consumable by BenchRunner.

        The produced dict follows the minimal required field set documented in
        the dataset schema plan.  Optional blocks (virtual_humans, canonical recording,
        qa) are only included when the corresponding data is present.

        Args:
            robot_defaults: Robot and navmesh configuration to embed.

        Returns:
            A scenario dict that can be placed inside {"scenarios": [...]}.
        """
        # NativeAdapter direct pass-through
        if "raw_scenario" in self.extra:
            return self.extra["raw_scenario"]

        goal_pos = list(self.goal_position) if self.goal_position else [0.0, 0.0, 0.0]

        scenario: Dict[str, Any] = {
            "id": self.episode_id,
            "scene": self._build_scene_block(),
            "navmesh": self._build_navmesh_block(robot_defaults),
            "robots": self._build_robots_block(robot_defaults),
            "task": self._build_task_block(goal_pos),
        }
        if isinstance(self.recording_payload, dict):
            scenario["recording"] = self.recording_payload
        elif self.reference_path:
            scenario["recording"] = build_recording_payload(
                instruction=self.instruction,
                gt_path=self._build_gt_waypoints(),
                metadata={"source": "dataset_adapter"},
            )
        return scenario

    # ------------------------------------------------------------------
    # Private block builders
    # ------------------------------------------------------------------

    def _build_scene_block(self) -> Dict[str, Any]:
        if self.scene_category == "MP3D":
            return {
                "category": "MP3D",
                "type": "matterport",
                "use_matterport": True,
                "manage_simulation": False,
                "units_in_meters": self.units_in_meters,
                "matterport": {
                    "usd_path": self.scene_usd_path or "",
                    "obj_path": "",
                    "root_prim_path": "/World/terrain",
                    "add_ground_plane": True,
                },
            }
        else:
            return {
                "category": "CustomUSD",
                "type": "usd_stage",
                "usd_path": self.scene_usd_path or "",
                "root_prim_path": "/World",
                "navmesh_root_prim_path": "/World",
                "units_in_meters": self.units_in_meters,
            }

    def _build_navmesh_block(self, rd: "RobotDefaults") -> Dict[str, Any]:
        block: Dict[str, Any] = {
            "bake_root_prim_path": rd.navmesh_bake_root_prim_path,
            "include_volume_parent": "/World/NavMesh",
            "z_padding": rd.navmesh_z_padding,
            "agent_radius": rd.navmesh_agent_radius,
        }
        if rd.navmesh_max_step_height is not None:
            block["max_step_height"] = rd.navmesh_max_step_height
        if rd.navmesh_min_include_volume_size is not None:
            block["min_include_volume_size"] = rd.navmesh_min_include_volume_size
        if rd.navmesh_spawn_min_separation_m is not None:
            block["spawn_min_separation_m"] = rd.navmesh_spawn_min_separation_m
        return block

    def _build_robots_block(self, rd: "RobotDefaults") -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "type": rd.robot_type,
            "label": rd.robot_label or rd.robot_type,
            "spawn_path": rd.spawn_path,
            "usd_path": rd.usd_path,
            "initial_pose": {
                "position": list(self.start_position),
                "orientation_deg": self.start_rotation_deg,
            },
            "control": {
                "mode": rd.control_mode,
                "params": {
                    "base_velocity": rd.base_velocity,
                    "base_turn_rate": rd.base_turn_rate,
                },
            },
        }
        return {"entries": [entry]}

    def _build_gt_waypoints(self) -> List[Dict[str, Any]]:
        """Build canonical gt_path samples from reference_path."""
        waypoints = []
        for i, (x, y, z) in enumerate(self.reference_path):
            waypoints.append({
                "frame": i * 10,
                "xyz": [x, y, z],
                "yaw_deg": 0.0,
                "time_s": float(i),
            })
        return waypoints

    def _build_task_block(self, goal_pos: List[float]) -> Dict[str, Any]:
        objects = dict(self.goal_objects)
        # Ensure goal_object_id appears in objects if not already present
        if self.goal_object_id and self.goal_object_id not in objects and self.goal_position:
            objects[self.goal_object_id] = list(self.goal_position)

        subtasks_raw = [st.to_dict() for st in self.subtasks]
        sub_instructions_raw = list(self.sub_instructions)

        # Build default sub_instructions if none provided
        if not sub_instructions_raw and self.instruction:
            sub_instructions_raw = [{"step": 0, "type": "VLN", "text": self.instruction}]

        task: Dict[str, Any] = {
            "navigation": {
                "instruction": self.instruction,
                "goal_position": goal_pos,
                "objects": objects,
                "room_zone": {},
                "success_radius": float(self.success_radius),
            },
            "subtasks": subtasks_raw,
            "sub_instructions": sub_instructions_raw,
        }

        # EQA support
        if "qa" in self.extra:
            task["qa"] = self.extra["qa"]

        return task
