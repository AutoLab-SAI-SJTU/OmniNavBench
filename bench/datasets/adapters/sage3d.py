"""Sage3DAdapter — adapter for the SAGE-3D VLN dataset.

Dataset layout::

    {data_root}/{split}/Trajectory_{split}/{scene_id}/
        train_trajectories_overall_*.json

Each JSON has the structure::

    {
      "Data_2set_metaData_2": {...},
      "scenes": [
        {
          "scene_id": "839873",
          "scene_name": "839873",
          "samples": [
            {
              "trajectory_id": "0",
              "instructions": [
                {
                  "instruction_type": "Add_Causality",
                  "start": "pen_holder_1",
                  "end": "projection_screen_1",
                  "generated_instruction": "...",
                  "scene_id": "0033_839873"
                },
                ...   # Multiple types, all share the same trajectory (points)
              ],
              "points": [
                {
                  "point": "0",
                  "position": [x, y, z],
                  "rotation": [qx, qy, qz, qw],   # quaternion
                  ...
                },
                ...
              ]
            }
          ]
        }
      ]
    }

Key insight
-----------
All ``instructions`` within one sample share the **same** ``points``
(i.e. the same physical trajectory).  The different instruction entries
are merely alternative linguistic descriptions.

Instruction selection (via ``instruction_type`` / ``all_instructions``)
-----------------------------------------------------------------------
* Default (``instruction_type=None``, ``all_instructions=False``):
  Take the **first** instruction in the list → 1 episode per trajectory.
* ``instruction_type="Add_Causality"``: Keep only instructions of that
  type → one episode per matching instruction per trajectory.
* ``all_instructions=True``: Generate one episode per instruction entry
  (episode_id gets a ``_iN`` suffix to stay unique).

USD path
--------
``scene_usd_root`` **must** be supplied.  The USD file is resolved as::

    {scene_usd_root}/{scene_id}.usda

Raises a ``ValueError`` if not provided.

How to run
----------
Convert a SAGE-3D split to an envset JSON from the repository root::

    python - <<'PY'
    from bench.datasets import DatasetLoader

    loader = DatasetLoader.from_name("sage3d")
    loader.convert(
        data_path="/path/to/SAGE-3D_VLN_Data",
        output_path="./envsets/sage3d_train.json",
        split="train",
        scene_usd_root="/path/to/Sage-3D-usda",
    )
    PY

Filter by instruction type::

    python - <<'PY'
    from bench.datasets import DatasetLoader

    loader = DatasetLoader.from_name("sage3d")
    loader.convert(
        data_path="/path/to/SAGE-3D_VLN_Data",
        output_path="./envsets/sage3d_add_causality.json",
        split="train",
        scene_usd_root="/path/to/Sage-3D-usda",
        instruction_type="Add_Causality",
        max_episodes=100,
    )
    PY
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..base import DatasetAdapter, RobotDefaults
from ..registry import register_adapter
from ..schema import SubtaskSpec, UnifiedEpisode


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _quat_to_yaw_deg(qx: float, qy: float, qz: float, qw: float) -> float:
    """Convert a quaternion [x, y, z, w] to yaw angle in degrees (Z-up)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def _parse_position(raw: Any) -> Optional[Tuple[float, float, float]]:
    """Parse a [x, y, z] list into a typed tuple, or return None on failure."""
    try:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    except (TypeError, IndexError, ValueError):
        return None


def _parse_rotation_yaw(raw: Any) -> float:
    """Parse a [qx, qy, qz, qw] list into a yaw angle in degrees."""
    try:
        return _quat_to_yaw_deg(
            float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])
        )
    except (TypeError, IndexError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

@register_adapter
class Sage3DAdapter(DatasetAdapter):
    """Adapter for the SAGE-3D VLN dataset.

    See module docstring for dataset format details.
    """

    name = "sage3d"

    def load(
        self,
        data_path: Path,
        split: str = "train",
        *,
        scene_usd_root: Optional[str],
        instruction_type: Optional[str] = None,
        all_instructions: bool = False,
        max_episodes: Optional[int] = None,
        **kwargs: Any,
    ) -> List[UnifiedEpisode]:
        """Load episodes from a SAGE-3D VLN dataset directory.

        Args:
            data_path: Root of the SAGE-3D VLN Data directory.
            split: "train" or "test".
            scene_usd_root: **Required**. Directory containing
                ``{scene_id}.usda`` files (e.g.
                ``/home/user/Assets/sage/Sage-3D-usda``).
            instruction_type: If given, only instructions with this
                ``instruction_type`` value are used.  Examples:
                ``"Add_Causality"``, ``"Scenario_Driven"``,
                ``"Relative_Relationship"``, ``"Attribute-based"``,
                ``"Area-based"``.
            all_instructions: If True, each instruction entry in a sample
                produces a separate episode (episode_id gets ``_iN`` suffix).
                Ignored when ``instruction_type`` is set; in that case all
                matching instructions produce separate episodes.
            max_episodes: Cap on total episodes returned.

        Returns:
            List of UnifiedEpisode objects.

        Raises:
            ValueError: If ``scene_usd_root`` is not provided.
            FileNotFoundError: If the expected trajectory directory does not exist.
        """
        if scene_usd_root is None:
            raise ValueError(
                "[Sage3DAdapter] 'scene_usd_root' must be provided. "
                "Example: scene_usd_root='/home/user/Assets/sage/Sage-3D-usda'. "
                "This directory should contain files named '{scene_id}.usda'."
            )

        usd_root = Path(scene_usd_root)
        if not usd_root.exists():
            raise FileNotFoundError(
                f"[Sage3DAdapter] scene_usd_root does not exist: {usd_root}"
            )

        traj_dir = Path(data_path) / split / f"Trajectory_{split}"
        if not traj_dir.exists():
            raise FileNotFoundError(
                f"[Sage3DAdapter] Trajectory directory not found: {traj_dir}. "
                f"Check data_path and split arguments."
            )

        json_files = sorted(traj_dir.rglob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"[Sage3DAdapter] No JSON files found under: {traj_dir}"
            )

        episodes: List[UnifiedEpisode] = []
        skipped = 0

        for json_path in json_files:
            if max_episodes is not None and len(episodes) >= max_episodes:
                break
            try:
                loaded, sk = self._load_file(
                    json_path,
                    usd_root=usd_root,
                    instruction_type=instruction_type,
                    all_instructions=all_instructions,
                    remaining=(
                        max_episodes - len(episodes)
                        if max_episodes is not None
                        else None
                    ),
                )
                episodes.extend(loaded)
                skipped += sk
            except Exception as exc:
                print(f"[WARN][Sage3DAdapter] Skipping {json_path}: {exc}")
                skipped += 1

        if max_episodes is not None:
            episodes = episodes[:max_episodes]

        print(
            f"[Sage3DAdapter] Loaded {len(episodes)} episodes "
            f"(skipped {skipped} malformed entries) "
            f"from {traj_dir}"
        )
        return episodes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_file(
        self,
        json_path: Path,
        *,
        usd_root: Path,
        instruction_type: Optional[str],
        all_instructions: bool,
        remaining: Optional[int],
    ) -> Tuple[List[UnifiedEpisode], int]:
        """Parse one SAGE-3D trajectory JSON file."""
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, dict) or "scenes" not in data:
            raise ValueError(f"Missing 'scenes' key in {json_path}")

        episodes: List[UnifiedEpisode] = []
        skipped = 0

        for scene_entry in data["scenes"]:
            scene_id = str(scene_entry.get("scene_id", "unknown"))
            usd_path = str(usd_root / f"{scene_id}.usda")

            for sample in scene_entry.get("samples", []):
                if remaining is not None and len(episodes) >= remaining:
                    return episodes, skipped

                traj_id = str(sample.get("trajectory_id", "0"))
                points = sample.get("points", [])
                instructions = sample.get("instructions", [])

                if not points:
                    print(
                        f"[WARN][Sage3DAdapter] scene={scene_id} traj={traj_id}: "
                        f"empty points list, skipping."
                    )
                    skipped += 1
                    continue

                if not instructions:
                    print(
                        f"[WARN][Sage3DAdapter] scene={scene_id} traj={traj_id}: "
                        f"no instructions, skipping."
                    )
                    skipped += 1
                    continue

                # Filter instructions
                selected = self._select_instructions(
                    instructions, instruction_type, all_instructions
                )
                if not selected:
                    print(
                        f"[WARN][Sage3DAdapter] scene={scene_id} traj={traj_id}: "
                        f"no instructions match instruction_type={instruction_type!r}, skipping."
                    )
                    skipped += 1
                    continue

                # Parse trajectory geometry (shared across all instructions)
                start_pt = points[0]
                end_pt = points[-1]

                start_pos = _parse_position(start_pt.get("position"))
                start_rot_deg = _parse_rotation_yaw(start_pt.get("rotation", [0, 0, 0, 1]))
                goal_pos = _parse_position(end_pt.get("position"))

                if start_pos is None or goal_pos is None:
                    print(
                        f"[WARN][Sage3DAdapter] scene={scene_id} traj={traj_id}: "
                        f"invalid position data, skipping."
                    )
                    skipped += 1
                    continue

                reference_path = []
                for pt in points:
                    pos = _parse_position(pt.get("position"))
                    if pos is not None:
                        reference_path.append(pos)

                # Object IDs from first instruction (all share same start/end)
                first_instr = selected[0]
                start_obj_id = str(first_instr.get("start", ""))
                end_obj_id = str(first_instr.get("end", ""))

                goal_objects: Dict[str, List[float]] = {}
                if end_obj_id and goal_pos:
                    goal_objects[end_obj_id] = list(goal_pos)

                subtasks = []
                if end_obj_id:
                    subtasks.append(SubtaskSpec(type="GOTO_OBJECT", target_id=end_obj_id))

                # Build one episode per selected instruction
                for instr_idx, instr in enumerate(selected):
                    ep_suffix = f"_i{instr_idx}" if len(selected) > 1 else ""
                    ep_id = f"sage3d_{scene_id}_t{traj_id}{ep_suffix}"
                    instruction_text = str(instr.get("generated_instruction", ""))
                    instr_type_val = str(instr.get("instruction_type", ""))

                    sub_instructions = [
                        {"step": 0, "type": "VLN", "text": instruction_text}
                    ]

                    ep = UnifiedEpisode(
                        episode_id=ep_id,
                        scene_id=scene_id,
                        instruction=instruction_text,
                        task_type="vln",
                        scene_usd_path=usd_path,
                        scene_category="CustomUSD",
                        units_in_meters=1.0,
                        start_position=start_pos,
                        start_rotation_deg=start_rot_deg,
                        goal_position=goal_pos,
                        goal_object_id=end_obj_id or None,
                        goal_objects=goal_objects,
                        subtasks=list(subtasks),
                        sub_instructions=sub_instructions,
                        reference_path=reference_path,
                        extra={
                            "instruction_type": instr_type_val,
                            "start_object": start_obj_id,
                            "end_object": end_obj_id,
                            "source_scene_id": instr.get("scene_id", ""),
                        },
                    )
                    episodes.append(ep)

        return episodes, skipped

    @staticmethod
    def _select_instructions(
        instructions: List[Dict[str, Any]],
        instruction_type: Optional[str],
        all_instructions: bool,
    ) -> List[Dict[str, Any]]:
        """Return the subset of instructions to use for episode generation."""
        if instruction_type is not None:
            filtered = [
                ins for ins in instructions
                if str(ins.get("instruction_type", "")) == instruction_type
            ]
            return filtered  # May be empty; caller warns and skips

        if all_instructions:
            return list(instructions)

        # Default: first instruction only
        return instructions[:1]
