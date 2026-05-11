"""VLN-CE adapter for the R2R VLN-CE dataset family.

Supported input shapes
----------------------
The adapter accepts any of these ``data_path`` values:

* ``.../VLN-CE``
* ``.../R2R_VLNCE_v1-3``
* ``.../R2R_VLNCE_v1-3_preprocessed``

When the main dataset root is ``R2R_VLNCE_v1-3``, the adapter will look for
ground-truth trajectories in the sibling ``R2R_VLNCE_v1-3_preprocessed``
directory when present.

How to run
----------
Convert one split to an envset JSON from the repository root::

    python convert_vlnce.py \
        --data-path /path/to/VLN-CE/R2R_VLNCE_v1-3 \
        --split val_seen \
        --output ./envsets/vlnce_val_seen.json

Convert a single episode for smoke testing::

    python convert_vlnce.py \
        --data-path /path/to/VLN-CE/R2R_VLNCE_v1-3 \
        --split val_seen \
        --episode-id 4 \
        --output ./envsets/vlnce_val_seen_ep4.json

Or call the adapter directly from Python::

    from bench.datasets import DatasetLoader
    loader = DatasetLoader.from_name("vlnce")
    loader.convert(
        data_path="/path/to/VLN-CE/R2R_VLNCE_v1-3",
        output_path="./envsets/vlnce_val_seen.json",
        split="val_seen",
    )
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from OmniNavExt.envset.recording import build_recording_payload

from ..base import DatasetAdapter
from ..registry import register_adapter
from ..schema import SubtaskSpec, UnifiedEpisode


def _normalize_angle_deg(yaw_deg: float) -> float:
    """Normalize an angle into [-180, 180]."""
    while yaw_deg <= -180.0:
        yaw_deg += 360.0
    while yaw_deg > 180.0:
        yaw_deg -= 360.0
    return yaw_deg


def _habitat_position_to_isaac(raw: Any) -> Optional[Tuple[float, float, float]]:
    """Convert Habitat coordinates [x, y, z] to Isaac coordinates [x, -z, y]."""
    try:
        x = float(raw[0])
        y = float(raw[1])
        z = float(raw[2])
    except (TypeError, IndexError, ValueError):
        return None
    return (x, -z, y)


def _habitat_yaw_deg_from_quat(raw: Any) -> float:
    """Extract Habitat yaw around the Y-up axis from quaternion [x, y, z, w]."""
    try:
        qx = float(raw[0])
        qy = float(raw[1])
        qz = float(raw[2])
        qw = float(raw[3])
    except (TypeError, IndexError, ValueError):
        return 0.0

    siny_cosp = 2.0 * (qw * qy + qx * qz)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def _habitat_quat_to_isaac_yaw_deg(raw: Any) -> float:
    """Convert Habitat quaternion [x, y, z, w] to Isaac envset yaw in degrees.

    OmniNavBench envsets use Isaac Z-up yaw, and the current Matterport robot
    assets are authored with a +Y forward convention. This produces the
    empirically validated mapping required by the existing envset sample and
    focused tests:

        isaac_yaw = normalize(90 - habitat_yaw)
    """
    habitat_yaw = _habitat_yaw_deg_from_quat(raw)
    return _normalize_angle_deg(90.0 - habitat_yaw)


def _segment_heading_deg(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    fallback_yaw_deg: float,
) -> float:
    """Heading of the XY segment from *start* to *end* in Isaac coordinates."""
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        return fallback_yaw_deg
    return _normalize_angle_deg(math.degrees(math.atan2(dy, dx)))


def _load_json_maybe_gz(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_split_file(root: Optional[Path], split: str, *, suffix: str = "") -> Optional[Path]:
    if root is None:
        return None
    stem = f"{split}{suffix}"
    split_dir = root / split
    for candidate in (split_dir / f"{stem}.json.gz", split_dir / f"{stem}.json"):
        if candidate.is_file():
            return candidate
    return None


def _resolve_episode_and_gt_roots(data_path: Path, split: str) -> Tuple[Path, Optional[Path]]:
    """Resolve dataset root for episodes and optional GT root."""
    root = Path(data_path)

    if root.name == "R2R_VLNCE_v1-3":
        episode_candidates = [root]
    elif root.name == "R2R_VLNCE_v1-3_preprocessed":
        episode_candidates = [root]
    else:
        episode_candidates = [
            root / "R2R_VLNCE_v1-3",
            root / "R2R_VLNCE_v1-3_preprocessed",
            root,
        ]

    episode_root = next(
        (candidate for candidate in episode_candidates if _resolve_split_file(candidate, split) is not None),
        None,
    )
    if episode_root is None:
        searched = ", ".join(str(candidate) for candidate in episode_candidates)
        raise FileNotFoundError(
            f"[VlnceAdapter] Could not find split='{split}' under any VLN-CE root: {searched}"
        )

    gt_candidates: List[Path] = []
    if episode_root.name == "R2R_VLNCE_v1-3_preprocessed":
        gt_candidates.append(episode_root)
    elif episode_root.name == "R2R_VLNCE_v1-3":
        gt_candidates.append(episode_root.parent / "R2R_VLNCE_v1-3_preprocessed")
        gt_candidates.append(episode_root)
    else:
        gt_candidates.append(episode_root)

    gt_root = next(
        (candidate for candidate in gt_candidates if _resolve_split_file(candidate, split, suffix="_gt") is not None),
        None,
    )
    return episode_root, gt_root


def _extract_scene_name(scene_id: str) -> str:
    """Extract scene name from strings like 'mp3d/Scene/Scene.glb'."""
    parts = str(scene_id).split("/")
    if len(parts) >= 2:
        return parts[-2]
    stem = Path(str(scene_id)).stem
    return stem or str(scene_id)


def _instruction_text(raw_instruction: Any) -> str:
    if isinstance(raw_instruction, dict):
        text = raw_instruction.get("instruction_text")
        return str(text or "")
    return str(raw_instruction or "")


def _build_recording_from_points(
    *,
    instruction: str,
    points: List[Tuple[float, float, float]],
    start_yaw_deg: float,
    metadata: Dict[str, Any],
    action_count: Optional[int] = None,
) -> Dict[str, Any]:
    gt_path: List[Dict[str, Any]] = []
    total_xy = 0.0
    sample_count = len(points)

    if sample_count <= 1:
        frames = [0] * sample_count
    else:
        max_frame = int(action_count) if action_count is not None else (sample_count - 1)
        frames = [int(round(i * max_frame / (sample_count - 1))) for i in range(sample_count)]

    prev_yaw = float(start_yaw_deg)
    prev_point: Optional[Tuple[float, float, float]] = None
    for idx, point in enumerate(points):
        if prev_point is None:
            distance_xy = 0.0
            yaw_deg = float(start_yaw_deg)
        else:
            dx = float(point[0] - prev_point[0])
            dy = float(point[1] - prev_point[1])
            distance_xy = math.hypot(dx, dy)
            total_xy += distance_xy
            yaw_deg = _segment_heading_deg(prev_point, point, prev_yaw)
        gt_path.append(
            {
                "frame": frames[idx],
                "xyz": [float(point[0]), float(point[1]), float(point[2])],
                "yaw_deg": float(yaw_deg),
                "time_s": float(frames[idx]),
                "distance_xy": float(distance_xy),
                "distance_total_xy": float(total_xy),
            }
        )
        prev_point = point
        prev_yaw = yaw_deg

    out_metadata = dict(metadata)
    out_metadata["distance_total_xy"] = float(total_xy)
    out_metadata["sample_count"] = sample_count
    return build_recording_payload(
        instruction=instruction,
        gt_path=gt_path,
        metadata=out_metadata,
    )


def _goal_from_episode(raw_episode: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float, float]], float]:
    goals = raw_episode.get("goals")
    if not isinstance(goals, list) or not goals:
        return None, 1.0

    goal = goals[0]
    if not isinstance(goal, dict):
        return None, 1.0

    goal_position = _habitat_position_to_isaac(goal.get("position"))
    try:
        success_radius = float(goal.get("radius", 1.0))
    except (TypeError, ValueError):
        success_radius = 1.0
    return goal_position, success_radius


@register_adapter
class VlnceAdapter(DatasetAdapter):
    """Adapter for the R2R VLN-CE dataset."""

    name = "vlnce"

    def load(
        self,
        data_path: Path,
        split: str = "val_seen",
        *,
        max_episodes: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[UnifiedEpisode]:
        if split == "test":
            raise ValueError("[VlnceAdapter] split='test' is not supported because it has no goals/GT path.")

        episode_root, gt_root = _resolve_episode_and_gt_roots(Path(data_path), split)
        episode_file = _resolve_split_file(episode_root, split)
        if episode_file is None:
            raise FileNotFoundError(
                f"[VlnceAdapter] Could not locate episode file for split='{split}' under {episode_root}"
            )

        raw_data = _load_json_maybe_gz(episode_file)
        raw_episodes = raw_data.get("episodes") if isinstance(raw_data, dict) else None
        if not isinstance(raw_episodes, list):
            raise ValueError(f"[VlnceAdapter] Missing 'episodes' list in {episode_file}")

        gt_by_episode: Dict[str, Any] = {}
        gt_file = _resolve_split_file(gt_root, split, suffix="_gt")
        if gt_file is not None:
            loaded_gt = _load_json_maybe_gz(gt_file)
            if isinstance(loaded_gt, dict):
                gt_by_episode = loaded_gt

        episodes: List[UnifiedEpisode] = []
        skipped = 0

        for raw_episode in raw_episodes:
            if max_episodes is not None and len(episodes) >= max_episodes:
                break
            if not isinstance(raw_episode, dict):
                skipped += 1
                continue

            raw_episode_id = str(raw_episode.get("episode_id", ""))
            if episode_id is not None and raw_episode_id != str(episode_id):
                continue

            try:
                episode = self._convert_episode(raw_episode, split=split, gt_entry=gt_by_episode.get(raw_episode_id))
            except Exception as exc:
                print(f"[WARN][VlnceAdapter] Skipping episode_id={raw_episode_id or '<missing>'}: {exc}")
                skipped += 1
                continue

            episodes.append(episode)

        print(
            f"[VlnceAdapter] Loaded {len(episodes)} episodes (skipped {skipped}) "
            f"from {episode_root} split={split}"
        )
        return episodes

    def _convert_episode(
        self,
        raw_episode: Dict[str, Any],
        *,
        split: str,
        gt_entry: Optional[Any],
    ) -> UnifiedEpisode:
        raw_episode_id = str(raw_episode.get("episode_id", ""))
        if not raw_episode_id:
            raise ValueError("missing episode_id")

        scene_name = _extract_scene_name(str(raw_episode.get("scene_id", "")))
        if not scene_name:
            raise ValueError(f"episode_id={raw_episode_id}: invalid scene_id")

        start_position = _habitat_position_to_isaac(raw_episode.get("start_position"))
        if start_position is None:
            raise ValueError(f"episode_id={raw_episode_id}: invalid start_position")

        start_rotation_deg = _habitat_quat_to_isaac_yaw_deg(raw_episode.get("start_rotation"))
        instruction = _instruction_text(raw_episode.get("instruction"))
        goal_position, success_radius = _goal_from_episode(raw_episode)
        if goal_position is None:
            raise ValueError(f"episode_id={raw_episode_id}: missing valid goal position")

        reference_points = [
            point
            for point in (
                _habitat_position_to_isaac(raw_point)
                for raw_point in raw_episode.get("reference_path", [])
            )
            if point is not None
        ]

        recording_payload: Optional[Dict[str, Any]] = None
        trajectory_id = str(raw_episode.get("trajectory_id", ""))
        info = raw_episode.get("info") if isinstance(raw_episode.get("info"), dict) else {}

        if isinstance(gt_entry, dict):
            gt_locations = [
                point
                for point in (
                    _habitat_position_to_isaac(raw_point)
                    for raw_point in gt_entry.get("locations", [])
                )
                if point is not None
            ]
            if not gt_locations:
                raise ValueError(f"episode_id={raw_episode_id}: GT exists but locations are empty")

            actions = gt_entry.get("actions", [])
            action_count = len(actions) if isinstance(actions, list) else None
            recording_payload = _build_recording_from_points(
                instruction=instruction,
                points=gt_locations,
                start_yaw_deg=start_rotation_deg,
                action_count=action_count,
                metadata={
                    "source": "vlnce_gt",
                    "split": split,
                    "episode_id": raw_episode_id,
                    "trajectory_id": trajectory_id,
                    "scene_id": scene_name,
                    "action_count": action_count or 0,
                    "forward_steps": int(gt_entry.get("forward_steps", 0)),
                    "geodesic_distance": float(info.get("geodesic_distance", 0.0)),
                },
            )
        elif reference_points:
            recording_payload = _build_recording_from_points(
                instruction=instruction,
                points=reference_points,
                start_yaw_deg=start_rotation_deg,
                metadata={
                    "source": "vlnce_reference_path",
                    "split": split,
                    "episode_id": raw_episode_id,
                    "trajectory_id": trajectory_id,
                    "scene_id": scene_name,
                    "geodesic_distance": float(info.get("geodesic_distance", 0.0)),
                },
            )

        if recording_payload is None:
            raise ValueError(f"episode_id={raw_episode_id}: no GT or usable reference_path available")

        return UnifiedEpisode(
            episode_id=f"vlnce_{split}_{raw_episode_id}",
            scene_id=scene_name,
            instruction=instruction,
            task_type="vln",
            scene_usd_path=f"matterport_usd/{scene_name}/{scene_name}.usd",
            scene_category="MP3D",
            units_in_meters=1.0,
            start_position=start_position,
            start_rotation_deg=start_rotation_deg,
            goal_position=goal_position,
            goal_object_id="vln_goal",
            goal_objects={"vln_goal": list(goal_position)},
            success_radius=success_radius,
            subtasks=[SubtaskSpec(type="GOTO_LANDMARK", target_id="vln_goal")],
            sub_instructions=[{"step": 0, "type": "VLN", "text": instruction}],
            reference_path=reference_points,
            recording_payload=recording_payload,
            extra={
                "raw_episode_id": raw_episode_id,
                "trajectory_id": trajectory_id,
                "source_scene_id": str(raw_episode.get("scene_id", "")),
            },
        )
