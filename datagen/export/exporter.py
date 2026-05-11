from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from OmniNavExt.envset.recording import CANONICAL_RECORDING_KEY, write_recording_sidecar

if TYPE_CHECKING:
    from datagen.pipeline import GeneratedEpisode


def _log_info(msg: str) -> None:
    try:
        import carb  # type: ignore

        carb.log_info(msg)
    except Exception:
        print(msg)


@dataclass(frozen=True)
class ExportPaths:
    envset_json: Path


class EpisodeExporter:
    """Write generated episodes to disk using the canonical recording/media layout."""

    @staticmethod
    def _validate_standard_media_output(episode_dir: Path) -> None:
        rgb_video_path = episode_dir / "video" / "rgb.mp4"
        depth_video_path = episode_dir / "video" / "depth.mp4"
        if not rgb_video_path.is_file():
            raise RuntimeError(f"Missing standard RGB video output: {rgb_video_path}")
        if not depth_video_path.is_file():
            raise RuntimeError(f"Missing standard depth video output: {depth_video_path}")
        legacy_node_dirs = sorted(p for p in episode_dir.iterdir() if p.is_dir() and p.name.startswith("node_"))
        if legacy_node_dirs:
            raise RuntimeError(
                f"Legacy node directories should not be present in exported episode output: {legacy_node_dirs}"
            )
        temp_keyframe_dir = episode_dir / ".tmp_keyframes"
        if temp_keyframe_dir.exists():
            raise RuntimeError(f"Temporary keyframe directory should not remain in exported output: {temp_keyframe_dir}")

    def export(
        self,
        *,
        episode: GeneratedEpisode,
        template_scenario: Dict[str, Any],
        room_zone: Optional[Dict[str, Any]] = None,
    ) -> ExportPaths:
        episode_dir = Path(episode.output_dir)
        episode_dir.mkdir(parents=True, exist_ok=True)

        scenario = copy.deepcopy(template_scenario)
        scenario["id"] = f"generated_{episode.episode_id}"

        task = scenario.setdefault("task", {})
        nav = task.setdefault("navigation", {})
        nav["instruction"] = episode.instruction

        metadata = episode.metadata if isinstance(episode.metadata, dict) else {}
        task_types = [str(t).strip().lower() for t in (metadata.get("task_types") or []) if t]
        start_pose = metadata.get("start_pose")

        target = (episode.metadata or {}).get("target") if isinstance(episode.metadata, dict) else None
        objects: Dict[str, Any] = {}
        if isinstance(target, dict):
            object_id = target.get("object_id")
            cat = target.get("category")
            pos = target.get("position")
            if not object_id:
                raise ValueError("Generated target is missing object_id; external annotations must provide unique ids")
            if cat and isinstance(pos, (list, tuple)) and len(pos) >= 3:
                objects[str(object_id)] = [float(pos[0]), float(pos[1]), float(pos[2])]
                nav["goal_position"] = [float(pos[0]), float(pos[1]), float(pos[2])]
                nav["goal_type"] = "object"
                nav["goal_id"] = str(object_id)
        nav["objects"] = objects
        if isinstance(target, dict):
            nav["objects_meta"] = {"target": target}

        if room_zone is not None:
            nav["room_zone"] = room_zone

        subtask_type: Optional[str] = None
        if any(t in {"objectnav", "objnav"} for t in task_types):
            subtask_type = "GOTO_OBJECT"
        elif any(t == "vln" for t in task_types):
            subtask_type = "GOTO_LANDMARK"

        if isinstance(target, dict) and subtask_type is not None:
            object_id = str(target["object_id"])
            task["subtasks"] = [{"type": subtask_type, "object_id": object_id}]
            task["sub_instructions"] = [{"step": 0, "type": "VLN", "text": episode.instruction}]

        # FOLLOW/EQA payloads (optional)
        follow = (episode.metadata or {}).get("follow") if isinstance(episode.metadata, dict) else None
        if isinstance(follow, dict):
            vh = scenario.setdefault("virtual_humans", {})
            move_routes = follow.get("move_routes")
            if isinstance(move_routes, list) and move_routes:
                vh["move_routes"] = move_routes
                vh["routes"] = move_routes  # backward-compatible alias
            vh_gt = follow.get("vh_gt_waypoints")
            if isinstance(vh_gt, dict) and vh_gt:
                vh["vh_gt_waypoints"] = vh_gt
            stop_events = follow.get("stop_events")
            if isinstance(stop_events, dict) and stop_events:
                vh["stop_events"] = stop_events
            # Always emit move_routes to match the intended schema (even if empty).
            vh.setdefault("move_routes", move_routes if isinstance(move_routes, list) else [])
            vh.setdefault("routes", vh.get("move_routes"))
            # Keep lightweight follow stats under task.navigation for downstream consumers.
            nav["follow"] = {
                "human_name": follow.get("human_name"),
                "band_stats": follow.get("band_stats"),
                "personal_space": follow.get("personal_space"),
            }
            # Debug-only: observed stop events inferred from runtime command state.
            observed = follow.get("stop_events_observed")
            if isinstance(observed, dict) and observed:
                nav["follow_debug"] = {"stop_events_observed": observed}

        eqa = (episode.metadata or {}).get("eqa") if isinstance(episode.metadata, dict) else None
        if isinstance(eqa, dict):
            answer = eqa.get("answer")
            if answer is not None:
                nav["answer"] = answer
                # Backward compatibility: keep answer at scenario root too.
                scenario["answer"] = answer
            evidence = eqa.get("evidence_frames")
            if isinstance(evidence, list):
                nav["evidence_frames"] = evidence
            evidence_capture = eqa.get("evidence_capture_frames")
            if isinstance(evidence_capture, list):
                nav["evidence_capture_frames"] = evidence_capture

        vln = (episode.metadata or {}).get("vln") if isinstance(episode.metadata, dict) else None
        if isinstance(vln, dict):
            landmarks = vln.get("landmarks")
            if isinstance(landmarks, list):
                nav["landmarks"] = landmarks
            evidence = vln.get("landmarks_evidence")
            if isinstance(evidence, dict):
                nav["landmarks_evidence"] = evidence
            min_pixels = vln.get("min_pixels")
            if min_pixels is not None:
                nav["landmarks_min_pixels"] = min_pixels

        robots = scenario.setdefault("robots", {})
        entries = robots.get("entries") or []
        if not entries:
            raise ValueError("template scenario missing robots.entries")
        if isinstance(start_pose, dict):
            pos = start_pose.get("position")
            yaw_deg = start_pose.get("orientation_deg")
            if isinstance(pos, (list, tuple)) and len(pos) >= 3 and yaw_deg is not None:
                entries[0]["initial_pose"] = {
                    "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "orientation_deg": float(yaw_deg),
                }
        entries[0].pop("rb_gt_waypoints", None)
        recording = getattr(episode, "recording", None)
        if not isinstance(recording, dict):
            raise ValueError("GeneratedEpisode.recording must be populated before export")
        scenario[CANONICAL_RECORDING_KEY] = recording

        envset = {"scenarios": [scenario]}
        out_path = episode_dir / "episode.json"
        self._validate_standard_media_output(episode_dir)
        out_path.write_text(json.dumps(envset, indent=2, ensure_ascii=False), encoding="utf-8")
        write_recording_sidecar(out_path, recording)
        _log_info(f"[Exporter] Wrote {out_path}")
        return ExportPaths(envset_json=out_path)
