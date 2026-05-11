from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datagen.pipeline import GeneratedEpisode


def _log_warn(msg: str) -> None:
    try:
        import carb  # type: ignore

        carb.log_warn(msg)
    except Exception:
        print(f"[WARN] {msg}")


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reasons: Tuple[str, ...] = ()


class EpisodeValidator:
    """Minimal validators to keep generation stable.

    Note: segmentation-based visibility checks are enforced inside the datagen pipeline
    (fail-fast during capture). This validator focuses on local file/waypoint invariants.
    """

    def validate(self, episode: GeneratedEpisode) -> ValidationResult:
        reasons: List[str] = []

        if not episode.instruction:
            reasons.append("empty_instruction")

        canonical_waypoints = []
        if isinstance(getattr(episode, "recording", None), dict):
            gt_path = episode.recording.get("gt_path")
            if isinstance(gt_path, list):
                canonical_waypoints = gt_path
        reasons.extend(self._validate_waypoints(canonical_waypoints or episode.gt_path))
        reasons.extend(self._validate_media(Path(episode.output_dir)))

        return ValidationResult(ok=(len(reasons) == 0), reasons=tuple(reasons))

    @staticmethod
    def _validate_waypoints(gt_path: List[Dict[str, Any]]) -> List[str]:
        if not gt_path:
            return ["missing_gt_path"]
        reasons: List[str] = []
        last_frame = None
        last_time = None
        for wp in gt_path:
            frame = wp.get("frame")
            if frame is None:
                reasons.append("waypoint_missing_frame")
                break
            if last_frame is not None and int(frame) <= int(last_frame):
                reasons.append("non_increasing_frames")
                break
            last_frame = int(frame)
            if "time_s" in wp:
                t = float(wp["time_s"])
                if last_time is not None and t < float(last_time):
                    reasons.append("non_increasing_time_s")
                    break
                last_time = t
        return reasons

    @staticmethod
    def _validate_media(episode_dir: Path) -> List[str]:
        if not episode_dir.exists():
            return ["missing_episode_dir"]
        rgb_video = episode_dir / "video" / "rgb.mp4"
        depth_video = episode_dir / "video" / "depth.mp4"
        if not rgb_video.is_file():
            _log_warn(f"[Validator] Missing standard RGB video under {episode_dir}")
            return ["missing_rgb_video"]
        if not depth_video.is_file():
            _log_warn(f"[Validator] Missing standard depth video under {episode_dir}")
            return ["missing_depth_video"]
        return []
