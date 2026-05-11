from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FollowBandStats:
    total_steps: int
    violations: int
    violation_ratio: float
    min_distance: float
    max_distance: float
    mean_distance: float


@dataclass(frozen=True)
class PersonalSpaceStats:
    total_steps: int
    violations: int
    violation_ratio: float


def compute_follow_band_stats(
    *,
    robot_positions: Sequence[np.ndarray],
    human_positions: Sequence[np.ndarray],
    band_min_units: float,
    band_max_units: float,
) -> FollowBandStats:
    n = min(len(robot_positions), len(human_positions))
    if n <= 0:
        return FollowBandStats(total_steps=0, violations=0, violation_ratio=1.0, min_distance=0.0, max_distance=0.0, mean_distance=0.0)

    dists: List[float] = []
    violations = 0
    lo = float(min(band_min_units, band_max_units))
    hi = float(max(band_min_units, band_max_units))
    for r, h in zip(robot_positions[:n], human_positions[:n]):
        r = np.asarray(r, dtype=np.float32)
        h = np.asarray(h, dtype=np.float32)
        d = float(np.linalg.norm((r - h)[:2]))
        dists.append(d)
        if d < lo or d > hi:
            violations += 1

    arr = np.asarray(dists, dtype=np.float32)
    return FollowBandStats(
        total_steps=int(n),
        violations=int(violations),
        violation_ratio=float(violations / max(n, 1)),
        min_distance=float(arr.min()) if arr.size else 0.0,
        max_distance=float(arr.max()) if arr.size else 0.0,
        mean_distance=float(arr.mean()) if arr.size else 0.0,
    )


def detect_stop_events_from_idle_flags(
    *,
    idle_flags: Sequence[bool],
    start_frame: int,
    min_duration_steps: int = 1,
) -> List[Dict[str, int]]:
    """Convert per-step idle flags into stop events with frame ranges (inclusive)."""
    events: List[Dict[str, int]] = []
    if not idle_flags:
        return events
    min_steps = max(1, int(min_duration_steps))

    in_idle = False
    start = 0
    for i, flag in enumerate(idle_flags):
        if flag and not in_idle:
            in_idle = True
            start = i
        if in_idle and (not flag):
            end = i - 1
            if end - start + 1 >= min_steps:
                events.append({"start_frame": int(start_frame) + int(start), "end_frame": int(start_frame) + int(end)})
            in_idle = False
    if in_idle:
        end = len(idle_flags) - 1
        if end - start + 1 >= min_steps:
            events.append({"start_frame": int(start_frame) + int(start), "end_frame": int(start_frame) + int(end)})
    return events


def detect_stop_events_from_stop_flags(
    *,
    stop_flags: Sequence[bool],
    start_frame: int,
    dt: float,
    min_duration_steps: int = 1,
    segment_kinds: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Convert per-step stop flags into stop events aligned to frame/time axes.

    Returns a list of events:
      {"start_frame": int, "end_frame": int, "start_time_s": float, "end_time_s": float, "duration_s": float}
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    raw = detect_stop_events_from_idle_flags(
        idle_flags=stop_flags,
        start_frame=start_frame,
        min_duration_steps=min_duration_steps,
    )
    out: List[Dict[str, Any]] = []
    for idx, e in enumerate(raw):
        sf = int(e["start_frame"])
        ef = int(e["end_frame"])
        st = float(sf) * float(dt)
        et = float(ef) * float(dt)
        payload: Dict[str, Any] = {
            "start_frame": sf,
            "end_frame": ef,
            "start_time_s": st,
            "end_time_s": et,
            "duration_s": float(max(0.0, et - st)),
        }

        if segment_kinds is not None:
            # segment_kinds is per-step; map frame range back to step indices.
            s_idx = max(0, int(sf - int(start_frame)))
            e_idx = max(s_idx, int(ef - int(start_frame)))
            if s_idx >= len(segment_kinds) or e_idx >= len(segment_kinds):
                raise ValueError(
                    "segment_kinds length mismatch: "
                    f"len={len(segment_kinds)} but requested window [{s_idx}, {e_idx}] "
                    f"from frames [{sf}, {ef}] (start_frame={start_frame})"
                )
            window = [str(k) for k in segment_kinds[s_idx : e_idx + 1] if k]
            if window:
                uniq = sorted(set(window))
                payload["kind"] = uniq[0] if len(uniq) == 1 else "stop"
                payload["kinds"] = uniq

        payload["event_index"] = int(idx)
        out.append(payload)
    return out


def compute_personal_space_stats(
    *,
    robot_positions: Sequence[np.ndarray],
    human_positions: Sequence[np.ndarray],
    human_yaws_rad: Sequence[float],
    a_units: float,
    b_units: float,
) -> PersonalSpaceStats:
    """Simple forward-ellipse personal space in the human local frame.

    Violation when robot is in front half-plane (x>0) and inside ellipse:
      (x/a)^2 + (y/b)^2 <= 1
    """
    n = min(len(robot_positions), len(human_positions), len(human_yaws_rad))
    if n <= 0:
        return PersonalSpaceStats(total_steps=0, violations=0, violation_ratio=0.0)
    a = float(max(a_units, 1e-6))
    b = float(max(b_units, 1e-6))
    violations = 0
    for r, h, yaw in zip(robot_positions[:n], human_positions[:n], human_yaws_rad[:n]):
        r = np.asarray(r, dtype=np.float32)
        h = np.asarray(h, dtype=np.float32)
        dx, dy = float(r[0] - h[0]), float(r[1] - h[1])
        cy = float(np.cos(float(yaw)))
        sy = float(np.sin(float(yaw)))
        # rotate into human frame: x forward, y left
        x_f = dx * cy + dy * sy
        y_l = -dx * sy + dy * cy
        if x_f <= 0.0:
            continue
        val = (x_f / a) ** 2 + (y_l / b) ** 2
        if val <= 1.0:
            violations += 1
    return PersonalSpaceStats(
        total_steps=int(n),
        violations=int(violations),
        violation_ratio=float(violations / max(n, 1)),
    )
