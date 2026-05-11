from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from datagen.core.geometry import GeometryInterface


@dataclass(frozen=True)
class RouteSegment:
    kind: str  # "goto" | "idle" | "lookaround"
    target: Optional[np.ndarray] = None
    duration_s: Optional[float] = None


def parse_route_segments(commands: Sequence[str], *, agent_name: str) -> List[RouteSegment]:
    """Parse envset virtual_humans move_routes commands into structured segments.

    Expected command formats (examples):
      - "Character GoTo x y z _"
      - "Character Idle 1.0"
      - "Character LookAround 1.0"
    """
    segments: List[RouteSegment] = []
    for raw in commands:
        if not raw:
            continue
        tokens = str(raw).strip().split()
        if len(tokens) < 2:
            continue
        if str(tokens[0]) != str(agent_name):
            continue
        cmd = str(tokens[1]).strip().lower()
        if cmd in {"idle", "lookaround"}:
            dur = 1.0
            if len(tokens) >= 3:
                try:
                    dur = float(tokens[2])
                except Exception:
                    dur = 1.0
            segments.append(RouteSegment(kind=cmd, duration_s=float(max(dur, 0.0))))
            continue

        if cmd == "goto" or cmd.startswith("goto"):
            if len(tokens) < 5:
                continue
            try:
                x = float(tokens[2])
                y = float(tokens[3])
                z = float(tokens[4])
            except Exception:
                continue
            segments.append(RouteSegment(kind="goto", target=np.asarray([x, y, z], dtype=np.float32)))
            continue

    return segments


def simulate_route_positions(
    *,
    geometry: GeometryInterface,
    start_pos: np.ndarray,
    segments: Sequence[RouteSegment],
    dt: float,
    speed_units_per_s: float,
) -> List[np.ndarray]:
    """Simulate agent positions over time for a route description.

    This is a kinematic approximation used to synthesize a time-indexed reference
    for FOLLOW tasks.
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if speed_units_per_s <= 0:
        raise ValueError("speed_units_per_s must be > 0")

    cur = np.asarray(start_pos, dtype=np.float32)
    cur = geometry.snap_point(cur)
    positions: List[np.ndarray] = [cur.copy()]

    step_dist = float(speed_units_per_s) * float(dt)
    step_dist = max(step_dist, 1e-6)

    for seg in segments:
        if seg.kind in {"idle", "lookaround"}:
            dur = float(seg.duration_s or 0.0)
            steps = max(1, int(round(dur / float(dt)))) if dur > 0 else 1
            for _ in range(steps):
                positions.append(cur.copy())
            continue

        if seg.kind == "goto" and seg.target is not None:
            goal = geometry.snap_point(np.asarray(seg.target, dtype=np.float32))
            path = geometry.query_path(cur, goal)
            if not path:
                # If navmesh path is unavailable, fall back to straight line stepping.
                path = [cur.copy(), goal.copy()]

            poly = [np.asarray(p, dtype=np.float32) for p in path]
            poly = _dedupe_polyline(poly)
            if len(poly) < 2:
                cur = goal
                positions.append(cur.copy())
                continue

            # Walk along the polyline at constant speed.
            s = 0.0
            cum = _cumdist(poly)
            total = float(cum[-1])
            while s + step_dist < total:
                s += step_dist
                positions.append(_interp_polyline(poly, cum, s))
            cur = goal
            positions.append(cur.copy())
            continue

    return positions


@dataclass(frozen=True)
class RouteTimeline:
    """Per-step (physics step) timeline synthesized from move_routes."""

    positions: List[np.ndarray]
    stop_flags: List[bool]  # True when segment is an explicit "stop" (idle/lookaround)
    segment_kinds: List[str]  # per-step segment kind


def simulate_route_timeline(
    *,
    geometry: GeometryInterface,
    start_pos: np.ndarray,
    segments: Sequence[RouteSegment],
    dt: float,
    speed_units_per_s: float,
) -> RouteTimeline:
    """Simulate route into a per-step timeline aligned to the datagen frame axis.

    Notes:
      - Index 0 corresponds to the route start pose at the start frame.
      - For "idle"/"lookaround" segments, stop_flags are True for the whole segment.
      - For "goto" segments, stop_flags are False.
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if speed_units_per_s <= 0:
        raise ValueError("speed_units_per_s must be > 0")

    cur = np.asarray(start_pos, dtype=np.float32)
    cur = geometry.snap_point(cur)
    positions: List[np.ndarray] = [cur.copy()]
    stop_flags: List[bool] = [False]
    kinds: List[str] = ["start"]

    step_dist = float(speed_units_per_s) * float(dt)
    step_dist = max(step_dist, 1e-6)

    for seg in segments:
        if seg.kind in {"idle", "lookaround"}:
            dur = float(seg.duration_s or 0.0)
            steps = max(1, int(round(dur / float(dt)))) if dur > 0 else 1
            for _ in range(steps):
                positions.append(cur.copy())
                stop_flags.append(True)
                kinds.append(seg.kind)
            continue

        if seg.kind == "goto" and seg.target is not None:
            goal = geometry.snap_point(np.asarray(seg.target, dtype=np.float32))
            path = geometry.query_path(cur, goal)
            if not path:
                path = [cur.copy(), goal.copy()]
            poly = [np.asarray(p, dtype=np.float32) for p in path]
            poly = _dedupe_polyline(poly)
            if len(poly) < 2:
                cur = goal
                positions.append(cur.copy())
                stop_flags.append(False)
                kinds.append("goto")
                continue

            cum = _cumdist(poly)
            total = float(cum[-1])
            s = 0.0
            while s + step_dist < total:
                s += step_dist
                positions.append(_interp_polyline(poly, cum, s))
                stop_flags.append(False)
                kinds.append("goto")
            cur = goal
            positions.append(cur.copy())
            stop_flags.append(False)
            kinds.append("goto")
            continue

    if len(positions) != len(stop_flags) or len(positions) != len(kinds):
        raise RuntimeError("RouteTimeline internal length mismatch")

    return RouteTimeline(positions=positions, stop_flags=stop_flags, segment_kinds=kinds)


def compute_yaws_from_positions(positions: Sequence[np.ndarray]) -> List[float]:
    """Compute yaw (rad) for each step from consecutive positions."""
    if not positions:
        return []
    yaws: List[float] = []
    last = 0.0
    for i in range(len(positions)):
        if i < len(positions) - 1:
            d = np.asarray(positions[i + 1], dtype=np.float32) - np.asarray(positions[i], dtype=np.float32)
            if float(d[0] * d[0] + d[1] * d[1]) > 1e-8:
                last = float(np.arctan2(float(d[1]), float(d[0])))
        yaws.append(last)
    return yaws


def compute_follow_robot_positions(
    *,
    geometry: GeometryInterface,
    human_positions: Sequence[np.ndarray],
    human_yaws_rad: Sequence[float],
    follow_distance_units: float,
    z_keep: Optional[float] = None,
) -> List[np.ndarray]:
    """Compute a robot trajectory that follows behind the human."""
    if not human_positions:
        return []
    if len(human_positions) != len(human_yaws_rad):
        raise ValueError("human_positions and human_yaws_rad must have same length")
    d = float(max(follow_distance_units, 0.0))
    out: List[np.ndarray] = []
    for p, yaw in zip(human_positions, human_yaws_rad):
        p = np.asarray(p, dtype=np.float32)
        back = np.array([np.cos(float(yaw)), np.sin(float(yaw)), 0.0], dtype=np.float32)
        candidate = p - d * back
        snapped = geometry.snap_point(candidate)
        if z_keep is not None:
            snapped = np.asarray([snapped[0], snapped[1], float(z_keep)], dtype=np.float32)
        out.append(snapped)
    return out


def build_reference_waypoints_from_timed_poses(
    *,
    positions: Sequence[np.ndarray],
    yaws_rad: Sequence[float],
    dt: float,
    start_frame: int,
    stride_frames: int = 3,
) -> List[Dict[str, Any]]:
    """Build sim-step-aligned reference waypoints from a timed pose sequence."""
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if len(positions) != len(yaws_rad):
        raise ValueError("positions and yaws_rad must have same length")
    if not positions:
        return []
    stride = max(1, int(stride_frames))

    wps: List[Dict[str, Any]] = []
    last_p = None
    total_xy = 0.0
    for step in range(0, len(positions), stride):
        p = np.asarray(positions[step], dtype=np.float32)
        yaw = float(yaws_rad[step])
        frame = int(start_frame) + int(step)
        distance_xy = 0.0
        if last_p is not None:
            dxy = (p - last_p)[:2]
            distance_xy = float(np.linalg.norm(dxy))
            total_xy += distance_xy
        last_p = p
        wps.append(
            {
                "frame": frame,
                "time_s": float(frame) * float(dt),
                "xyz": [float(p[0]), float(p[1]), float(p[2])],
                "yaw_deg": float(np.degrees(yaw)),
                "distance_xy": float(distance_xy),
                "distance_total_xy": float(total_xy),
                "command": {"v": 0.0, "w": 0.0, "lateral": 0.0},
            }
        )

    # Always include the final step for stability.
    last_step = int(len(positions) - 1)
    if not wps or int(wps[-1]["frame"]) != int(start_frame) + last_step:
        p = np.asarray(positions[last_step], dtype=np.float32)
        yaw = float(yaws_rad[last_step])
        dxy = (p - last_p)[:2] if last_p is not None else np.zeros(2, dtype=np.float32)
        distance_xy = float(np.linalg.norm(dxy))
        total_xy += distance_xy
        frame = int(start_frame) + last_step
        wps.append(
            {
                "frame": frame,
                "time_s": float(frame) * float(dt),
                "xyz": [float(p[0]), float(p[1]), float(p[2])],
                "yaw_deg": float(np.degrees(yaw)),
                "distance_xy": float(distance_xy),
                "distance_total_xy": float(total_xy),
                "command": {"v": 0.0, "w": 0.0, "lateral": 0.0},
            }
        )
    return wps


def _dedupe_polyline(points: List[np.ndarray], eps: float = 1e-6) -> List[np.ndarray]:
    if not points:
        return []
    out = [points[0]]
    for p in points[1:]:
        if float(np.linalg.norm(np.asarray(p) - np.asarray(out[-1]))) > float(eps):
            out.append(p)
    return out


def _cumdist(points: Sequence[np.ndarray]) -> np.ndarray:
    cum = [0.0]
    total = 0.0
    for a, b in zip(points[:-1], points[1:]):
        total += float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
        cum.append(total)
    return np.asarray(cum, dtype=np.float32)


def _interp_polyline(points: Sequence[np.ndarray], cum: np.ndarray, s: float) -> np.ndarray:
    s = float(np.clip(float(s), 0.0, float(cum[-1])))
    idx = int(np.searchsorted(cum, s, side="right") - 1)
    idx = max(0, min(idx, len(points) - 2))
    s0 = float(cum[idx])
    s1 = float(cum[idx + 1])
    if s1 <= s0 + 1e-9:
        return np.asarray(points[idx + 1], dtype=np.float32).copy()
    t = (s - s0) / (s1 - s0)
    a = np.asarray(points[idx], dtype=np.float32)
    b = np.asarray(points[idx + 1], dtype=np.float32)
    return a + (b - a) * float(t)
