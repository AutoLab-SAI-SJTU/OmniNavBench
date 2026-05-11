from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np


def build_reference_waypoints(
    path_points: List[np.ndarray],
    dt: float,
    nominal_speed: float,
    start_frame: int = 0,
) -> List[Dict[str, Any]]:
    """Convert a polyline path into sim-step-aligned reference waypoints."""
    if dt <= 0:
        raise ValueError("dt must be > 0")
    nominal_speed = float(nominal_speed)
    if nominal_speed <= 0:
        raise ValueError("nominal_speed must be > 0")

    pts = [np.asarray(p, dtype=np.float32) for p in path_points]
    if len(pts) < 2:
        return []

    wps: List[Dict[str, Any]] = []
    frame = int(start_frame)
    last_yaw_deg = 0.0
    total_xy = 0.0

    for idx, p in enumerate(pts):
        if idx < len(pts) - 1:
            d = pts[idx + 1] - p
            last_yaw_deg = math.degrees(math.atan2(float(d[1]), float(d[0])))

        distance_xy = 0.0
        if idx > 0:
            dist = float(np.linalg.norm(p - pts[idx - 1]))
            frame += max(1, int(math.ceil(dist / max(nominal_speed * float(dt), 1e-6))))
            distance_xy = float(np.linalg.norm((p - pts[idx - 1])[:2]))
            total_xy += distance_xy

        wps.append(
            {
                "frame": int(frame),
                "time_s": float(frame) * float(dt),
                "xyz": [float(p[0]), float(p[1]), float(p[2])],
                "yaw_deg": float(last_yaw_deg),
                "distance_xy": float(distance_xy),
                "distance_total_xy": float(total_xy),
                "command": {"v": 0.0, "w": 0.0, "lateral": 0.0},
            }
        )
    return wps
