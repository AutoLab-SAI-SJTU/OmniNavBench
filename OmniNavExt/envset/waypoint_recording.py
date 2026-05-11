from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _angle_diff_rad(a: float, b: float) -> float:
    d = float(a) - float(b)
    return (d + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class PoseSample:
    frame: int
    time_s: float
    xyz: Tuple[float, float, float]
    yaw_rad: float
    sim_step: Optional[int] = None
    command: Optional[Dict[str, float]] = None


class WaypointRecorder:
    """Collect fixed-frequency pose samples and build waypoint payloads."""

    def __init__(self, *, forward_scale: float = 1.0, rot_scale: float = 1.0):
        self._forward_scale = float(forward_scale)
        self._rot_scale = float(rot_scale)
        self._samples: Dict[int, PoseSample] = {}

    def reset(self) -> None:
        self._samples.clear()

    def add_sample(
        self,
        *,
        frame: int,
        time_s: float,
        xyz: Sequence[float],
        yaw_rad: float,
        sim_step: Optional[int] = None,
        command: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if len(xyz) < 3:
            raise ValueError("xyz must contain at least 3 values")

        normalized_command = None
        if isinstance(command, Mapping):
            normalized_command = {
                "v": float(command.get("v", 0.0)),
                "w": float(command.get("w", 0.0)),
                "lateral": float(command.get("lateral", 0.0)),
            }

        self._samples[int(frame)] = PoseSample(
            frame=int(frame),
            time_s=float(time_s),
            xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            yaw_rad=float(yaw_rad),
            sim_step=(int(sim_step) if sim_step is not None else None),
            command=normalized_command,
        )

    def has_frame(self, frame: int) -> bool:
        return int(frame) in self._samples

    def last_frame(self) -> Optional[int]:
        if not self._samples:
            return None
        return max(self._samples.keys())

    def build(self) -> list[Dict[str, Any]]:
        out: list[Dict[str, Any]] = []
        prev = None
        total_xy = 0.0

        for frame in sorted(self._samples.keys()):
            sample = self._samples[frame]

            distance_xy = 0.0
            cmd_v = 0.0
            cmd_w = 0.0
            cmd_lateral = 0.0

            if prev is not None:
                dx = float(sample.xyz[0]) - float(prev.xyz[0])
                dy = float(sample.xyz[1]) - float(prev.xyz[1])
                distance_xy = math.sqrt(dx * dx + dy * dy)
                total_xy += distance_xy

                if sample.command is not None:
                    cmd_v = float(sample.command.get("v", 0.0))
                    cmd_w = float(sample.command.get("w", 0.0))
                    cmd_lateral = float(sample.command.get("lateral", 0.0))
                else:
                    dt_seg = float(sample.time_s) - float(prev.time_s)
                    if dt_seg <= 0.0:
                        dt_seg = 1e-6
                    vx = dx / dt_seg
                    vy = dy / dt_seg
                    forward_mps = vx * math.cos(sample.yaw_rad) + vy * math.sin(sample.yaw_rad)
                    w_rps = _angle_diff_rad(sample.yaw_rad, prev.yaw_rad) / dt_seg
                    if self._forward_scale != 0.0:
                        cmd_v = _clamp(forward_mps / self._forward_scale, -1.0, 1.0)
                    if self._rot_scale != 0.0:
                        cmd_w = _clamp(w_rps / self._rot_scale, -1.0, 1.0)
            elif sample.command is not None:
                cmd_v = float(sample.command.get("v", 0.0))
                cmd_w = float(sample.command.get("w", 0.0))
                cmd_lateral = float(sample.command.get("lateral", 0.0))

            waypoint = {
                "frame": int(sample.frame),
                "time_s": float(sample.time_s),
                "xyz": [float(sample.xyz[0]), float(sample.xyz[1]), float(sample.xyz[2])],
                "yaw_deg": float(math.degrees(sample.yaw_rad)),
                "distance_xy": float(distance_xy),
                "distance_total_xy": float(total_xy),
                "command": {
                    "v": float(cmd_v),
                    "w": float(cmd_w),
                    "lateral": float(cmd_lateral),
                },
            }
            if sample.sim_step is not None:
                waypoint["sim_step"] = int(sample.sim_step)
            out.append(waypoint)
            prev = sample

        return out
