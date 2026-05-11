import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from OmniNav.core.util import log


class EpisodeLogger:
    """Records and saves episode data to a JSON file."""

    def __init__(
        self,
        log_path: str,
        instruction: str = "",
        distance_threshold: float = 0.0,
        robot_path: Optional[str] = None,
        objects: Optional[Dict[str, Any]] = None,
        room_zone: Optional[Dict[str, Any]] = None,
        answer: Optional[str] = None,
        meters_per_env_unit: Optional[float] = None,
        meters_per_stage_unit: Optional[float] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ):
        self._log_path = Path(log_path) if log_path else None
        self._instruction = instruction.strip()
        self._distance_threshold = max(float(distance_threshold), 0.0)
        self._robot_path = robot_path
        self._objects = copy.deepcopy(objects) if objects else None
        self._room_zone = copy.deepcopy(room_zone) if room_zone else None
        self._answer = answer
        self._extra_fields = copy.deepcopy(extra_fields) if extra_fields else {}

        self._meters_per_env_unit = self._ensure_positive(
            meters_per_env_unit, default=1.0, field="meters_per_env_unit"
        )
        self._meters_per_stage_unit = self._ensure_positive(
            meters_per_stage_unit, default=1.0, field="meters_per_stage_unit"
        )
        self._stage_to_env_scale = self._meters_per_stage_unit / self._meters_per_env_unit

        # Robot trajectory state (single agent, kept for backward compatibility)
        self._path_entries: List[dict] = []
        self._last_position: Optional[Tuple[float, float, float]] = None
        self._total_distance_xy = 0.0
        self._first_pose: Optional[dict] = None
        self._last_pose: Optional[dict] = None

        # Virtual human trajectories (multi-agent, keyed by agent_name)
        # Each agent maintains its own path entries and distance accumulator.
        self._vh_path_entries: Dict[str, List[dict]] = {}
        self._vh_last_position: Dict[str, Tuple[float, float, float]] = {}
        self._vh_total_distance_xy: Dict[str, float] = {}
        self._vh_first_pose: Dict[str, dict] = {}
        self._vh_last_pose: Dict[str, dict] = {}

        if self._log_path:
            log.info(f"[EpisodeLogger] Enabled: path={self._log_path}, threshold={self._distance_threshold:.3f}")

    def record_step(
        self,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],
        command: Union[Dict[str, float], Tuple[float, float]],
        frame_idx: int,
        time_s: Optional[float] = None,
        force: bool = False,
    ):
        """Record one step.

        ``command`` may be either a dict ``{"v": forward, "w": rotation, "lateral": lateral}``
        or a 2-tuple ``(v, w)`` for backward compatibility.
        """
        if not self._log_path:
            return

        x, y, z = self._stage_to_env_coordinates(position)
        yaw_deg = self._quat_to_yaw_deg(orientation)

        distance_xy = 0.0
        if self._last_position:
            dx = x - self._last_position[0]
            dy = y - self._last_position[1]
            distance_xy = (dx * dx + dy * dy) ** 0.5

            if not force and distance_xy < self._distance_threshold:
                return

            self._total_distance_xy += distance_xy
        else:
            self._first_pose = {
                "xyz": [x, y, z],
                "yaw_deg": yaw_deg,
            }

        self._last_position = (x, y, z)
        self._last_pose = {
            "xyz": [x, y, z],
            "yaw_deg": yaw_deg,
        }

        if isinstance(command, dict):
            cmd_dict = command
        elif isinstance(command, (tuple, list)) and len(command) >= 2:
            # Backward-compatible tuple form.
            cmd_dict = {"v": float(command[0]), "w": float(command[1])}
        else:
            cmd_dict = {"v": 0.0, "w": 0.0}

        entry = {
            "frame": int(frame_idx),
            "xyz": [x, y, z],
            "yaw_deg": float(yaw_deg),
            "distance_xy": float(distance_xy),
            "distance_total_xy": float(self._total_distance_xy),
            "command": {
                "v": float(cmd_dict.get("v", 0.0)),
                "w": float(cmd_dict.get("w", 0.0)),
            },
        }

        if "lateral" in cmd_dict:
            entry["command"]["lateral"] = float(cmd_dict["lateral"])

        if time_s is not None:
            entry["time_s"] = float(time_s)

        self._path_entries.append(entry)

    def record_virtual_human_step(
        self,
        agent_name: str,
        position: Tuple[float, float, float],
        yaw_deg: Optional[float],
        frame_idx: int,
        time_s: Optional[float] = None,
        force: bool = False,
    ):
        """Record one step for a virtual human agent.

        Position is expected in stage units and will be converted into env units,
        sharing the same distance_threshold configuration as robots.
        """
        if not self._log_path:
            return

        if not agent_name:
            raise ValueError("[EpisodeLogger] agent_name must be a non-empty string for virtual human logging")

        # Convert to env coordinates
        x, y, z = self._stage_to_env_coordinates(position)

        # Initialize per-agent accumulators if needed
        if agent_name not in self._vh_path_entries:
            self._vh_path_entries[agent_name] = []
            self._vh_last_position.pop(agent_name, None)
            self._vh_total_distance_xy[agent_name] = 0.0
            self._vh_first_pose.pop(agent_name, None)
            self._vh_last_pose.pop(agent_name, None)

        last_pos = self._vh_last_position.get(agent_name)
        distance_xy = 0.0
        if last_pos is not None:
            dx = x - last_pos[0]
            dy = y - last_pos[1]
            distance_xy = (dx * dx + dy * dy) ** 0.5
            # Distance threshold filtering (same threshold as robot)
            if not force and distance_xy < self._distance_threshold:
                return
            self._vh_total_distance_xy[agent_name] += distance_xy
        else:
            # First pose for this agent
            pose = {"xyz": [x, y, z]}
            if yaw_deg is not None:
                pose["yaw_deg"] = float(yaw_deg)
            self._vh_first_pose[agent_name] = pose

        # Update last position & pose
        self._vh_last_position[agent_name] = (x, y, z)
        pose = {"xyz": [x, y, z]}
        if yaw_deg is not None:
            pose["yaw_deg"] = float(yaw_deg)
        self._vh_last_pose[agent_name] = pose

        # Build entry
        entry: Dict[str, Any] = {
            "frame": int(frame_idx),
            "xyz": [x, y, z],
            "distance_xy": float(distance_xy),
            "distance_total_xy": float(self._vh_total_distance_xy[agent_name]),
        }
        if yaw_deg is not None:
            entry["yaw_deg"] = float(yaw_deg)
        if time_s is not None:
            entry["time_s"] = float(time_s)

        self._vh_path_entries[agent_name].append(entry)

    def save_episode(self):
        """Save episode data to a JSON file."""
        log.info(f"[EpisodeLogger] save_episode() called: log_path={self._log_path}, entries={len(self._path_entries)}")

        if not self._log_path or (not self._path_entries and not self._vh_path_entries):
            log.warn(
                f"[EpisodeLogger] save_episode() skipped: "
                f"log_path={self._log_path}, robot_entries={len(self._path_entries)}, "
                f"vh_agents={len(self._vh_path_entries)}"
            )
            return

        self._path_entries.sort(key=lambda entry: entry.get('frame', 0))
        for entries in self._vh_path_entries.values():
            entries.sort(key=lambda entry: entry.get('frame', 0))

        metadata = {
            "robot_path": self._robot_path or "",
            "distance_threshold_xy": self._distance_threshold,
            "distance_total_xy": self._total_distance_xy,
            "sample_count": len(self._path_entries),
            "meters_per_env_unit": self._meters_per_env_unit,
            "meters_per_stage_unit": self._meters_per_stage_unit,
        }

        if self._first_pose:
            metadata["robot_initial_pose"] = copy.deepcopy(self._first_pose)
        if self._last_pose:
            metadata["robot_final_pose"] = copy.deepcopy(self._last_pose)
        # Write objects, room_zone, and answer even if they are empty (empty dict or empty string)
        if self._objects is not None:
            metadata["objects"] = copy.deepcopy(self._objects)
        if self._room_zone is not None:
            metadata["room_zone"] = copy.deepcopy(self._room_zone)
        if self._answer is not None:
            metadata["answer"] = self._answer
        
        # Automatically add all extra fields to metadata
        if self._extra_fields:
            for key, value in self._extra_fields.items():
                # Skip fields that are already explicitly handled above
                if key not in {"objects", "room_zone", "answer"}:
                    # Write extra fields even if they are empty (empty dict, empty string, etc.)
                    metadata[key] = copy.deepcopy(value)

        vh_paths: Dict[str, Any] = {}
        for agent_name, entries in self._vh_path_entries.items():
            if not entries:
                continue
            total_dist = self._vh_total_distance_xy.get(agent_name, 0.0)
            vh_meta: Dict[str, Any] = {
                "agent_name": agent_name,
                "distance_threshold_xy": self._distance_threshold,
                "distance_total_xy": float(total_dist),
                "sample_count": len(entries),
            }
            first_pose = self._vh_first_pose.get(agent_name)
            last_pose = self._vh_last_pose.get(agent_name)
            if first_pose is not None:
                vh_meta["initial_pose"] = copy.deepcopy(first_pose)
            if last_pose is not None:
                vh_meta["final_pose"] = copy.deepcopy(last_pose)
            vh_paths[agent_name] = {
                "path": entries,
                "metadata": vh_meta,
            }

        payload = {
            "instruction": self._instruction,
            "gt_path": self._path_entries,
            "metadata": metadata,
        }
        if vh_paths:
            payload["vh_paths"] = vh_paths

        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._log_path.with_suffix(self._log_path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2, ensure_ascii=False)
                fp.flush()
                os.fsync(fp.fileno())
            tmp_path.replace(self._log_path)
            log.info(f"[EpisodeLogger] Saved {len(self._path_entries)} samples to {self._log_path}")
        except Exception as exc:
            log.error(f"[EpisodeLogger] Failed to save episode to {self._log_path}: {exc}")
            raise

    def reset(self):
        """Reset per-episode state for the start of a new episode."""
        self._path_entries.clear()
        self._last_position = None
        self._total_distance_xy = 0.0
        self._first_pose = None
        self._last_pose = None

    @staticmethod
    def _quat_to_yaw_deg(quat: Tuple[float, float, float, float]) -> float:
        """Convert quaternion to yaw angle in degrees."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        return math.degrees(yaw_rad)

    @staticmethod
    def _extract_velocity_command(action_dict: Dict[str, Any], robot) -> Dict[str, float]:
        """Extract a velocity command from action_dict or controller observations.

        Returns ``{"v": forward, "w": rotation, "lateral": lateral}``; ``lateral`` is
        optional for robots that do not expose strafe motion. Controller observations
        are preferred (they reflect autonomous-mode output); ``action_dict`` is a
        fallback for keyboard / external input.
        """
        result = {"v": 0.0, "w": 0.0}

        # Prefer controller observations (covers autonomous-mode output).
        try:
            obs = robot.get_obs()
            controllers_obs = obs.get("controllers", {})
            for controller_name, controller_obs in controllers_obs.items():
                if isinstance(controller_obs, dict):
                    if "forward_speed" in controller_obs and "rotation_speed" in controller_obs:
                        result["v"] = float(controller_obs["forward_speed"])
                        result["w"] = float(controller_obs["rotation_speed"])
                        if "lateral_speed" in controller_obs:
                            result["lateral"] = float(controller_obs["lateral_speed"])
                        return result
                    # Legacy field names.
                    if "v" in controller_obs and "w" in controller_obs:
                        result["v"] = float(controller_obs["v"])
                        result["w"] = float(controller_obs["w"])
                        if "lateral" in controller_obs:
                            result["lateral"] = float(controller_obs["lateral"])
                        return result
        except Exception:
            pass

        # Fallback: read from action_dict.
        if action_dict:
            for controller_name, action_value in action_dict.items():
                if isinstance(action_value, (list, tuple)):
                    try:
                        # Accept 2-component (v, w) or 3-component (v, lateral, w).
                        if len(action_value) >= 2:
                            result["v"] = float(action_value[0])
                            result["w"] = float(action_value[2]) if len(action_value) >= 3 else float(action_value[1])
                            if len(action_value) >= 3:
                                result["lateral"] = float(action_value[1])
                        return result
                    except (ValueError, TypeError, IndexError):
                        continue
                elif isinstance(action_value, dict):
                    if "v" in action_value and "w" in action_value:
                        result["v"] = float(action_value["v"])
                        result["w"] = float(action_value["w"])
                        if "lateral" in action_value:
                            result["lateral"] = float(action_value["lateral"])
                        return result

        return result

    @staticmethod
    def _ensure_positive(value: Optional[float], default: float, field: str) -> float:
        if value is None:
            value = default
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"[EpisodeLogger] {field} must be numeric, got {value}") from exc
        if value <= 0.0:
            raise ValueError(f"[EpisodeLogger] {field} must be > 0, got {value}")
        return value

    def _stage_to_env_coordinates(self, position: Tuple[float, float, float]) -> Tuple[float, float, float]:
        scale = self._stage_to_env_scale
        x = float(position[0]) * scale
        y = float(position[1]) * scale
        z = float(position[2]) * scale
        return x, y, z

