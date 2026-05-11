"""Offline test evaluator for submitted navigation trajectories."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .gt_anchor import GTAnchor, build_episode_gt_anchors
from ..metrics.navigation import (
    compute_eqa,
    compute_follow_human_success_ratio,
    compute_follow_human_task_success,
    compute_spl_offline,
)


XYZ = Tuple[float, float, float]
_RESULT_REL_PATH_KEY = "_result_relpath"
_RESULT_TASK_DIRS = {"car", "dog", "human"}


@dataclass(frozen=True)
class _OfflineTrajectoryPoint:
    position: XYZ
    orientation: Optional[Tuple[float, float, float, float]] = None
    yaw_rad: Optional[float] = None
    frame: Optional[int] = None
    step: Optional[int] = None
    time_s: Optional[float] = None


def evaluate_episode_result(private_scenario: Dict[str, Any], result_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one submitted episode result against a private scenario.

    Public output is intentionally redacted: it does not include target names,
    target coordinates, GT anchor coordinates, d_gt, or effective radius.
    """

    anchor_result = build_episode_gt_anchors(private_scenario)
    anchors_by_subtask_index = {
        int(anchor.subtask_index): anchor
        for anchor in anchor_result.anchors
    }
    subtasks = _subtasks(private_scenario)
    units = _units_in_meters(private_scenario)
    coordinate_scale = _submission_coordinate_scale(private_scenario, result_payload, units)
    trajectory_points = _trajectory_points(result_payload, coordinate_scale)
    trajectory = [point.position for point in trajectory_points]
    human_paths = _submitted_human_paths(result_payload, coordinate_scale)
    social_violation_ratio = _social_violation_ratio(
        private_scenario=private_scenario,
        trajectory_points=trajectory_points,
        human_paths=human_paths,
        units=units,
    )
    _objs = (_navigation(private_scenario).get("objects") or {})
    _has_static = any(str(k).startswith("Human") for k in (_objs if isinstance(_objs, dict) else {}))
    _has_dynamic = bool(human_paths) or bool(_vh_gt_waypoint_paths(private_scenario, units))
    is_social = _has_static or _has_dynamic
    stop_step = int(result_payload.get("stop_step", -1))
    eqa_answer = result_payload.get("eqa_answer")

    # --- episode-level metrics: goal_position, qa from private data ---
    goal_position_m = _coerce_xyz_m(
        _navigation(private_scenario).get("goal_position"), units,
    )
    success_radius_m = 2.0
    path_length = 0.0
    for i in range(len(trajectory) - 1):
        path_length += _distance_xy(trajectory[i], trajectory[i + 1])

    if not trajectory:
        subtask_outputs = [
            _subtask_output(index=index, subtask_type=_subtask_type(subtask), success=False, progress=0.0)
            for index, subtask in enumerate(subtasks)
            if isinstance(subtask, dict)
        ]
        return {
            "sr": False,
            "csr": 0.0,
            "softsr": 0.0,
            "spl": 0.0,
            "ne": float("inf") if goal_position_m else 0.0,
            "osr": 0.0,
            "num_subtasks": len(subtask_outputs),
            "subtasks": subtask_outputs,
            "social_violation_ratio": social_violation_ratio,
            "failure_reason": "empty_trajectory",
        }

    subtask_outputs = []
    progresses = []
    successes = []
    timestamps = []
    order_start_index = 0
    follow_human_success_values = []
    follow_human_ratio_values = []
    for index, subtask in enumerate(subtasks):
        if not isinstance(subtask, dict):
            continue
        subtask_type = _subtask_type(subtask)
        skipped = False
        if subtask_type == "FOLLOW_HUMAN" and not _selected_human_path(human_paths, _target_id(subtask)):
            success, progress, timestamp = True, 1.0, -1
            skipped = True
        else:
            success, progress, timestamp = _evaluate_subtask(
                private_scenario=private_scenario,
                subtask_index=index,
                subtask=subtask,
                trajectory=trajectory,
                trajectory_points=trajectory_points,
                units=units,
                anchors_by_subtask_index=anchors_by_subtask_index,
                human_paths=human_paths,
                order_start_index=order_start_index,
            )
        subtask_outputs.append(
            _subtask_output(
                index=index,
                subtask_type=subtask_type,
                success=success,
                progress=progress,
                skipped=skipped,
            )
        )
        progresses.append(progress)
        successes.append(success)
        timestamps.append((success, timestamp, subtask_type))
        if subtask_type not in _ORDER_EXEMPT_SUBTASK_TYPES and success and timestamp >= 0:
            order_start_index = timestamp + 1
        if subtask_type == "FOLLOW_HUMAN" and not skipped:
            follow_human_success_values.append(1.0 if success else 0.0)
            follow_human_ratio_values.append(float(progress))

    overall_success = bool(successes and all(successes))
    order_ok = _subtasks_completed_in_order(timestamps)
    softsr = float(sum(progresses) / len(progresses)) if progresses else 0.0
    follow_human_success = (
        float(sum(follow_human_success_values) / len(follow_human_success_values))
        if follow_human_success_values
        else None
    )
    follow_human_success_ratio = (
        float(sum(follow_human_ratio_values) / len(follow_human_ratio_values))
        if follow_human_ratio_values
        else None
    )
    failure_reason = None
    if not overall_success:
        failure_reason = "target_not_reached"
    elif not order_ok:
        failure_reason = "subtask_order_invalid"

    # --- episode-level metrics ---
    sr = bool(
        stop_step >= 0
        and goal_position_m is not None
        and _distance_xy(trajectory[-1], goal_position_m) <= success_radius_m
    )
    csr = 1.0 if overall_success and order_ok else 0.0

    final_pos = trajectory[-1]
    ne = _distance_xy(final_pos, goal_position_m) if goal_position_m else float("inf")

    osr = 0.0
    osr_radius_m = 2.0
    if goal_position_m and trajectory:
        last_subtask_type: Optional[str] = None
        for subtask in reversed(subtasks):
            if isinstance(subtask, dict):
                last_subtask_type = _subtask_type(subtask)
                break
        start_in_radius = _distance_xy(trajectory[0], goal_position_m) <= osr_radius_m
        require_leave_first = bool(last_subtask_type == "RETURN_TO" or start_in_radius)
        if require_leave_first:
            leave_index = _first_distance_greater_than_threshold_index(
                trajectory, goal_position_m, 2.0,
            )
            search_from = leave_index + 1 if leave_index >= 0 else len(trajectory)
            scan = list(range(search_from, len(trajectory)))
        else:
            scan = list(range(len(trajectory)))
        for index in scan:
            pt = trajectory[index]
            if _distance_xy(pt, goal_position_m) <= osr_radius_m:
                osr = 1.0
                break
            if index > 0 and (not require_leave_first or index - 1 >= search_from):
                if _segment_min_distance_xy(trajectory[index - 1], pt, goal_position_m) <= osr_radius_m:
                    osr = 1.0
                    break

    spl = 0.0
    if goal_position_m and trajectory:
        start = trajectory[0]
        shortest_path = _estimate_shortest_path([start, goal_position_m])
        spl = float(compute_spl_offline(success=sr, path_length=path_length, shortest_path=shortest_path))

    eqa_accuracy = None
    qa = _private_qa(private_scenario)
    if qa and eqa_answer is not None:
        ground_truth = str(qa.get("answer", ""))
        eqa_accuracy = bool(compute_eqa(ground_truth, str(eqa_answer)))

    output: Dict[str, Any] = {
        "sr": sr,
        "csr": csr,
        "softsr": softsr,
        "spl": spl,
        "ne": ne,
        "osr": osr,
        "num_subtasks": len(subtask_outputs),
        "subtasks": subtask_outputs,
        "social_violation_ratio": social_violation_ratio,
        "is_social": is_social,
        "failure_reason": failure_reason,
    }
    if follow_human_success is not None:
        output["follow_human_success"] = follow_human_success
        output["follow_human_success_ratio"] = follow_human_success_ratio
    if eqa_accuracy is not None:
        output["eqa_accuracy"] = eqa_accuracy
    return output


def evaluate_submission_results(
    *,
    private_scenarios: Sequence[Dict[str, Any]],
    result_payloads: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate a submitted set of per-episode result payloads."""

    private_by_key: Dict[str, Dict[str, Any]] = {}
    private_by_id: Dict[str, List[Dict[str, Any]]] = {}
    for scenario in private_scenarios:
        if isinstance(scenario, dict):
            key = f"{scenario.get('id', '')}:{scenario.get('_relpath', '')}"
            private_by_key[key] = scenario
            private_by_id.setdefault(str(scenario.get("id", "")), []).append(scenario)

    episodes: List[Dict[str, Any]] = []
    for index, result_payload in enumerate(result_payloads):
        scenario_id = str(result_payload.get("scenario_id", "")) if isinstance(result_payload, dict) else ""
        source_envset = str(result_payload.get("source_envset", "")) if isinstance(result_payload, dict) else ""
        relpath = _source_envset_private_relpath(source_envset)
        compound_key = f"{scenario_id}:{relpath}"
        private_scenario = private_by_key.get(compound_key)
        if private_scenario is None:
            result_suffix = _result_task_suffix(result_payload)
            if result_suffix:
                for candidate in private_by_id.get(scenario_id, []):
                    if _path_task_suffix(str(candidate.get("_relpath", ""))) == result_suffix:
                        private_scenario = candidate
                        break
        if private_scenario is None:
            # Fallback: try scenario_id-only match
            compound_key = f"{scenario_id}:"
            for k, v in private_by_key.items():
                if k.startswith(compound_key):
                    private_scenario = v
                    break
        if private_scenario is None:
            episodes.append(
                {
                    "episode_id": scenario_id or f"episode_{index}",
                    "sr": False,
                    "csr": 0.0,
                    "softsr": 0.0,
                    "spl": 0.0,
                    "ne": 0.0,
                    "osr": 0.0,
                    "social_violation_ratio": 0.0,
                    "failure_reason": "private_scenario_not_found",
                }
            )
            continue

        episode_result = evaluate_episode_result(private_scenario, result_payload)
        episodes.append(
            {
                "episode_id": scenario_id,
                "source_envset": source_envset,
                "sr": bool(episode_result["sr"]),
                "csr": float(episode_result["csr"]),
                "softsr": float(episode_result["softsr"]),
                "spl": float(episode_result["spl"]),
                "ne": float(episode_result["ne"]),
                "osr": float(episode_result["osr"]),
                "social_violation_ratio": float(episode_result.get("social_violation_ratio", 0.0)),
                "is_social": bool(episode_result.get("is_social", False)),
                "failure_reason": episode_result.get("failure_reason"),
            }
        )
        if "follow_human_success" in episode_result:
            episodes[-1]["follow_human_success"] = float(episode_result["follow_human_success"])
            episodes[-1]["follow_human_success_ratio"] = float(episode_result["follow_human_success_ratio"])
        if "eqa_accuracy" in episode_result:
            episodes[-1]["eqa_accuracy"] = bool(episode_result["eqa_accuracy"])

    count = len(episodes)
    sr_values = [float(episode["sr"]) for episode in episodes]
    csr_values = [float(episode["csr"]) for episode in episodes]
    softsr_values = [float(episode["softsr"]) for episode in episodes]
    spl_values = [float(episode["spl"]) for episode in episodes]
    ne_values = [float(episode["ne"]) for episode in episodes if math.isfinite(episode["ne"])]
    osr_values = [float(episode["osr"]) for episode in episodes]
    social_values = [float(episode.get("social_violation_ratio", 0.0)) for episode in episodes if episode.get("is_social")]
    follow_human_success_values = [
        float(episode["follow_human_success"])
        for episode in episodes
        if isinstance(episode.get("follow_human_success"), (int, float))
    ]
    follow_human_ratio_values = [
        float(episode["follow_human_success_ratio"])
        for episode in episodes
        if isinstance(episode.get("follow_human_success_ratio"), (int, float))
    ]
    eqa_values = [
        float(episode["eqa_accuracy"])
        for episode in episodes
        if isinstance(episode.get("eqa_accuracy"), (int, float, bool))
    ]
    return {
        "summary": {
            "num_episodes": count,
            "sr": float(sum(sr_values) / count) if count else 0.0,
            "csr": float(sum(csr_values) / count) if count else 0.0,
            "softsr": float(sum(softsr_values) / count) if count else 0.0,
            "spl": float(sum(spl_values) / count) if count else 0.0,
            "ne": float(sum(ne_values) / len(ne_values)) if ne_values else 0.0,
            "osr": float(sum(osr_values) / count) if count else 0.0,
            "follow_human_success": (
                float(sum(follow_human_success_values) / len(follow_human_success_values))
                if follow_human_success_values
                else 0.0
            ),
            "follow_human_success_ratio": (
                float(sum(follow_human_ratio_values) / len(follow_human_ratio_values))
                if follow_human_ratio_values
                else 0.0
            ),
            "eqa": float(sum(eqa_values) / len(eqa_values)) if eqa_values else 0.0,
            "sii": float(sum(social_values) / len(social_values)) if social_values else 0.0,
        },
        "episodes": episodes,
    }


def evaluate_submission_paths(
    *,
    private_path: str | Path,
    results_path: str | Path,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Evaluate submitted result JSON files against private scenario JSON files."""

    private_scenarios = load_private_scenarios(private_path)
    result_payloads = load_result_payloads(results_path)
    output = evaluate_submission_results(
        private_scenarios=private_scenarios,
        result_payloads=result_payloads,
    )
    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    return output


def load_private_scenarios(path: str | Path) -> List[Dict[str, Any]]:
    """Load private scenarios from a scenario file, envset file, or directory."""

    root = Path(path)
    scenarios: List[Dict[str, Any]] = []
    for json_path in _json_files(path):
        payload = _read_json(json_path)
        relpath = str(json_path.relative_to(root)) if root.is_dir() else json_path.name
        for s in _extract_private_scenarios(payload):
            if isinstance(s, dict):
                s["_relpath"] = relpath
            scenarios.append(s)
    return scenarios


def load_result_payloads(path: str | Path) -> List[Dict[str, Any]]:
    """Load submitted result payloads from a result file or directory."""

    root = Path(path)
    payloads: List[Dict[str, Any]] = []
    for json_path in _json_files(path):
        text = json_path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        payload = json.loads(text)
        relpath = str(json_path.relative_to(root)) if root.is_dir() else json_path.name
        for item in _extract_result_payloads(payload):
            item_with_source = dict(item)
            item_with_source.setdefault(_RESULT_REL_PATH_KEY, relpath)
            payloads.append(item_with_source)
    return payloads


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Offline private judge for OmniNavBench result JSON files.")
    parser.add_argument("--private", "--private-path", dest="private_path", required=True)
    parser.add_argument("--results", "--results-path", dest="results_path", required=True)
    parser.add_argument("--output", dest="output_path", required=True)
    args = parser.parse_args(argv)

    output = evaluate_submission_paths(
        private_path=args.private_path,
        results_path=args.results_path,
        output_path=args.output_path,
    )
    summary = output.get("summary", {})
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _trajectory_positions(result_payload: Dict[str, Any], coordinate_scale: float = 1.0) -> List[XYZ]:
    return [point.position for point in _trajectory_points(result_payload, coordinate_scale)]


def _trajectory_points(
    result_payload: Dict[str, Any],
    coordinate_scale: float = 1.0,
) -> List[_OfflineTrajectoryPoint]:
    raw_trajectory = result_payload.get("trajectory")
    if not isinstance(raw_trajectory, list):
        return []
    points: List[_OfflineTrajectoryPoint] = []
    scale = _positive_float(coordinate_scale, 1.0)
    for item in raw_trajectory:
        if not isinstance(item, dict):
            continue
        position = item.get("position")
        if not isinstance(position, (list, tuple)) or len(position) < 3:
            continue
        try:
            xyz = (
                float(position[0]) * scale,
                float(position[1]) * scale,
                float(position[2]) * scale,
            )
        except (TypeError, ValueError):
            continue
        points.append(
            _OfflineTrajectoryPoint(
                position=xyz,
                orientation=_coerce_quat_wxyz(item.get("orientation", item.get("orientation_wxyz"))),
                yaw_rad=_coerce_yaw_rad(item),
                frame=_coerce_optional_int(item.get("frame")),
                step=_coerce_optional_int(item.get("step")),
                time_s=_coerce_optional_float(item.get("time_s")),
            )
        )
    return points


def _coerce_quat_wxyz(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    except (TypeError, ValueError):
        return None


def _submission_coordinate_scale(
    private_scenario: Mapping[str, Any],
    result_payload: Mapping[str, Any],
    units: float,
) -> float:
    if units == 1.0:
        return 1.0

    coordinate_space = _submission_coordinate_space(result_payload)
    if coordinate_space in {"meter", "meters", "metre", "metres", "m", "world", "world_meters"}:
        return 1.0
    if coordinate_space in {"scene", "scene_unit", "scene_units", "raw", "env_unit", "env_units"}:
        return units

    submitted = _raw_trajectory_positions(result_payload)
    raw_gt = _expert_path_raw(private_scenario)
    if not submitted or not raw_gt:
        return 1.0

    scaled_gt = [_scale_xyz(point, units) for point in raw_gt]
    raw_error = _mean_aligned_xy_error(submitted, raw_gt)
    scaled_error = _mean_aligned_xy_error(submitted, scaled_gt)
    if raw_error < scaled_error and raw_error <= max(1e-6, scaled_error * 0.25):
        return units
    return 1.0


def _submission_coordinate_space(result_payload: Mapping[str, Any]) -> Optional[str]:
    candidates: List[Any] = [
        result_payload.get("coordinate_space"),
        result_payload.get("coordinate_units"),
        result_payload.get("coordinates"),
    ]
    metadata = result_payload.get("metadata")
    if isinstance(metadata, Mapping):
        candidates.extend(
            [
                metadata.get("coordinate_space"),
                metadata.get("coordinate_units"),
                metadata.get("coordinates"),
            ]
        )
    for value in candidates:
        if value is not None:
            return str(value).strip().lower().replace(" ", "_")
    return None


def _raw_trajectory_positions(result_payload: Mapping[str, Any]) -> List[XYZ]:
    raw_trajectory = result_payload.get("trajectory")
    if not isinstance(raw_trajectory, list):
        return []
    positions: List[XYZ] = []
    for item in raw_trajectory:
        if not isinstance(item, Mapping):
            continue
        position = item.get("position")
        point = _coerce_xyz_m(position, 1.0)
        if point is not None:
            positions.append(point)
    return positions


def _expert_path_raw(private_scenario: Mapping[str, Any]) -> List[XYZ]:
    robots = private_scenario.get("robots")
    entries = robots.get("entries") if isinstance(robots, Mapping) else None
    if not isinstance(entries, list) or not entries or not isinstance(entries[0], Mapping):
        return []
    waypoints = entries[0].get("rb_gt_waypoints")
    if not isinstance(waypoints, list):
        return []
    path: List[XYZ] = []
    for waypoint in waypoints:
        if not isinstance(waypoint, Mapping):
            continue
        point = _coerce_xyz_m(waypoint.get("xyz"), 1.0)
        if point is not None:
            path.append(point)
    return path


def _scale_xyz(point: XYZ, scale: float) -> XYZ:
    return (float(point[0]) * scale, float(point[1]) * scale, float(point[2]) * scale)


def _mean_aligned_xy_error(path: Sequence[XYZ], reference: Sequence[XYZ]) -> float:
    if not path or not reference:
        return float("inf")
    count = min(len(path), len(reference), 16)
    if count <= 0:
        return float("inf")
    if count == 1:
        return _distance_xy(path[0], reference[0])

    total = 0.0
    for sample_idx in range(count):
        path_idx = round(sample_idx * (len(path) - 1) / (count - 1))
        ref_idx = round(sample_idx * (len(reference) - 1) / (count - 1))
        total += _distance_xy(path[path_idx], reference[ref_idx])
    return total / count


def _coerce_yaw_rad(item: Mapping[str, Any]) -> Optional[float]:
    for key in ("yaw_rad", "yaw"):
        value = item.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    value = item.get("yaw_deg")
    if value is None:
        return None
    try:
        return math.radians(float(value))
    except (TypeError, ValueError):
        return None


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_files(path: str | Path) -> List[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"Path does not exist: {root}")
    return sorted(item for item in root.rglob("*.json") if item.is_file())


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_envset_private_relpath(source_envset: str) -> str:
    return source_envset.split("/test/", 1)[1] if "/test/" in source_envset else ""


def _result_task_suffix(result_payload: Mapping[str, Any]) -> str:
    for value in (
        result_payload.get(_RESULT_REL_PATH_KEY),
        result_payload.get("source_envset"),
    ):
        suffix = _path_task_suffix(str(value or ""))
        if suffix:
            return suffix
    return ""


def _path_task_suffix(path_value: str) -> str:
    parts = [part for part in path_value.replace("\\", "/").split("/") if part]
    for index, part in enumerate(parts):
        if part in _RESULT_TASK_DIRS:
            return "/".join(parts[index:])
    return ""


def _extract_private_scenarios(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        scenarios = payload.get("scenarios")
        if isinstance(scenarios, list):
            return [scenario for scenario in scenarios if isinstance(scenario, dict)]
        if "task" in payload and "scene" in payload:
            return [payload]
    if isinstance(payload, list):
        return [scenario for scenario in payload if isinstance(scenario, dict)]
    return []


def _extract_result_payloads(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("results", "episodes"):
            entries = payload.get(key)
            if isinstance(entries, list):
                return [entry for entry in entries if isinstance(entry, dict)]
        if "trajectory" in payload or "scenario_id" in payload:
            return [payload]
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    return []


def _subtasks(private_scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    task = private_scenario.get("task")
    if not isinstance(task, dict):
        return []
    subtasks = task.get("subtasks")
    if not isinstance(subtasks, list):
        return []
    return [item for item in subtasks if isinstance(item, dict)]


def _subtask_type(subtask: Mapping[str, Any]) -> str:
    return str(subtask.get("type", "")).upper()


def _subtask_output(
    *,
    index: int,
    subtask_type: str,
    success: bool,
    progress: float,
    skipped: bool = False,
) -> Dict[str, Any]:
    output = {
        "subtask_index": int(index),
        "type": str(subtask_type),
        "success": bool(success),
        "progress": float(progress),
    }
    if skipped:
        output["skipped"] = True
    return output


def _evaluate_subtask(
    *,
    private_scenario: Dict[str, Any],
    subtask_index: int,
    subtask: Mapping[str, Any],
    trajectory: Sequence[XYZ],
    trajectory_points: Sequence[_OfflineTrajectoryPoint],
    units: float,
    anchors_by_subtask_index: Mapping[int, GTAnchor],
    human_paths: Mapping[str, List[Dict[str, XYZ]]],
    order_start_index: int = 0,
) -> Tuple[bool, float, int]:
    subtask_type = _subtask_type(subtask)

    if subtask_type in {"GOTO_OBJECT", "GOTO_LANDMARK", "RETURN_TO"}:
        anchor = anchors_by_subtask_index.get(subtask_index)
        if anchor is None:
            return False, 0.0, -1
        if subtask_type == "RETURN_TO":
            leave_index = _first_distance_greater_than_threshold_index(
                trajectory,
                anchor.target_center_m,
                2.0,
                start_index=order_start_index,
            )
            if leave_index < 0:
                return False, 0.0, -1
            min_dist, timestamp = _min_distance_xy_with_index(
                trajectory[leave_index + 1 :],
                anchor.target_center_m,
            )
            timestamp = timestamp + leave_index + 1 if timestamp >= 0 else -1
            success = bool(min_dist <= anchor.effective_radius_m)
            progress = 1.0 if success else _progress(anchor.effective_radius_m, min_dist)
            if success:
                timestamp = _first_distance_within_threshold_index(
                    trajectory,
                    anchor.target_center_m,
                    anchor.effective_radius_m,
                    start_index=leave_index + 1,
                )
            return success, progress, timestamp if success else -1
        min_dist, timestamp = _min_distance_xy_with_index(trajectory, anchor.target_center_m)
        success = bool(min_dist <= anchor.effective_radius_m)
        progress = 1.0 if success else _progress(anchor.effective_radius_m, min_dist)
        if success:
            timestamp = _first_distance_within_threshold_index(
                trajectory,
                anchor.target_center_m,
                anchor.effective_radius_m,
                start_index=order_start_index,
            )
        return success, progress, timestamp if success else -1

    if subtask_type == "GOTO_POINT":
        target = _coerce_xyz_m(subtask.get("position"), units)
        if target is None:
            return False, 0.0, -1
        threshold_m = max(_positive_float(subtask.get("radius"), 0.36), 0.36)
        min_dist, timestamp = _min_distance_xy_with_index(trajectory, target)
        success = bool(min_dist <= threshold_m)
        progress = 1.0 if success else _progress(threshold_m, min_dist)
        if success:
            timestamp = _first_distance_within_threshold_index(
                trajectory,
                target,
                threshold_m,
                start_index=order_start_index,
            )
        return success, progress, timestamp if success else -1

    if subtask_type == "GOTO_ROOM":
        room_id = _target_id(subtask)
        room_aabb = _room_aabb_m(private_scenario, room_id, units)
        if room_aabb is None:
            return False, 0.0, -1
        first_timestamp = _first_room_entry_index(trajectory, room_aabb)
        success = first_timestamp >= 0
        timestamp = _first_room_entry_index(trajectory, room_aabb, start_index=order_start_index) if success else -1
        return success, 1.0 if success else 0.0, timestamp

    if subtask_type == "FOLLOW_HUMAN":
        human_path = _selected_human_path(human_paths, _target_id(subtask))
        if not human_path:
            return False, 0.0, -1
        follow_distance = _positive_float(
            _navigation(private_scenario).get("follow_distance"),
            3.0,
        )
        inner_threshold = _positive_float(
            _navigation(private_scenario).get("follow_inner_threshold"),
            0.1,
        )
        selected_paths = {"human": human_path}
        success = float(
            compute_follow_human_task_success(
                human_paths=selected_paths,
                trajectory=list(trajectory_points),
                distance_threshold=follow_distance,
                inner_threshold=inner_threshold,
            )
        )
        ratio = float(
            compute_follow_human_success_ratio(
                human_paths=selected_paths,
                trajectory=list(trajectory_points),
                distance_threshold=follow_distance,
                inner_threshold=inner_threshold,
            )
        )
        return success >= 1.0, max(0.0, min(1.0, ratio)), -1

    # Non-navigational instruction markers — always pass, don't affect ordering
    if subtask_type in ("VLN", "OBJ", "SOCIAL", "EQA"):
        return True, 1.0, -1

    return False, 0.0, -1


def _units_in_meters(private_scenario: Mapping[str, Any]) -> float:
    scene = private_scenario.get("scene")
    if not isinstance(scene, dict):
        return 1.0
    try:
        units = float(scene.get("units_in_meters", 1.0))
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(units) or units <= 0:
        return 1.0
    return units


def _target_id(subtask: Mapping[str, Any]) -> Optional[str]:
    for key in ("object_id", "landmark_id", "room_id", "target_id", "target"):
        value = subtask.get(key)
        if value is not None:
            return str(value)
    for key, value in subtask.items():
        if isinstance(key, str) and key.lower().endswith("_id") and value is not None:
            return str(value)
    return None


def _navigation(private_scenario: Mapping[str, Any]) -> Mapping[str, Any]:
    task = private_scenario.get("task")
    if not isinstance(task, dict):
        return {}
    navigation = task.get("navigation")
    if not isinstance(navigation, dict):
        return {}
    return navigation


def _private_qa(private_scenario: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    task = private_scenario.get("task")
    if not isinstance(task, dict):
        return None
    qa = task.get("qa")
    return qa if isinstance(qa, dict) else None


def _room_aabb_m(
    private_scenario: Mapping[str, Any],
    room_id: Optional[str],
    units: float,
) -> Optional[Tuple[XYZ, XYZ]]:
    if not room_id:
        return None
    room_zones = _navigation(private_scenario).get("room_zone")
    if not isinstance(room_zones, dict):
        return None
    key = _match_mapping_key(room_zones, room_id)
    if key is None:
        return None
    zone = room_zones.get(key)
    if isinstance(zone, dict) and "room_zone" in zone and "aabb_min" not in zone:
        nested = zone.get("room_zone")
        if isinstance(nested, dict):
            zone = nested
    if not isinstance(zone, dict):
        return None
    aabb_min = _coerce_xyz_m(zone.get("aabb_min"), units)
    aabb_max = _coerce_xyz_m(zone.get("aabb_max"), units)
    if aabb_min is None or aabb_max is None:
        return None
    return aabb_min, aabb_max


def _match_mapping_key(mapping: Mapping[str, Any], target_id: str) -> Optional[str]:
    if target_id in mapping:
        return target_id
    target_normalized = _normalize_key(target_id)
    for key in mapping:
        if _normalize_key(str(key)) == target_normalized:
            return str(key)
    return None


def _normalize_key(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _coerce_xyz_m(value: Any, units: float) -> Optional[XYZ]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return (
            float(value[0]) * units,
            float(value[1]) * units,
            float(value[2]) * units,
        )
    except (TypeError, ValueError):
        return None


def _submitted_human_paths(
    result_payload: Mapping[str, Any],
    coordinate_scale: float = 1.0,
) -> Dict[str, List[Dict[str, XYZ]]]:
    raw_human_paths = result_payload.get("human_paths")
    if not isinstance(raw_human_paths, dict):
        return {}
    human_paths: Dict[str, List[Dict[str, XYZ]]] = {}
    scale = _positive_float(coordinate_scale, 1.0)
    for human_id, raw_path in raw_human_paths.items():
        if not isinstance(raw_path, list):
            continue
        path = []
        for waypoint in raw_path:
            if isinstance(waypoint, dict):
                xyz = waypoint.get("position", waypoint.get("xyz"))
            else:
                xyz = waypoint
            position = _coerce_xyz_m(xyz, 1.0)
            if position is not None:
                item = {"position": _scale_xyz(position, scale)}
                if isinstance(waypoint, Mapping):
                    frame = _coerce_optional_int(waypoint.get("frame"))
                    step = _coerce_optional_int(waypoint.get("step"))
                    time_s = _coerce_optional_float(waypoint.get("time_s"))
                    if frame is not None:
                        item["frame"] = frame
                    if step is not None:
                        item["step"] = step
                    if time_s is not None:
                        item["time_s"] = time_s
                path.append(item)
        if path:
            human_paths[str(human_id)] = path
    return human_paths


def _selected_human_path(
    human_paths: Mapping[str, List[Dict[str, XYZ]]],
    human_id: Optional[str],
) -> Optional[List[Dict[str, XYZ]]]:
    if not human_paths:
        return None
    if human_id:
        key = _match_mapping_key(human_paths, human_id)
        if key is not None:
            return human_paths.get(key)
    return next(iter(human_paths.values()))


def _social_violation_ratio(
    *,
    private_scenario: Mapping[str, Any],
    trajectory_points: Sequence[_OfflineTrajectoryPoint],
    human_paths: Mapping[str, List[Dict[str, XYZ]]],
    units: float,
) -> float:
    if not trajectory_points:
        return 0.0

    static_humans: List[XYZ] = []
    objects = (_navigation(private_scenario).get("objects") or {})
    if isinstance(objects, dict):
        for name, pos in objects.items():
            if not str(name).startswith("Human"):
                continue
            xyz = _coerce_xyz_m(pos, units)
            if xyz is not None:
                static_humans.append(xyz)

    step_aligned_paths = (
        [_step_position_index(path) for path in human_paths.values()] if human_paths else []
    )
    frame_aligned_paths = (
        _vh_gt_waypoint_paths(private_scenario, units) if not step_aligned_paths else []
    )

    if not static_humans and not step_aligned_paths and not frame_aligned_paths:
        return 0.0

    threshold_m = 1.2
    violation_steps = 0
    for robot_point in trajectory_points:
        robot_position = robot_point.position
        violated = False
        for human_pos in static_humans:
            if _distance_xy(robot_position, human_pos) <= threshold_m:
                violated = True
                break
        if not violated and step_aligned_paths:
            robot_step = robot_point.step
            if robot_step is not None:
                for path_by_step in step_aligned_paths:
                    human_pos = path_by_step.get(int(robot_step))
                    if human_pos is not None and _distance_xy(robot_position, human_pos) <= threshold_m:
                        violated = True
                        break
        if not violated and frame_aligned_paths:
            robot_step = robot_point.step
            if robot_step is not None:
                for seq in frame_aligned_paths:
                    human_pos = _interp_xyz_at_frame(seq, int(robot_step))
                    if _distance_xy(robot_position, human_pos) <= threshold_m:
                        violated = True
                        break
        if violated:
            violation_steps += 1
    return float(violation_steps / len(trajectory_points))


def _vh_gt_waypoint_paths(
    private_scenario: Mapping[str, Any],
    units: float = 1.0,
) -> List[List[Tuple[int, float, float, float]]]:
    """Extract per-character (frame, x, y, z) waypoint sequences from virtual_humans.vh_gt_waypoints.

    Trajectory ``step`` and waypoint ``frame`` are the same simulation index under different
    names, so callers should align by ``robot_point.step``.
    """
    vh = private_scenario.get("virtual_humans")
    if not isinstance(vh, dict):
        return []
    waypoints = vh.get("vh_gt_waypoints")
    if not isinstance(waypoints, dict):
        return []
    paths: List[List[Tuple[int, float, float, float]]] = []
    for _name, wps in waypoints.items():
        if not isinstance(wps, list):
            continue
        seq: List[Tuple[int, float, float, float]] = []
        for wp in wps:
            if not isinstance(wp, Mapping):
                continue
            frame = _coerce_optional_int(wp.get("frame"))
            xyz = _coerce_xyz_m(wp.get("xyz"), units)
            if frame is None or xyz is None:
                continue
            seq.append((int(frame), float(xyz[0]), float(xyz[1]), float(xyz[2])))
        seq.sort(key=lambda item: item[0])
        if seq:
            paths.append(seq)
    return paths


def _interp_xyz_at_frame(
    seq: Sequence[Tuple[int, float, float, float]],
    frame: int,
) -> XYZ:
    if not seq:
        return (0.0, 0.0, 0.0)
    frames = [item[0] for item in seq]
    from bisect import bisect_left
    index = bisect_left(frames, frame)
    if index <= 0:
        _, x, y, z = seq[0]
        return (x, y, z)
    if index >= len(seq):
        _, x, y, z = seq[-1]
        return (x, y, z)
    f1, x1, y1, z1 = seq[index - 1]
    f2, x2, y2, z2 = seq[index]
    if f2 == f1:
        return (x1, y1, z1)
    t = (frame - f1) / (f2 - f1)
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1), z1 + t * (z2 - z1))


def _step_position_index(path: Sequence[Mapping[str, Any]]) -> Dict[int, XYZ]:
    positions: Dict[int, XYZ] = {}
    for entry in path:
        if not isinstance(entry, Mapping):
            continue
        step = _coerce_optional_int(entry.get("step"))
        position = entry.get("position")
        if step is not None and isinstance(position, tuple) and len(position) >= 3:
            positions[int(step)] = position
    return positions


def _positive_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result) or result <= 0:
        return float(default)
    return result


def _min_distance_xy(trajectory: Sequence[XYZ], target_center: XYZ) -> float:
    min_dist, _ = _min_distance_xy_with_index(trajectory, target_center)
    return min_dist


def _min_distance_xy_with_index(trajectory: Sequence[XYZ], target_center: XYZ) -> Tuple[float, int]:
    if not trajectory:
        return float("inf"), -1
    best_dist = float("inf")
    best_index = -1
    for index, point in enumerate(trajectory):
        dist = _distance_xy(point, target_center)
        if dist < best_dist:
            best_dist = dist
            best_index = index
    return best_dist, best_index


def _first_room_entry_index(
    trajectory: Sequence[XYZ],
    room_aabb: Tuple[XYZ, XYZ],
    *,
    start_index: int = 0,
) -> int:
    aabb_min, aabb_max = room_aabb
    min_x, max_x = sorted((float(aabb_min[0]), float(aabb_max[0])))
    min_y, max_y = sorted((float(aabb_min[1]), float(aabb_max[1])))
    min_z, max_z = sorted((float(aabb_min[2]), float(aabb_max[2])))
    for index in range(max(0, int(start_index)), len(trajectory)):
        point = trajectory[index]
        if (
            min_x <= float(point[0]) <= max_x
            and min_y <= float(point[1]) <= max_y
            and min_z <= float(point[2]) <= max_z
        ):
            return index
    return -1


def _first_distance_greater_than_threshold_index(
    trajectory: Sequence[XYZ],
    target_center: XYZ,
    threshold_m: float,
    *,
    start_index: int = 0,
) -> int:
    for index in range(max(0, int(start_index)), len(trajectory)):
        point = trajectory[index]
        if _distance_xy(point, target_center) > threshold_m:
            return index
    return -1


def _first_distance_within_threshold_index(
    trajectory: Sequence[XYZ],
    target_center: XYZ,
    threshold_m: float,
    *,
    start_index: int = 0,
) -> int:
    for index in range(max(0, int(start_index)), len(trajectory)):
        point = trajectory[index]
        if _distance_xy(point, target_center) <= threshold_m:
            return index
    return -1


_ORDER_EXEMPT_SUBTASK_TYPES = {"FOLLOW_HUMAN", "VLN", "OBJ", "SOCIAL", "EQA"}


def _subtasks_completed_in_order(timestamps: Sequence[Tuple[bool, int, str]]) -> bool:
    previous_timestamp = -1
    for success, timestamp, subtask_type in timestamps:
        if subtask_type in _ORDER_EXEMPT_SUBTASK_TYPES:
            continue
        if not success or timestamp < 0:
            return False
        if previous_timestamp >= 0 and timestamp <= previous_timestamp:
            return False
        previous_timestamp = timestamp
    return True


def _progress(effective_radius: float, min_dist: float) -> float:
    if not math.isfinite(min_dist) or min_dist <= 0:
        return 0.0
    return float(min(1.0, effective_radius / min_dist))


def _estimate_shortest_path(waypoints: Sequence[XYZ]) -> float:
    """Same logic as BenchRunner._estimate_shortest_path / EpisodeRunner._estimate_shortest_path."""
    if len(waypoints) < 2:
        return 0.0
    total = 0.0
    for i in range(len(waypoints) - 1):
        total += _distance_xy(waypoints[i], waypoints[i + 1])
    return total


def _distance_xy(a: XYZ, b: XYZ) -> float:
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return (dx * dx + dy * dy) ** 0.5


def _segment_min_distance_xy(a: XYZ, b: XYZ, p: XYZ) -> float:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    px, py = float(p[0]), float(p[1])
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 0.0:
        return _distance_xy(a, p)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    cx, cy = ax + t * dx, ay + t * dy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


if __name__ == "__main__":
    sys.exit(main())
