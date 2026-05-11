"""Private GT-anchor matching for offline navigation judging."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


XYZ = Tuple[float, float, float]


@dataclass(frozen=True)
class GTAnchor:
    """Internal anchor record used by the private judge.

    This object may contain private target and GT coordinates. Do not serialize it
    into public artifacts.
    """

    subtask_index: int
    subtask_type: str
    target_id: str
    target_center_m: XYZ
    gt_anchor_m: XYZ
    gt_anchor_index: int
    d_gt_m: float
    threshold_m: float
    effective_radius_m: float


@dataclass(frozen=True)
class GTAnchorBuildResult:
    episode_id: str
    units_in_meters: float
    anchors: List[GTAnchor] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _AnchorSpec:
    subtask_index: int
    subtask_type: str
    target_id: str
    target_center_m: XYZ
    threshold_m: float


def build_episode_gt_anchors(scenario: Dict[str, Any], source_path: Optional[Path] = None) -> GTAnchorBuildResult:
    """Build GT anchors for distance-to-landmark/object subtasks.

    The returned anchors are private judge internals. Use
    build_public_anchor_validation_report() for redacted aggregate diagnostics.
    """

    del source_path  # Kept for future private diagnostics without changing API.
    episode_id = str(scenario.get("id", ""))
    units = _units_in_meters(scenario)
    subtasks = _subtasks(scenario)
    objects = _objects(scenario)
    gt_path = _gt_path_m(scenario, units)

    anchors: List[GTAnchor] = []
    warnings: List[Dict[str, Any]] = []
    anchor_specs: List[_AnchorSpec] = []

    if not gt_path:
        warnings.append({"subtask_index": -1, "code": "MISSING_GT_PATH"})

    for subtask_index, subtask in enumerate(subtasks):
        if not isinstance(subtask, dict):
            continue
        subtask_type = str(subtask.get("type", "")).upper()
        if subtask_type not in {"GOTO_OBJECT", "GOTO_LANDMARK", "RETURN_TO"}:
            continue

        target_id = _target_id(subtask)
        target_raw = objects.get(target_id) if target_id is not None else None
        target_center = _coerce_xyz_m(target_raw, units)
        if target_id is None or target_center is None:
            warnings.append({"subtask_index": subtask_index, "code": "MISSING_TARGET"})
            continue
        threshold = 1.0 if subtask_type == "GOTO_OBJECT" else 3.0
        anchor_specs.append(
            _AnchorSpec(
                subtask_index=subtask_index,
                subtask_type=subtask_type,
                target_id=target_id,
                target_center_m=target_center,
                threshold_m=threshold,
            )
        )

    if not gt_path:
        return GTAnchorBuildResult(
            episode_id=episode_id,
            units_in_meters=units,
            anchors=anchors,
            warnings=warnings,
        )

    matches = _monotonic_gt_anchor_matches(anchor_specs=anchor_specs, gt_path=gt_path)
    for spec, match in zip(anchor_specs, matches):
        if match is None:
            warnings.append({"subtask_index": spec.subtask_index, "code": "NO_GT_AFTER_PREVIOUS_ANCHOR"})
            continue
        gt_anchor_index, gt_anchor, d_gt = match
        effective_radius = max(spec.threshold_m, d_gt)
        if effective_radius > spec.threshold_m:
            warnings.append({"subtask_index": spec.subtask_index, "code": "EFFECTIVE_RADIUS_EXPANDED"})

        anchors.append(
            GTAnchor(
                subtask_index=spec.subtask_index,
                subtask_type=spec.subtask_type,
                target_id=spec.target_id,
                target_center_m=spec.target_center_m,
                gt_anchor_m=gt_anchor,
                gt_anchor_index=gt_anchor_index,
                d_gt_m=d_gt,
                threshold_m=spec.threshold_m,
                effective_radius_m=effective_radius,
            )
        )

    return GTAnchorBuildResult(
        episode_id=episode_id,
        units_in_meters=units,
        anchors=anchors,
        warnings=warnings,
    )


def build_public_anchor_validation_report(results: Sequence[GTAnchorBuildResult]) -> Dict[str, Any]:
    """Build a redacted aggregate report for checking anchor health."""

    warnings: List[Dict[str, Any]] = []
    num_radius_expanded = 0
    num_missing_targets = 0
    num_missing_gt_paths = 0

    for episode_index, result in enumerate(results):
        for warning in result.warnings:
            code = str(warning.get("code", "UNKNOWN"))
            if code == "EFFECTIVE_RADIUS_EXPANDED":
                num_radius_expanded += 1
            elif code == "MISSING_TARGET":
                num_missing_targets += 1
            elif code == "MISSING_GT_PATH":
                num_missing_gt_paths += 1
            warnings.append(
                {
                    "episode_index": episode_index,
                    "subtask_index": int(warning.get("subtask_index", -1)),
                    "code": code,
                }
            )

    return {
        "summary": {
            "num_episodes": len(results),
            "num_anchors": sum(len(result.anchors) for result in results),
            "num_skipped_subtasks": num_missing_targets + num_missing_gt_paths,
            "num_missing_targets": num_missing_targets,
            "num_missing_gt_paths": num_missing_gt_paths,
            "num_radius_expanded": num_radius_expanded,
            "num_warnings": len(warnings),
        },
        "warnings": warnings,
    }


def load_anchor_results_from_envset_path(path: Path) -> List[GTAnchorBuildResult]:
    results: List[GTAnchorBuildResult] = []
    for json_file in _iter_json_files(path):
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        scenarios = payload.get("scenarios")
        if isinstance(scenarios, list):
            for scenario in scenarios:
                if isinstance(scenario, dict):
                    results.append(build_episode_gt_anchors(scenario, source_path=json_file))
        elif isinstance(payload, dict):
            results.append(build_episode_gt_anchors(payload, source_path=json_file))
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build a redacted GT-anchor validation report.")
    parser.add_argument("--private-root", type=Path, required=True, help="Private envset JSON file or directory.")
    parser.add_argument("--output", type=Path, required=True, help="Redacted report output JSON.")
    args = parser.parse_args(argv)

    results = load_anchor_results_from_envset_path(args.private_root)
    report = build_public_anchor_validation_report(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _iter_json_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    yield from sorted(p for p in path.rglob("*.json") if not p.name.endswith(".backup.json"))


def _units_in_meters(scenario: Dict[str, Any]) -> float:
    scene = scenario.get("scene")
    if not isinstance(scene, dict):
        return 1.0
    try:
        return float(scene.get("units_in_meters", 1.0))
    except (TypeError, ValueError):
        return 1.0


def _subtasks(scenario: Dict[str, Any]) -> List[Any]:
    task = scenario.get("task")
    if not isinstance(task, dict):
        return []
    subtasks = task.get("subtasks")
    return subtasks if isinstance(subtasks, list) else []


def _objects(scenario: Dict[str, Any]) -> Dict[str, Any]:
    task = scenario.get("task")
    if not isinstance(task, dict):
        return {}
    navigation = task.get("navigation")
    if not isinstance(navigation, dict):
        return {}
    objects = navigation.get("objects")
    return objects if isinstance(objects, dict) else {}


def _gt_path_m(scenario: Dict[str, Any], units: float) -> List[XYZ]:
    robots = scenario.get("robots")
    entries = robots.get("entries") if isinstance(robots, dict) else None
    if not isinstance(entries, list) or not entries or not isinstance(entries[0], dict):
        return []
    waypoints = entries[0].get("rb_gt_waypoints")
    if not isinstance(waypoints, list):
        return []

    path: List[XYZ] = []
    for waypoint in waypoints:
        if not isinstance(waypoint, dict):
            continue
        point = _coerce_xyz_m(waypoint.get("xyz"), units)
        if point is not None:
            path.append(point)
    return path


def _target_id(subtask: Dict[str, Any]) -> Optional[str]:
    for key in ("object_id", "landmark_id", "target_id", "target"):
        value = subtask.get(key)
        if value is not None:
            return str(value)
    return None


def _coerce_xyz_m(value: Any, units: float) -> Optional[XYZ]:
    if isinstance(value, dict):
        for key in ("position", "center", "xyz"):
            if key in value:
                return _coerce_xyz_m(value.get(key), units)
        return None
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


def _nearest_gt_anchor(*, target_center: XYZ, gt_path: Sequence[XYZ], start_index: int) -> Tuple[int, XYZ, float]:
    best_index = start_index
    best_point = gt_path[start_index]
    best_dist = _distance_xy(target_center, best_point)
    for idx in range(start_index + 1, len(gt_path)):
        point = gt_path[idx]
        dist = _distance_xy(target_center, point)
        if dist < best_dist:
            best_index = idx
            best_point = point
            best_dist = dist
    return best_index, best_point, best_dist


def _monotonic_gt_anchor_matches(
    *,
    anchor_specs: Sequence[_AnchorSpec],
    gt_path: Sequence[XYZ],
) -> List[Optional[Tuple[int, XYZ, float]]]:
    """Choose a globally monotonic GT index assignment for ordered subtasks."""

    num_specs = len(anchor_specs)
    num_points = len(gt_path)
    if num_specs == 0:
        return []
    if num_points == 0:
        return [None] * num_specs
    if num_specs > num_points:
        matches = _monotonic_gt_anchor_matches(anchor_specs=anchor_specs[:num_points], gt_path=gt_path)
        return matches + [None] * (num_specs - num_points)

    costs: List[List[float]] = [
        [_distance_xy(spec.target_center_m, point) for point in gt_path]
        for spec in anchor_specs
    ]
    dp: List[List[float]] = [[float("inf")] * num_points for _ in range(num_specs)]
    back: List[List[int]] = [[-1] * num_points for _ in range(num_specs)]
    for point_index in range(num_points):
        dp[0][point_index] = costs[0][point_index]

    for spec_index in range(1, num_specs):
        best_prev_cost = float("inf")
        best_prev_index = -1
        for point_index in range(num_points):
            previous_index = point_index - 1
            if previous_index >= 0 and dp[spec_index - 1][previous_index] < best_prev_cost:
                best_prev_cost = dp[spec_index - 1][previous_index]
                best_prev_index = previous_index
            if best_prev_index >= 0:
                dp[spec_index][point_index] = best_prev_cost + costs[spec_index][point_index]
                back[spec_index][point_index] = best_prev_index

    end_index = min(range(num_points), key=lambda point_index: dp[num_specs - 1][point_index])
    if dp[num_specs - 1][end_index] == float("inf"):
        return [None] * num_specs

    indices = [0] * num_specs
    indices[-1] = end_index
    for spec_index in range(num_specs - 1, 0, -1):
        indices[spec_index - 1] = back[spec_index][indices[spec_index]]

    return [
        (point_index, gt_path[point_index], costs[spec_index][point_index])
        for spec_index, point_index in enumerate(indices)
    ]


def _distance_xy(a: XYZ, b: XYZ) -> float:
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return (dx * dx + dy * dy) ** 0.5


if __name__ == "__main__":
    main()
