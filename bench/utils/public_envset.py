"""Utilities for producing public envset files from private source envsets."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, MutableMapping, Tuple


ROOT_PRIVATE_KEYS = {
    "recording",
    "gt_path",
    "gt_waypoints",
    "expert_path",
    "pending_annotations",
    "preprocessing",
    "qa",
    "judge_key",
    "private_judge_key",
    "private_candidates",
}

ROBOT_PRIVATE_KEYS = {
    "rb_gt_waypoints",
    "gt_waypoints",
    "expert_path",
}

NAVIGATION_PRIVATE_KEYS = {
    "answer",
    "goal_position",
    "objects",
    "objects_meta",
    "landmarks",
    "landmarks_evidence",
    "room_zone",
    "judge_key",
    "private_judge_key",
    "private_candidates",
    "candidate_points",
    "goal_candidates",
    "gt_anchor",
    "d_gt",
    "success_radius",
}

TASK_PRIVATE_KEYS = {
    "pending_annotations",
    "preprocessing",
    "qa",
    "sub_instructions",
    "subtasks",
}


def sanitize_envset_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return a public copy of an envset payload plus a removal report.

    The sanitizer is intentionally path-specific: it removes known answer/GT
    fields while preserving scene, simulator, NavMesh, robot, and dynamic-human
    route configuration needed for local execution.
    """

    sanitized = copy.deepcopy(payload)
    removed_fields: List[Dict[str, Any]] = []

    scenarios = sanitized.get("scenarios")
    if isinstance(scenarios, list):
        for idx, scenario in enumerate(scenarios):
            if isinstance(scenario, dict):
                _sanitize_scenario(
                    scenario,
                    path=f"scenarios[{idx}]",
                    removed_fields=removed_fields,
                )
    elif isinstance(sanitized, dict):
        _sanitize_scenario(sanitized, path="$", removed_fields=removed_fields)

    report = {
        "num_scenarios": len(scenarios) if isinstance(scenarios, list) else 1,
        "num_removed_fields": len(removed_fields),
        "removed_fields": removed_fields,
    }
    return sanitized, report


def _sanitize_scenario(
    scenario: MutableMapping[str, Any],
    *,
    path: str,
    removed_fields: List[Dict[str, Any]],
) -> None:
    scenario_id = str(scenario.get("id", ""))

    for key in sorted(ROOT_PRIVATE_KEYS):
        _drop_key(
            scenario,
            key,
            path=f"{path}.{key}",
            scenario_id=scenario_id,
            removed_fields=removed_fields,
        )

    robots = scenario.get("robots")
    if isinstance(robots, dict):
        entries = robots.get("entries")
        if isinstance(entries, list):
            for idx, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    continue
                for key in sorted(ROBOT_PRIVATE_KEYS):
                    _drop_key(
                        entry,
                        key,
                        path=f"{path}.robots.entries[{idx}].{key}",
                        scenario_id=scenario_id,
                        removed_fields=removed_fields,
                    )

    task = scenario.get("task")
    if not isinstance(task, dict):
        return
    for key in sorted(TASK_PRIVATE_KEYS):
        _drop_key(
            task,
            key,
            path=f"{path}.task.{key}",
            scenario_id=scenario_id,
            removed_fields=removed_fields,
        )

    navigation = task.get("navigation")
    if not isinstance(navigation, dict):
        return
    for key in sorted(NAVIGATION_PRIVATE_KEYS):
        _drop_key(
            navigation,
            key,
            path=f"{path}.task.navigation.{key}",
            scenario_id=scenario_id,
            removed_fields=removed_fields,
        )


def _drop_key(
    mapping: MutableMapping[str, Any],
    key: str,
    *,
    path: str,
    scenario_id: str,
    removed_fields: List[Dict[str, Any]],
) -> None:
    if key not in mapping:
        return
    mapping.pop(key)
    removed_fields.append({"scenario_id": scenario_id, "path": path})
