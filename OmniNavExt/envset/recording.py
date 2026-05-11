from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CANONICAL_RECORDING_KEY = "recording"


def _scenario_instruction(scenario: Dict[str, Any]) -> str:
    if not isinstance(scenario, dict):
        return ""
    task = scenario.get("task")
    if isinstance(task, dict):
        nav = task.get("navigation")
        if isinstance(nav, dict):
            instruction = nav.get("instruction")
            if instruction is not None:
                return str(instruction)
        instruction = task.get("instruction")
        if instruction is not None:
            return str(instruction)
    return ""


def _normalize_command(command: Any) -> Optional[Dict[str, float]]:
    if not isinstance(command, dict):
        return None
    return {
        "v": float(command.get("v", 0.0)),
        "w": float(command.get("w", 0.0)),
        "lateral": float(command.get("lateral", 0.0)),
    }


def normalize_recording_waypoint(waypoint: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(waypoint, dict):
        raise ValueError(f"Invalid waypoint payload: {waypoint!r}")

    frame = waypoint.get("frame")
    xyz = waypoint.get("xyz")
    yaw_deg = waypoint.get("yaw_deg")
    time_s = waypoint.get("time_s")
    if frame is None or not isinstance(xyz, (list, tuple)) or len(xyz) < 3 or yaw_deg is None or time_s is None:
        raise ValueError(f"Waypoint missing required canonical fields: {waypoint!r}")

    normalized: Dict[str, Any] = {
        "frame": int(frame),
        "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
        "yaw_deg": float(yaw_deg),
        "time_s": float(time_s),
        "distance_xy": float(waypoint.get("distance_xy", 0.0)),
        "distance_total_xy": float(waypoint.get("distance_total_xy", 0.0)),
    }
    sim_step = waypoint.get("sim_step")
    if sim_step is not None:
        normalized["sim_step"] = int(sim_step)

    command = _normalize_command(waypoint.get("command"))
    if command is not None:
        normalized["command"] = command

    return normalized


def build_recording_payload(
    *,
    instruction: str,
    gt_path: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_path = [normalize_recording_waypoint(wp) for wp in gt_path if isinstance(wp, dict)]
    out_metadata: Dict[str, Any] = {}
    if isinstance(metadata, dict):
        out_metadata.update(metadata)

    if "distance_total_xy" not in out_metadata:
        out_metadata["distance_total_xy"] = (
            float(normalized_path[-1].get("distance_total_xy", 0.0)) if normalized_path else 0.0
        )
    if "sample_count" not in out_metadata:
        out_metadata["sample_count"] = len(normalized_path)

    return {
        "instruction": str(instruction or ""),
        "gt_path": normalized_path,
        "metadata": out_metadata,
    }


def legacy_waypoints_to_recording(
    *,
    scenario: Dict[str, Any],
    waypoints: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return build_recording_payload(
        instruction=_scenario_instruction(scenario),
        gt_path=waypoints,
        metadata=metadata or {"source": "legacy_rb_gt_waypoints"},
    )


def get_embedded_recording(scenario: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(scenario, dict):
        return None
    payload = scenario.get(CANONICAL_RECORDING_KEY)
    if not isinstance(payload, dict):
        return None
    gt_path = payload.get("gt_path")
    if not isinstance(gt_path, list) or not gt_path:
        return None
    try:
        return build_recording_payload(
            instruction=str(payload.get("instruction") or _scenario_instruction(scenario)),
            gt_path=gt_path,
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        )
    except Exception:
        return None


def resolve_recording_dirs(main_json_path: Path) -> Tuple[Path, Path]:
    main_json_path = Path(main_json_path)
    if main_json_path.name == "episode.json":
        base_dir = main_json_path.parent
        return base_dir / "video", base_dir / "path"
    return main_json_path.parent / "video" / main_json_path.stem, main_json_path.parent / "path" / main_json_path.stem


def resolve_recording_sidecar_path(main_json_path: Path) -> Path:
    _, path_dir = resolve_recording_dirs(main_json_path)
    return path_dir / "path.json"


def load_recording_sidecar(main_json_path: Path) -> Optional[Dict[str, Any]]:
    if Path(main_json_path).is_dir():
        return None
    candidates = [resolve_recording_sidecar_path(main_json_path)]
    # Backward-compatible fallback for envsets that already have sibling path/path.json.
    legacy_candidate = Path(main_json_path).parent / "path" / "path.json"
    if legacy_candidate not in candidates:
        candidates.append(legacy_candidate)

    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        gt_path = payload.get("gt_path")
        if not isinstance(gt_path, list) or not gt_path:
            continue
        try:
            return build_recording_payload(
                instruction=str(payload.get("instruction") or ""),
                gt_path=gt_path,
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
        except Exception:
            continue
    return None


def resolve_recording_payload(
    scenario: Dict[str, Any],
    *,
    envset_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    embedded = get_embedded_recording(scenario)
    if embedded is not None:
        return embedded

    if envset_path is not None:
        sidecar = load_recording_sidecar(Path(envset_path))
        if sidecar is not None:
            return sidecar

    robots = scenario.get("robots") if isinstance(scenario, dict) else None
    entries = robots.get("entries") if isinstance(robots, dict) else None
    if isinstance(entries, list) and entries and isinstance(entries[0], dict):
        legacy_waypoints = entries[0].get("rb_gt_waypoints")
        if isinstance(legacy_waypoints, list) and legacy_waypoints:
            try:
                return legacy_waypoints_to_recording(scenario=scenario, waypoints=legacy_waypoints)
            except Exception:
                return None
    return None


def resolve_recording_waypoints(
    scenario: Dict[str, Any],
    *,
    envset_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    payload = resolve_recording_payload(scenario, envset_path=envset_path)
    if not isinstance(payload, dict):
        return []
    gt_path = payload.get("gt_path")
    if not isinstance(gt_path, list):
        return []
    return [normalize_recording_waypoint(wp) for wp in gt_path if isinstance(wp, dict)]


def write_recording_sidecar(main_json_path: Path, payload: Dict[str, Any]) -> Path:
    sidecar_path = resolve_recording_sidecar_path(main_json_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return sidecar_path
