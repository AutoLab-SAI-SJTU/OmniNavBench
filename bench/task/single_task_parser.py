"""Single-task parser for VLN evaluation.

Parses the instruction and goal of the first VLN subtask in an envset scenario.
"""

from typing import Any, Dict, List, Optional, Tuple

# Subtask types that count as a "VLN" subtask for single-task mode.
VLN_SUBTASK_TYPES = {"GOTO_OBJECT", "GOTO_LANDMARK", "GOTO_ROOM", "RETURN_TO"}


def parse_single_task(scenario: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse single-task info from a scenario.

    Args:
        scenario: a single scenario dict from an envset.

    Returns:
        A dict with the following keys, or None if the episode should be skipped:
        - instruction:   str
        - goal_position: (x, y, z) tuple
        - goal_type:     "object" or "room"
        - goal_id:       target object/room id
        - goal_aabb:     room AABB (only for room goals)
    """
    task_cfg = scenario.get("task", {})
    if not isinstance(task_cfg, dict):
        raise ValueError(f"scenario.task must be a dict, got {type(task_cfg)}")

    nav_cfg = task_cfg.get("navigation", {})
    if not isinstance(nav_cfg, dict):
        raise ValueError(f"scenario.task.navigation must be a dict")

    subtasks = task_cfg.get("subtasks", [])
    sub_instructions = task_cfg.get("sub_instructions", [])

    if not subtasks:
        raise ValueError("scenario.task.subtasks is empty")

    vln_subtasks = _extract_vln_subtasks(subtasks)
    if not vln_subtasks:
        raise ValueError("No valid VLN subtasks found")

    last_subtask = vln_subtasks[-1]
    goal_info = _get_goal_info(last_subtask, nav_cfg)

    instruction = _build_instruction(sub_instructions)

    return {
        "instruction": instruction,
        "goal_position": goal_info["position"],
        "goal_type": goal_info["type"],
        "goal_id": goal_info["id"],
        "goal_aabb": goal_info.get("aabb"),
        "subtasks": vln_subtasks,
    }


def _extract_vln_subtasks(subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pick the leading run of VLN-type subtasks (stops at the first non-VLN type)."""
    result = []
    for st in subtasks:
        st_type = str(st.get("type", "")).upper()
        if st_type in VLN_SUBTASK_TYPES:
            result.append(st)
        else:
            break
    return result


def _get_goal_info(subtask: Dict[str, Any], nav_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the goal position and type from a VLN subtask."""
    st_type = str(subtask.get("type", "")).upper()
    objects = nav_cfg.get("objects", {})
    room_zones = nav_cfg.get("room_zone", {})

    target_id = None
    for key in ["object_id", "room_id", "landmark_id", "target"]:
        if key in subtask and subtask[key] is not None:
            target_id = str(subtask[key])
            break

    if target_id is None:
        raise ValueError(f"subtask is missing the target id: {subtask}")

    if st_type == "GOTO_ROOM":
        # Room goal: use the AABB centre as the target position.
        if target_id not in room_zones:
            # Try a case-insensitive match that also tolerates space/underscore variants.
            target_id_lower = target_id.lower()
            target_id_normalized = target_id_lower.replace(' ', '_')
            matched = None
            for k in room_zones:
                k_lower = k.lower()
                k_normalized = k_lower.replace(' ', '_')
                if k_lower == target_id_lower or k_normalized == target_id_normalized:
                    matched = k
                    break
            if matched is None:
                raise ValueError(f"room '{target_id}' not found in room_zone")
            target_id = matched

        zone = room_zones[target_id]
        aabb_min = zone.get("aabb_min", [0, 0, 0])
        aabb_max = zone.get("aabb_max", [0, 0, 0])
        center = (
            (aabb_min[0] + aabb_max[0]) / 2,
            (aabb_min[1] + aabb_max[1]) / 2,
            (aabb_min[2] + aabb_max[2]) / 2,
        )
        return {
            "type": "room",
            "id": target_id,
            "position": center,
            "aabb": {"aabb_min": aabb_min, "aabb_max": aabb_max},
        }
    else:
        # GOTO_OBJECT / GOTO_LANDMARK / RETURN_TO: use the object coordinate.
        if target_id not in objects:
            # Try a case-insensitive match that also tolerates space/underscore variants.
            target_id_lower = target_id.lower()
            target_id_normalized = target_id_lower.replace(' ', '_')
            matched = None
            for k in objects:
                k_lower = k.lower()
                k_normalized = k_lower.replace(' ', '_')
                if k_lower == target_id_lower or k_normalized == target_id_normalized:
                    matched = k
                    break
            if matched is None:
                raise ValueError(f"object '{target_id}' not found in objects")
            target_id = matched

        pos = objects[target_id]
        if not isinstance(pos, (list, tuple)) or len(pos) < 3:
            raise ValueError(f"object '{target_id}' has invalid position: {pos}")

        return {
            "type": "object",
            "id": target_id,
            "position": (float(pos[0]), float(pos[1]), float(pos[2])),
        }


def _build_instruction(sub_instructions: List[Dict[str, Any]]) -> str:
    """Build the task instruction from sub_instructions.

    Rules:
    - Take the first VLN-type text.
    - Append the first SOCIAL-type text, if present.
    """
    vln_text = None
    social_text = None

    for si in sub_instructions:
        si_type = str(si.get("type", "")).upper()
        text = si.get("text", "")

        if si_type == "VLN" and vln_text is None:
            vln_text = text
        elif si_type == "SOCIAL" and social_text is None:
            social_text = text

    if vln_text is None:
        raise ValueError("No VLN instruction found in sub_instructions")

    if social_text:
        return f"{vln_text} {social_text}"
    return vln_text
