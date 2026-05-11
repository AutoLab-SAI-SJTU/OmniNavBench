from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


_ROOT_LIST_KEYS = ("objects", "annotations", "items", "entries")


def load_object_annotations(
    *,
    objects_path: Optional[Path] = None,
    inline_objects: Any = None,
) -> List[Dict[str, Any]]:
    """Load external semantic object annotations from a file and/or inline payload."""
    objects: List[Dict[str, Any]] = []

    if objects_path is not None:
        path = Path(objects_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        objects.extend(normalize_object_annotations(raw, source=str(path)))

    if inline_objects is not None:
        objects.extend(normalize_object_annotations(inline_objects, source="inline"))

    return objects


def normalize_object_annotations(raw: Any, *, source: str = "external") -> List[Dict[str, Any]]:
    items = _unwrap_items(raw)
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        out.append(_normalize_item(item, idx=idx, source=source))
    return out


def _unwrap_items(raw: Any) -> List[Any]:
    if raw is None:
        return []

    if isinstance(raw, list):
        return list(raw)

    if isinstance(raw, dict):
        for key in _ROOT_LIST_KEYS:
            value = raw.get(key)
            if isinstance(value, list):
                return list(value)

        if raw and all(isinstance(v, dict) for v in raw.values()):
            items: List[Dict[str, Any]] = []
            for key, value in raw.items():
                item = dict(value)
                item.setdefault("id", key)
                items.append(item)
            return items

    raise ValueError("Object annotations must be a list, a wrapper dict with objects, or an id->object mapping")


def _normalize_item(item: Any, *, idx: int, source: str) -> Dict[str, Any]:
    if not isinstance(item, Mapping):
        raise ValueError(f"Object annotation #{idx} from {source} must be a mapping")

    object_id = _first_str(item, ("id", "object_id", "name", "key"))
    if not object_id:
        raise ValueError(f"Object annotation #{idx} from {source} is missing id/object_id/name")

    category = _first_str(item, ("category", "class", "label", "semanticLabel", "semantic_label"))
    if not category:
        raise ValueError(f"Object annotation '{object_id}' from {source} is missing category/class/label")

    position = _extract_vec3(item, ("position", "xyz", "center", "location"))
    if position is None and all(k in item for k in ("x", "y", "z")):
        position = [float(item["x"]), float(item["y"]), float(item["z"])]
    if position is None:
        raise ValueError(f"Object annotation '{object_id}' from {source} is missing position/xyz")

    prim_path = _first_str(item, ("prim_path", "path"))
    if not prim_path:
        prim_path = f"/ExternalObjects/{object_id}"
    elif not prim_path.startswith("/"):
        prim_path = "/" + prim_path

    room_id = _first_str(item, ("room_id", "room"))

    return {
        "object_id": str(object_id),
        "category": str(category),
        "position": [float(position[0]), float(position[1]), float(position[2])],
        "prim_path": str(prim_path),
        "room_id": str(room_id) if room_id else None,
    }


def _extract_vec3(item: Mapping[str, Any], keys: Sequence[str]) -> Optional[List[float]]:
    for key in keys:
        value = item.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return [float(value[0]), float(value[1]), float(value[2])]
    return None
def _first_str(item: Mapping[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None
