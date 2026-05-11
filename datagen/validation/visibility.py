from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass
class VisibilityStats:
    frames_checked: int = 0
    max_pixels: int = 0
    last_pixels: int = 0
    matched_instance_ids: Tuple[int, ...] = ()


def count_visible_pixels_instance_id(
    instance_id_segmentation: Any,
    *,
    target_prim_path: str,
    target_category: Optional[str] = None,
    allow_category_fallback: bool = True,
) -> Tuple[int, Tuple[int, ...]]:
    """Return (pixels, matched_instance_ids) for the target.

    The annotator output is expected to follow replicator conventions:
      - Either a dict with keys: "data" (HxW) and "info" containing "idToLabels"
      - Or a raw numpy array of ids (no mapping)
    """
    data = None
    info = None

    if isinstance(instance_id_segmentation, dict):
        data = instance_id_segmentation.get("data")
        info = instance_id_segmentation.get("info") or {}
    else:
        data = instance_id_segmentation
        info = {}

    if data is None:
        return 0, ()

    arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        return 0, ()

    id_to_labels = {}
    if isinstance(info, dict):
        id_to_labels = info.get("idToLabels") or {}

    ids = _match_instance_ids(
        id_to_labels,
        target_prim_path=target_prim_path,
        target_category=target_category,
        allow_category_fallback=bool(allow_category_fallback),
    )
    if not ids:
        return 0, ()

    if len(ids) == 1:
        pixels = int(np.count_nonzero(arr == ids[0]))
        return pixels, tuple(ids)

    mask = np.isin(arr, np.asarray(ids, dtype=arr.dtype))
    return int(np.count_nonzero(mask)), tuple(ids)


def _match_instance_ids(
    id_to_labels: Any,
    *,
    target_prim_path: str,
    target_category: Optional[str],
    allow_category_fallback: bool,
) -> List[int]:
    """Map replicator idToLabels to instance ids for a target prim path.

    When allow_category_fallback=False, only prim-path based matches are allowed.
    """
    if not isinstance(id_to_labels, dict) or not target_prim_path:
        return []

    target_prim_path = str(target_prim_path)
    target_norm = target_prim_path.rstrip("/")

    matched: List[int] = []
    for raw_id, label in id_to_labels.items():
        try:
            inst_id = int(raw_id)
        except Exception:
            try:
                inst_id = int(str(raw_id))
            except Exception:
                continue

        # Common cases:
        # - label is a dict containing prim path fields + class
        # - label is a simple string label (class name)
        if isinstance(label, dict):
            prim_candidates: List[str] = []
            for k in ("primPath", "prim_path", "path", "prim", "instancePrimPath"):
                v = label.get(k)
                if isinstance(v, str) and v:
                    prim_candidates.append(v)
            v_multi = label.get("primPaths")
            if isinstance(v_multi, (list, tuple)):
                prim_candidates.extend([str(x) for x in v_multi if isinstance(x, str)])

            for p in prim_candidates:
                p_norm = str(p).rstrip("/")
                if p_norm == target_norm or p_norm.endswith(target_norm):
                    matched.append(inst_id)
                    break
            else:
                # Optional fallback by class label (can be ambiguous). Disable in strict mode.
                if allow_category_fallback and target_category:
                    cls = label.get("class") or label.get("label") or label.get("semanticLabel")
                    if isinstance(cls, str) and cls.lower() == str(target_category).lower():
                        matched.append(inst_id)
        elif allow_category_fallback and target_category and isinstance(label, str):
            if label.lower() == str(target_category).lower():
                matched.append(inst_id)

    return sorted(set(matched))


def count_visible_pixels_by_category(
    instance_id_segmentation: Any,
    *,
    allowed_categories: Optional[Set[str]] = None,
) -> Dict[str, int]:
    """Return {category: pixels} from an instance-id segmentation annotator output.

    This aggregates pixels for all instance ids that share the same class/category label.
    """
    data = None
    info = None
    if isinstance(instance_id_segmentation, dict):
        data = instance_id_segmentation.get("data")
        info = instance_id_segmentation.get("info") or {}
    else:
        data = instance_id_segmentation
        info = {}

    if data is None:
        return {}

    arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        return {}

    id_to_labels: Dict[Any, Any] = {}
    if isinstance(info, dict):
        id_to_labels = info.get("idToLabels") or {}

    # Compute pixel counts per instance id.
    ids, counts = np.unique(arr, return_counts=True)
    id_to_count = {int(i): int(c) for i, c in zip(ids.tolist(), counts.tolist())}

    out: Dict[str, int] = {}
    for raw_id, label in (id_to_labels or {}).items():
        try:
            inst_id = int(raw_id)
        except Exception:
            try:
                inst_id = int(str(raw_id))
            except Exception:
                continue
        pixels = id_to_count.get(inst_id, 0)
        if pixels <= 0:
            continue
        cat = _extract_class_label(label)
        if not cat:
            continue
        if allowed_categories is not None and cat not in allowed_categories:
            continue
        out[cat] = int(out.get(cat, 0) + int(pixels))

    return out


def _extract_class_label(label: Any) -> Optional[str]:
    if isinstance(label, str):
        val = label.strip()
        return val or None
    if isinstance(label, dict):
        for k in ("class", "label", "semanticLabel", "semantic_label", "category"):
            v = label.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None
