from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _log_info(msg: str) -> None:
    try:
        import carb  # type: ignore

        carb.log_info(msg)
    except Exception:
        print(msg)


def _log_warn(msg: str) -> None:
    try:
        import carb  # type: ignore

        carb.log_warn(msg)
    except Exception:
        print(f"[WARN] {msg}")


@dataclass(frozen=True)
class InstructionContext:
    task_type: str
    target_category: Optional[str] = None
    target_room_name: Optional[str] = None
    available_categories: Sequence[str] = ()
    vln_landmarks: Sequence[str] = ()
    vln_landmark_evidence: Dict[str, Sequence[int]] = field(default_factory=dict)

class InstructionGenerator(ABC):
    @abstractmethod
    def generate(self, keyframes: List[str], ctx: InstructionContext) -> str: ...

class TemplateGenerator(InstructionGenerator):
    """
    Generates deterministic instructions based on object/room names.
    Useful for 'Follow' tasks or simple 'GoTo'.
    """
    def generate(self, keyframes: List[str], ctx: InstructionContext) -> str:
        if ctx.task_type.lower() == "vln":
            target = ctx.target_category or "target"
            lms = [str(x) for x in (ctx.vln_landmarks or []) if x]
            lms = [x for x in lms if ctx.target_category is None or x.lower() != str(ctx.target_category).lower()]
            if len(lms) >= 2:
                return f"Go to the {target}. On the way, pass the {lms[0]} and the {lms[1]}."
            if len(lms) == 1:
                return f"Go to the {target}. You should see a {lms[0]} along the way."
            return f"Go to the {target}."
        if ctx.task_type.lower() == "follow":
            return "Follow the person ahead and stop when they stop."
        if ctx.target_category and ctx.target_room_name and ctx.target_room_name != "unknown":
            return f"Go to the {ctx.target_category} in the {ctx.target_room_name}."
        if ctx.target_category:
            return f"Go to the {ctx.target_category}."
        return "Navigate to the target location."

class VLMGenerator(InstructionGenerator):
    """
    Interfaces with VLM (e.g., Qwen-VL) to generate instructions from frames.
    """
    def __init__(self, api_key: str = None, model_name: str = "qwen3-vl-plus"):
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
        self._model = model_name

    def generate(self, keyframes: List[str], ctx: InstructionContext) -> str:
        """
        Synthesizes an instruction from the visual sequence.
        """
        if not keyframes:
            raise RuntimeError("VLM instruction generation requires keyframes but got an empty list")

        if not self._api_key:
            raise RuntimeError("VLM instruction generation requires an API key (set DASHSCOPE_API_KEY/QWEN_API_KEY)")

        _log_info(f"[VLMGenerator] Generating instruction from {len(keyframes)} frames using model={self._model}")

        text = self._call_dashscope_vl(keyframes)
        if not text:
            raise RuntimeError("VLM returned an empty response")

        verified = self._verify_landmarks(
            text,
            available_categories=ctx.available_categories,
            target_category=ctx.target_category,
            allowed_landmarks=set(ctx.vln_landmarks) if ctx.vln_landmarks else None,
            landmark_evidence=ctx.vln_landmark_evidence if ctx.vln_landmark_evidence else None,
        )
        if not verified:
            raise RuntimeError("VLM instruction failed landmark verification")

        return text

    def _call_dashscope_vl(self, keyframes: List[str]) -> Optional[str]:
        """Call DashScope/Qwen-VL if available.

        Network access is required; if unavailable, returns None.
        """
        try:
            import dashscope  # type: ignore
        except Exception:
            raise RuntimeError("dashscope is not installed; cannot call VLM")

        dashscope.api_key = self._api_key
        # Keep payload minimal: a single user message with multiple images.
        content: List[Dict[str, Any]] = [{"type": "text", "text": "Generate a navigation instruction for this trajectory."}]
        for p in keyframes[:8]:  # cap to keep request small
            content.append({"type": "image", "image": p})

        try:
            resp = dashscope.MultiModalConversation.call(
                model=self._model,
                messages=[{"role": "user", "content": content}],
            )
        except Exception as exc:
            raise RuntimeError(f"VLM call failed: {exc}") from exc

        # Best-effort response parsing
        try:
            choices = resp.get("output", {}).get("choices", [])
            if not choices:
                raise RuntimeError("VLM response has no choices")
            message = choices[0].get("message", {})
            parts = message.get("content", [])
            for part in parts:
                if part.get("type") == "text":
                    return str(part.get("text", "")).strip()
            raise RuntimeError("VLM response contains no text content")
        except Exception as exc:
            raise RuntimeError(f"Failed to parse VLM response: {exc}") from exc

    @staticmethod
    def _verify_landmarks(
        text: str,
        *,
        available_categories: Sequence[str],
        target_category: Optional[str] = None,
        allowed_landmarks: Optional[Set[str]] = None,
        landmark_evidence: Optional[Dict[str, Sequence[int]]] = None,
    ) -> bool:
        """Verification that reduces language/vision mismatch.

        - Always: disallow mentioning categories absent from the registry.
        - For VLN (when allowed_landmarks provided): disallow mentioning categories
          outside {target_category} ∪ allowed_landmarks, and require evidence for mentioned landmarks.
        """
        if not available_categories:
            return True
        text_l = text.lower()
        cats = {str(c).lower() for c in available_categories if c}
        mentioned = {c for c in cats if _mentions_category(text_l, c)}
        # If it mentions nothing, accept (could be a geometry-based instruction).
        if not mentioned:
            return True

        # If it mentions something, ensure those mentions exist in registry.
        if not all(m in cats for m in mentioned):
            return False

        if allowed_landmarks is None:
            return True

        allowed_lc = {str(x).lower() for x in allowed_landmarks if x}
        target_lc = str(target_category).lower() if target_category else ""
        permitted = set(allowed_lc)
        if target_lc:
            permitted.add(target_lc)

        # Reject mentions outside the permitted set (for VLN closure).
        if any(m not in permitted for m in mentioned):
            return False

        # Evidence requirement for landmarks (target may be handled separately).
        if landmark_evidence:
            evidence_lc = {str(k).lower(): v for k, v in landmark_evidence.items() if k}
            for m in mentioned:
                if target_lc and m == target_lc:
                    continue
                frames = evidence_lc.get(m)
                if not frames:
                    return False

        return True


def _mentions_category(text_l: str, category_lc: str) -> bool:
    """Heuristic mention detector tolerant to '_' vs whitespace."""
    cat = str(category_lc).strip().lower()
    if not cat:
        return False
    # Build a regex that treats '_' and whitespace as interchangeable separators.
    parts = [re.escape(p) for p in re.split(r"[_\\s]+", cat) if p]
    if not parts:
        return False
    pattern = r"\\b" + r"[_\\s]+".join(parts) + r"\\b"
    return re.search(pattern, text_l) is not None
