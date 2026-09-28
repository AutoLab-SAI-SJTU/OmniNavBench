"""Explicit opt-in EQA execution; no model-name or capability guessing."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


class EQASetupError(ValueError):
    """EQA was explicitly enabled but the adapter is not ready."""


def resolve_eqa_enabled(config_path: Path, override: bool | None = None) -> bool:
    """CLI override > top-level YAML enable_eqa > False (strict booleans)."""
    import yaml

    with Path(config_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, Mapping):
        raise EQASetupError("The simulation YAML must be a mapping.")
    enabled = config.get("enable_eqa", False) if override is None else override
    if type(enabled) is not bool:
        raise EQASetupError("enable_eqa must be a YAML boolean: true or false, not a quoted string.")
    return enabled


def validate_eqa_interface(policy: Any, enabled: bool) -> None:
    """Check wiring, not model capability, before launching a simulation."""
    if type(enabled) is not bool:
        raise EQASetupError("enable_eqa must be a boolean.")
    if enabled and not callable(getattr(policy, "predict_text", None)):
        raise EQASetupError(
            "EQA is enabled, but the policy has no callable predict_text(question, rgb). "
            "Implement that interface, or explicitly disable EQA with --no-enable-eqa "
            "or enable_eqa: false."
        )


def answer_text(value: Any) -> str | None:
    """Preserve numeric zero; distinguish an empty response from bad data."""
    if value is None:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            if math.isfinite(value):
                return str(value)
        except (ValueError, OverflowError):
            pass
    raise ValueError("EQA answer must be text, a finite number, or None.")


@dataclass(frozen=True)
class EQAExecution:
    status: str
    answer: str | None = None
    reason: str | None = None


def execute_eqa(policy: Any, enabled: bool, extra: Mapping[str, Any], rgb: Any) -> EQAExecution:
    """Call only for required tasks after opt-in. Never use a reference answer as input."""
    raw_qa = extra.get("qa")
    subtasks = extra.get("subtasks")
    subtasks = subtasks if isinstance(subtasks, list) else []
    required = bool(raw_qa) or any(
        isinstance(st, Mapping) and str(st.get("type", "")).upper() == "EQA"
        for st in subtasks
    )
    if not required:
        return EQAExecution("not_applicable")
    if not enabled:
        # User choice is not evidence that the model lacks EQA capability.
        return EQAExecution("disabled", reason="disabled_by_user")
    validate_eqa_interface(policy, enabled)
    qa = raw_qa if isinstance(raw_qa, Mapping) else {}
    question = qa.get("question")
    if not isinstance(question, str) or not question.strip():
        return EQAExecution("not_run", reason="missing_question")
    if rgb is None or getattr(rgb, "size", 1) == 0:
        return EQAExecution("not_run", reason="missing_final_rgb")
    try:
        raw_answer = policy.predict_text(question, rgb)
    except Exception:
        # Exceptions can contain tokens/URLs/prompts; export only a safe reason code.
        return EQAExecution("error", reason="question_answering_failed")
    try:
        answer = answer_text(raw_answer)
    except ValueError:
        return EQAExecution("error", reason="invalid_answer_type")
    return EQAExecution("answered", answer) if answer is not None else EQAExecution("no_answer")


def reusable_eqa_result(path: Path, enabled: bool) -> bool:
    """Do not silently reuse EQA-disabled results after enabling the switch."""
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    if "eqa_enabled" not in data:
        # Legacy navigation-only results remain resumable with EQA off.
        return not enabled and not data.get("eqa_answer") and not data.get("eqa_status")
    if data.get("eqa_enabled") is not enabled:
        return False
    if data.get("eqa_status") in {"error", "not_run"}:
        return False
    allowed = {"answered", "no_answer", "not_applicable"} if enabled else {"disabled", "not_applicable"}
    return data.get("eqa_status") in allowed
