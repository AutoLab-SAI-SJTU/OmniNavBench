from __future__ import annotations

from typing import Mapping

from bench.configs.execution import ExecutionMode


POLICY_MODE_MAP: dict[str, ExecutionMode] = {
    "ForwardPolicy": ExecutionMode.STEP_ACTION,
    "UniNaVidHTTPPolicy": ExecutionMode.STEP_ACTION,
    "UniNaVidWaypointHTTPPolicy": ExecutionMode.STEP_ACTION,
    "UniNaVidWaypointPointsHTTPPolicy": ExecutionMode.WAYPOINT,
    "MTU3DHTTPPolicy": ExecutionMode.WAYPOINT,
    "PoliFormerHTTPPolicy": ExecutionMode.STEP_ACTION,
    "NaVILAHTTPPolicy": ExecutionMode.STEP_ACTION,
    "OmniNavHTTPPolicy": ExecutionMode.WAYPOINT,
}


def default_policy_mode_map() -> dict[str, ExecutionMode]:
    """Return a copy of the built-in policy-to-execution-mode registry."""
    return dict(POLICY_MODE_MAP)


def resolve_policy_mode(policy, overrides: Mapping[str, ExecutionMode] | None = None) -> ExecutionMode:
    """Resolve an execution mode from policy attribute or registry."""
    declared_mode = getattr(policy, "execution_mode", None)
    if declared_mode is not None:
        return ExecutionMode(declared_mode)

    mapping = dict(POLICY_MODE_MAP)
    if overrides:
        mapping.update(overrides)

    name = policy.__class__.__name__
    if name in mapping:
        return mapping[name]
    lower_map = {key.lower(): value for key, value in mapping.items()}
    if name.lower() in lower_map:
        return lower_map[name.lower()]
    raise ValueError(f"No execution mode configured for policy {name}")
