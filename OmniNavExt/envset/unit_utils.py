from __future__ import annotations

import re
from typing import Final, Sequence, Tuple


class GoToCommandError(ValueError):
    """Raised when GoTo commands contain invalid coordinates."""



def resolve_env_unit_scale(units_value, *, context: str = "scene.units_in_meters") -> float:
    """
    Normalize envset scene units (meters per env unit).

    Args:
        units_value: Raw value from envset (can be str/float/int).
        context: Human readable field name for error reporting.

    Returns:
        Positive float.
    """
    if units_value is None:
        raise ValueError(f"[Envset] Missing {context}; please set scene.units_in_meters in envset.")
    try:
        units = float(units_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"[Envset] Invalid {context}={units_value}: {exc}") from exc
    if units <= 0.0:
        raise ValueError(f"[Envset] {context} must be > 0, got {units}")
    return units


def scale_env_position(position: Sequence, env_unit_scale: float) -> Tuple[float, float, float]:
    """Scale a 3D position from env units into stage meters."""
    try:
        x, y, z = position
    except (TypeError, ValueError):
        raise TypeError(f"[Envset] Invalid position '{position}'")
    try:
        return (
            float(x) * env_unit_scale,
            float(y) * env_unit_scale,
            float(z) * env_unit_scale,
        )
    except (TypeError, ValueError):
        raise TypeError(f"[Envset] Failed to scale position '{position}'")

_GOTO_PATTERN: Final = re.compile(
    r"""
    ^(?P<prefix>\S+)\s+             # character name or other prefix
    (?P<cmd>goto\w*)\s+             # GoTo / GoToSomething
    (?P<x>-?\d+(?:\.\d+)?)\s+       # x
    (?P<y>-?\d+(?:\.\d+)?)\s+       # y
    (?P<z>-?\d+(?:\.\d+)?)
    (?P<suffix>\s+.*)?$             # remaining args, e.g. "_" or other flags
    """,
    re.IGNORECASE | re.VERBOSE,
)

def _format_coord(value: float) -> str:
    """Format a coordinate: keep up to 6 decimals, then trim trailing zeros / dots."""
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"

def try_scale_goto_command(command: str | None, env_unit_scale: float) -> str:
    """Detect and rescale the xyz coordinates of GoTo-style commands.

    The transform is applied only when the command matches
    ``"<prefix> GoTo* x y z [suffix...]"``; other inputs are returned unchanged.
    """

    if command is None:
        return ""

    stripped = command.strip()
    if not stripped:
        return ""

    match = _GOTO_PATTERN.match(stripped)
    if not match:
        # Not a GoTo command; return the original string unchanged.
        return command

    try:
        x = float(match["x"]) * env_unit_scale
        y = float(match["y"]) * env_unit_scale
        z = float(match["z"]) * env_unit_scale
    except (ValueError, TypeError) as exc:
        raise GoToCommandError(
            f"[Envset] GoTo command '{command}' contains non-numeric coordinates"
        ) from exc

    scaled = (
        f'{match["prefix"]} {match["cmd"]} '
        f"{_format_coord(x)} {_format_coord(y)} {_format_coord(z)}"
    )
    if match["suffix"]:
        scaled += match["suffix"]

    return scaled

