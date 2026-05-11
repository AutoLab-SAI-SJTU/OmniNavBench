"""Shared path resolution for local assets and runtime kit generation."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_PATHS_ENV = REPO_ROOT / "local_paths.env"
KIT_TEMPLATE = REPO_ROOT / "omninav.python.kit"
KIT_DYNAMIC_MARKER = "    # __OMNINAV_DYNAMIC_EXT_FOLDERS__"

SCENE_ROOT_ENV = "OMNINAV_SCENE_ROOT"
BENCH_DATASET_ROOT_ENV = "OMNINAV_BENCH_DATASET_ROOT"
ISAACLAB_SOURCE_ENV = "OMNINAV_ISAACLAB_SOURCE"

_KNOWN_KEYS = (
    SCENE_ROOT_ENV,
    BENCH_DATASET_ROOT_ENV,
    ISAACLAB_SOURCE_ENV,
)


def repo_root() -> Path:
    """Return the repository root."""
    return REPO_ROOT


def repo_path(*parts: str) -> Path:
    """Build a repository-relative path."""
    return REPO_ROOT.joinpath(*parts)


def _strip_shell_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def expand_path(value: str | os.PathLike[str], *, base: Path | None = None) -> Path:
    """Expand env vars and home markers, then resolve relative to an optional base."""
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if expanded.is_absolute() or base is None:
        return expanded.resolve()
    return (base / expanded).resolve()


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _strip_shell_quotes(value.strip())
    return values


def load_local_path_values(env_path: Path | None = None) -> dict[str, str]:
    """Parse local path values from the tracked repository config."""
    return _load_env_file(env_path or LOCAL_PATHS_ENV)


def apply_local_path_env(*, env_path: Path | None = None) -> dict[str, str]:
    """Load local path defaults into the process environment without overriding explicit env."""
    os.environ.setdefault("OMNINAV_REPO_ROOT", str(REPO_ROOT))
    values = load_local_path_values(env_path)
    for key in _KNOWN_KEYS:
        value = values.get(key)
        if value and key not in os.environ:
            os.environ[key] = value
    return values


def _resolve_explicit_path(
    cli_value: str | os.PathLike[str] | None,
    env_name: str,
    *,
    env_path: Path | None = None,
    required: bool = False,
) -> Path | None:
    if cli_value is not None:
        return expand_path(cli_value)

    apply_local_path_env(env_path=env_path)
    value = os.environ.get(env_name)
    if value:
        return expand_path(value)

    if required:
        raise FileNotFoundError(
            f"Required path {env_name} is not configured. "
            f"Set it in the environment or edit {LOCAL_PATHS_ENV.name}."
        )
    return None


def resolve_scene_root(
    cli_value: str | os.PathLike[str] | None = None,
    *,
    env_path: Path | None = None,
    required: bool = True,
) -> Path | None:
    """Resolve the shared scene root from CLI or local config."""
    return _resolve_explicit_path(
        cli_value,
        SCENE_ROOT_ENV,
        env_path=env_path,
        required=required,
    )


def resolve_bench_dataset_root(
    cli_value: str | os.PathLike[str] | None = None,
    *,
    env_path: Path | None = None,
    required: bool = True,
) -> Path | None:
    """Resolve the default benchmark dataset root from CLI or local config."""
    return _resolve_explicit_path(
        cli_value,
        BENCH_DATASET_ROOT_ENV,
        env_path=env_path,
        required=required,
    )


def _is_valid_isaaclab_source(path: Path) -> bool:
    return path.is_dir() and (path / "omni.isaac.matterport").is_dir()


def candidate_isaaclab_sources() -> list[Path]:
    """Return likely IsaacLab/source locations on the local machine."""
    home_candidate = Path.home() / "IsaacLab" / "source"
    repo_neighbor_candidate = REPO_ROOT.parents[1] / "IsaacLab" / "source"

    seen: set[Path] = set()
    candidates: list[Path] = []
    for candidate in (home_candidate, repo_neighbor_candidate):
        candidate = candidate.resolve()
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def detect_isaaclab_source(candidates: Iterable[Path] | None = None) -> Path | None:
    """Find IsaacLab/source automatically from known locations."""
    for candidate in candidates or candidate_isaaclab_sources():
        if _is_valid_isaaclab_source(candidate):
            return candidate
    return None


def resolve_isaaclab_source(
    cli_value: str | os.PathLike[str] | None = None,
    *,
    env_path: Path | None = None,
    candidates: Iterable[Path] | None = None,
    required: bool = True,
) -> Path | None:
    """Resolve IsaacLab/source with env override, auto-detect, then local-file fallback."""
    if cli_value is not None:
        resolved = expand_path(cli_value)
        if not _is_valid_isaaclab_source(resolved):
            raise FileNotFoundError(f"Invalid IsaacLab source path: {resolved}")
        return resolved

    explicit_env = os.environ.get(ISAACLAB_SOURCE_ENV)
    if explicit_env:
        resolved = expand_path(explicit_env)
        if not _is_valid_isaaclab_source(resolved):
            raise FileNotFoundError(f"Invalid IsaacLab source path from {ISAACLAB_SOURCE_ENV}: {resolved}")
        return resolved

    detected = detect_isaaclab_source(candidates=candidates)
    if detected is not None:
        os.environ.setdefault(ISAACLAB_SOURCE_ENV, str(detected))
        return detected

    local_values = load_local_path_values(env_path)
    fallback = local_values.get(ISAACLAB_SOURCE_ENV)
    if fallback:
        resolved = expand_path(fallback)
        if not _is_valid_isaaclab_source(resolved):
            raise FileNotFoundError(
                f"Invalid IsaacLab source path in {LOCAL_PATHS_ENV.name}: {resolved}"
            )
        os.environ.setdefault(ISAACLAB_SOURCE_ENV, str(resolved))
        return resolved

    if required:
        raise FileNotFoundError(
            "Could not locate IsaacLab/source with omni.isaac.matterport. "
            f"Set {ISAACLAB_SOURCE_ENV} in the environment or edit {LOCAL_PATHS_ENV.name}."
        )
    return None


def _kit_escape(path: str) -> str:
    return path.replace("\\", "\\\\").replace('"', '\\"')


def resolve_runtime_kit_path(
    *,
    template_path: Path | None = None,
    output_dir: Path | None = None,
    isaaclab_source: Path | None = None,
) -> Path:
    """Render a runtime kit file with the resolved IsaacLab extension folder injected."""
    template = (template_path or KIT_TEMPLATE).resolve()
    if not template.is_file():
        raise FileNotFoundError(f"Kit template not found: {template}")

    source_path = isaaclab_source or resolve_isaaclab_source()
    if source_path is None:
        raise FileNotFoundError("IsaacLab source path is required to render runtime kit.")

    template_text = template.read_text(encoding="utf-8")
    if KIT_DYNAMIC_MARKER not in template_text:
        raise ValueError(f"Kit template is missing dynamic marker: {KIT_DYNAMIC_MARKER}")

    injected_line = f'    "{_kit_escape(str(source_path))}",'
    rendered = template_text.replace(KIT_DYNAMIC_MARKER, injected_line)

    digest = hashlib.sha1(
        f"{template}:{template.stat().st_mtime_ns}:{source_path}".encode("utf-8")
    ).hexdigest()[:12]
    cache_dir = output_dir or Path(tempfile.gettempdir()) / "uninav_runtime_kits"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rendered_path = cache_dir / f"uninav_python_{digest}.kit"

    if not rendered_path.exists() or rendered_path.read_text(encoding="utf-8") != rendered:
        rendered_path.write_text(rendered, encoding="utf-8")

    return rendered_path
