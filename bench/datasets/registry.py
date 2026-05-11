"""Adapter registry.

Adapters self-register via the :func:`register_adapter` decorator::

    from bench.datasets.registry import register_adapter
    from bench.datasets.base import DatasetAdapter

    @register_adapter
    class MyAdapter(DatasetAdapter):
        name = "my_dataset"
        ...

Import :mod:`bench.datasets.adapters` (or :mod:`bench.datasets`) to trigger
registration of all built-in adapters before calling :func:`get_adapter`.
"""

from __future__ import annotations

from typing import Dict, List, Type

from .base import DatasetAdapter

_REGISTRY: Dict[str, Type[DatasetAdapter]] = {}


def register_adapter(cls: Type[DatasetAdapter]) -> Type[DatasetAdapter]:
    """Class decorator that registers *cls* under ``cls.name``.

    Raises:
        AttributeError: If ``cls`` does not have a ``name`` class attribute.
        ValueError: If another adapter is already registered under the same name.
    """
    adapter_name: str = cls.name  # type: ignore[attr-defined]
    if adapter_name in _REGISTRY:
        raise ValueError(
            f"Adapter '{adapter_name}' is already registered "
            f"(existing: {_REGISTRY[adapter_name].__qualname__}, "
            f"new: {cls.__qualname__})."
        )
    _REGISTRY[adapter_name] = cls
    return cls


def get_adapter(name: str) -> DatasetAdapter:
    """Instantiate and return the adapter registered under *name*.

    Args:
        name: The adapter name (e.g. ``"native"``, ``"sage3d"``).

    Raises:
        KeyError: If no adapter is registered under *name*.

    Returns:
        A fresh adapter instance.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(
            f"Unknown adapter '{name}'. Available adapters: {available}. "
            f"Make sure 'bench.datasets' has been imported to trigger registration."
        )
    return _REGISTRY[name]()


def list_adapters() -> List[str]:
    """Return the names of all registered adapters (sorted)."""
    return sorted(_REGISTRY.keys())
