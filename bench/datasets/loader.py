"""High-level DatasetLoader: the single entry-point for end users.

Usage::

    from bench.datasets import DatasetLoader, RobotDefaults

    # Load episodes as UnifiedEpisode objects
    loader = DatasetLoader.from_name("native")
    episodes = loader.load("/path/to/OmniNavBenchData", split="train", category="human")

    # Or convert directly to an envset JSON ready for BenchRunner
    envset_path = loader.convert(
        data_path="/path/to/OmniNavBenchData",
        output_path="./envsets/omninav_train.json",
        split="train",
        category="human",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

# Importing adapters triggers self-registration via @register_adapter.
from . import adapters  # noqa: F401 – side-effect import
from .base import DatasetAdapter, RobotDefaults
from .registry import get_adapter, list_adapters
from .schema import UnifiedEpisode


class DatasetLoader:
    """Thin facade around a :class:`~bench.datasets.base.DatasetAdapter`.

    Args:
        adapter_name: Registered adapter name (e.g. ``"native"``, ``"sage3d"``).
    """

    def __init__(self, adapter_name: str) -> None:
        self._adapter: DatasetAdapter = get_adapter(adapter_name)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_name(cls, name: str) -> "DatasetLoader":
        """Create a DatasetLoader for the named adapter.

        Args:
            name: Adapter name as registered via :func:`~bench.datasets.registry.register_adapter`.

        Returns:
            A :class:`DatasetLoader` wrapping the corresponding adapter.
        """
        return cls(name)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def load(
        self,
        data_path: Union[str, Path],
        split: str = "train",
        **kwargs: Any,
    ) -> List[UnifiedEpisode]:
        """Load episodes from *data_path*.

        Args:
            data_path: Root directory of the dataset.
            split: Dataset split identifier (e.g. "train", "test").
            **kwargs: Forwarded to the underlying adapter's ``load()`` method.

        Returns:
            List of :class:`~bench.datasets.schema.UnifiedEpisode` objects.
        """
        return self._adapter.load(Path(data_path), split=split, **kwargs)

    def convert(
        self,
        data_path: Union[str, Path],
        output_path: Union[str, Path],
        split: str = "train",
        robot_defaults: Optional[RobotDefaults] = None,
        **kwargs: Any,
    ) -> Path:
        """Load → convert → write envset JSON in one call.

        Args:
            data_path: Root directory of the source dataset.
            output_path: Destination path for the envset JSON file.
            split: Dataset split identifier.
            robot_defaults: Robot/navmesh configuration for envset generation.
                Defaults to :class:`~bench.datasets.base.RobotDefaults()`.
            **kwargs: Forwarded to the underlying adapter's ``load()`` method.

        Returns:
            Resolved ``output_path`` as a :class:`pathlib.Path`.
        """
        episodes = self.load(data_path, split=split, **kwargs)
        return self._adapter.save_envset(episodes, Path(output_path), robot_defaults)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def adapter(self) -> DatasetAdapter:
        """The underlying adapter instance."""
        return self._adapter

    @staticmethod
    def available() -> List[str]:
        """Return names of all registered adapters."""
        return list_adapters()
