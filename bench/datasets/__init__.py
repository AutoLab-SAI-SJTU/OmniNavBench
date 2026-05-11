"""Unified dataset schema and adapter framework for OmniNavBench.

This package provides a common interface for loading any benchmark or dataset
into the envset JSON format consumed by :class:`~bench.evaluator.bench_runner.BenchRunner`.

Quick start
-----------
::

    from bench.datasets import DatasetLoader, RobotDefaults, list_adapters

    # See all registered adapters
    print(list_adapters())   # ['native', 'sage3d', 'vlnce']

    # --- OmniNavBenchData (already envset format) ---
    loader = DatasetLoader.from_name("native")
    envset_path = loader.convert(
        data_path="/path/to/OmniNavBenchData",
        output_path="./envsets/omninav_train_human.json",
        split="train",
        category="human",
        style="original",
    )

    # --- SAGE-3D VLN ---
    loader = DatasetLoader.from_name("sage3d")
    envset_path = loader.convert(
        data_path="/path/to/SAGE-3D_VLN_Data",
        output_path="./envsets/sage3d_train.json",
        split="train",
        scene_usd_root="/path/to/Sage-3D-usda",
        instruction_type="Add_Causality",
        max_episodes=100,
    )

    # --- VLN-CE (R2R VLN-CE preprocessed) ---
    loader = DatasetLoader.from_name("vlnce")
    envset_path = loader.convert(
        data_path="/path/to/VLN-CE",
        output_path="./envsets/vlnce_val_seen.json",
        split="val_seen",
        max_episodes=100,
    )

    # Pass envset_path directly to BenchRunner
    from bench.evaluator.bench_runner import BenchRunner, BenchConfig
    runner = BenchRunner(BenchConfig(envset_path=envset_path, ...), policy)
    runner.run()

Extending
---------
To add a new adapter, create a file under ``bench/datasets/adapters/`` and
register it::

    from bench.datasets.registry import register_adapter
    from bench.datasets.base import DatasetAdapter
    from bench.datasets.schema import UnifiedEpisode

    @register_adapter
    class MyAdapter(DatasetAdapter):
        name = "my_dataset"

        def load(self, data_path, split="train", **kwargs):
            ...
            return [UnifiedEpisode(...)]

Then add an import in ``bench/datasets/adapters/__init__.py``.
"""

from .base import DatasetAdapter, RobotDefaults
from .loader import DatasetLoader
from .registry import get_adapter, list_adapters, register_adapter
from .schema import SubtaskSpec, UnifiedEpisode

__all__ = [
    # Public API
    "DatasetLoader",
    "DatasetAdapter",
    "RobotDefaults",
    "UnifiedEpisode",
    "SubtaskSpec",
    # Registry helpers
    "register_adapter",
    "get_adapter",
    "list_adapters",
]
