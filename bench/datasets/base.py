"""Abstract base class and robot defaults for dataset adapters."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union

from .schema import UnifiedEpisode


# ---------------------------------------------------------------------------
# Robot / navmesh defaults
# ---------------------------------------------------------------------------

@dataclass
class RobotDefaults:
    """Default robot and navmesh configuration for envset generation.

    These values populate the ``robots`` and ``navmesh`` blocks when an
    adapter converts a UnifiedEpisode to an envset scenario dict.

    Attributes:
        robot_type: Isaac Sim robot type string (e.g. "carter_v1").
        robot_label: Human-readable label; defaults to robot_type if None.
        spawn_path: USD prim path where the robot is instantiated.
        usd_path: Relative path to the robot USD asset.
        control_mode: Controller mode string (e.g. "move_by_speed").
        base_velocity: Linear velocity in m/s.
        base_turn_rate: Angular velocity in rad/s.
        navmesh_z_padding: Vertical padding for navmesh baking.
        navmesh_agent_radius: Agent radius for navmesh baking.
        navmesh_max_step_height: Max climbable step height; None omits field.
        navmesh_min_include_volume_size: Min volume size dict; None omits field.
        navmesh_spawn_min_separation_m: Min spawn separation; None omits field.
    """

    robot_type: str = "carter_v1"
    robot_label: Optional[str] = None
    spawn_path: str = "/carter_v1"
    usd_path: str = "robots/Carter/carter_v1.usd"
    control_mode: str = "move_by_speed"
    base_velocity: float = 0.6
    base_turn_rate: float = 1.2

    # Navmesh
    navmesh_bake_root_prim_path: str = "/World"
    navmesh_z_padding: float = 2.0
    navmesh_agent_radius: float = 0.5
    navmesh_max_step_height: Optional[float] = None
    navmesh_min_include_volume_size: Optional[Dict[str, float]] = None
    navmesh_spawn_min_separation_m: Optional[float] = None


# ---------------------------------------------------------------------------
# Abstract adapter
# ---------------------------------------------------------------------------

class DatasetAdapter(ABC):
    """Base class for all dataset adapters.

    Subclasses must declare a class-level ``name`` attribute (used for
    registration) and implement :meth:`load`.

    Example::

        @register_adapter
        class MyAdapter(DatasetAdapter):
            name = "my_dataset"

            def load(self, data_path, split="train", **kwargs):
                ...
                return [UnifiedEpisode(...), ...]
    """

    name: ClassVar[str]

    @abstractmethod
    def load(
        self,
        data_path: Path,
        split: str = "train",
        **kwargs: Any,
    ) -> List[UnifiedEpisode]:
        """Load episodes from *data_path* for the given *split*.

        Args:
            data_path: Root directory (or file) of the dataset.
            split: Dataset split identifier (e.g. "train", "test").
            **kwargs: Adapter-specific keyword arguments.

        Returns:
            List of UnifiedEpisode objects.
        """

    # ------------------------------------------------------------------
    # Shared helper
    # ------------------------------------------------------------------

    def save_envset(
        self,
        episodes: List[UnifiedEpisode],
        output_path: Path,
        robot_defaults: Optional[RobotDefaults] = None,
    ) -> Path:
        """Serialise *episodes* to an envset JSON file.

        Args:
            episodes: List of unified episodes to serialise.
            output_path: Destination file path.
            robot_defaults: Robot/navmesh configuration.  Defaults to
                :class:`RobotDefaults` with no arguments.

        Returns:
            The resolved ``output_path``.
        """
        defaults = robot_defaults or RobotDefaults()
        scenarios = [ep.to_envset_scenario(defaults) for ep in episodes]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump({"scenarios": scenarios}, fh, indent=2, ensure_ascii=False)
        print(f"[DatasetAdapter] Saved {len(scenarios)} scenarios → {output_path}")
        return output_path
