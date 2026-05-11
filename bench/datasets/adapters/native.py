"""NativeAdapter — pass-through adapter for datasets already in envset format.

Supports OmniNavBenchData whose episodes are native envset JSON files
under the layout::

    {data_root}/{split}/{style}/{category}/{scene_id}/final_episode_N.json

Each JSON has the structure ``{"scenarios": [<envset scenario dict>]}``.
The adapter wraps each scenario in a UnifiedEpisode without any structural
conversion; ``to_envset_scenario()`` returns the original dict verbatim.

Supported filter parameters
----------------------------
category : str, default "human"
    Robot-embodiment sub-directory.  One of: human (H1 humanoid),
    dog (Aliengo quadruped), car (Carter wheeled).
style : str, default "original"
    Instruction style sub-directory.  One of: original, concise, verbose,
    first_person.
max_episodes : int | None
    Cap the number of episodes loaded (useful for quick smoke tests).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

from ..base import DatasetAdapter, RobotDefaults
from ..registry import register_adapter
from ..schema import SubtaskSpec, UnifiedEpisode


@register_adapter
class NativeAdapter(DatasetAdapter):
    """Pass-through adapter for datasets already in envset JSON format.

    The loaded UnifiedEpisode stores the original scenario dict in
    ``extra["raw_scenario"]``.  When ``to_envset_scenario()`` is called it
    returns that dict unchanged, so no information is lost.
    """

    name = "native"

    def load(
        self,
        data_path: Path,
        split: str = "train",
        *,
        category: str = "human",
        style: str = "original",
        max_episodes: Optional[int] = None,
        **kwargs: Any,
    ) -> List[UnifiedEpisode]:
        """Load episodes from an OmniNavBenchData-layout directory.

        Args:
            data_path: Root directory containing ``train/`` and ``test/``
                sub-directories.
            split: "train" or "test".
            category: Robot-embodiment directory ("human"=H1, "dog"=Aliengo,
                "car"=Carter).
            style: Instruction style ("original", "concise", "verbose",
                "first_person").
            max_episodes: Maximum number of episodes to return.

        Returns:
            List of UnifiedEpisode objects (raw pass-through mode).

        Raises:
            FileNotFoundError: If the resolved split/style/category path does
                not exist.
        """
        search_root = Path(data_path) / split / style / category
        if not search_root.exists():
            raise FileNotFoundError(
                f"[NativeAdapter] Directory not found: {search_root}. "
                f"Check data_path / split / style / category arguments."
            )

        json_files = sorted(search_root.rglob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"[NativeAdapter] No JSON files found under: {search_root}"
            )

        episodes: List[UnifiedEpisode] = []
        for json_path in json_files:
            if max_episodes is not None and len(episodes) >= max_episodes:
                break
            try:
                loaded = self._load_file(json_path)
                episodes.extend(loaded)
            except Exception as exc:
                print(f"[WARN][NativeAdapter] Skipping {json_path}: {exc}")

        print(
            f"[NativeAdapter] Loaded {len(episodes)} episodes "
            f"from {search_root} (split={split}, category={category}, style={style})"
        )
        if max_episodes is not None:
            episodes = episodes[:max_episodes]
        return episodes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_file(self, json_path: Path) -> List[UnifiedEpisode]:
        """Parse one envset JSON file into UnifiedEpisode objects."""
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, dict) or "scenarios" not in data:
            raise ValueError(f"Missing 'scenarios' key in {json_path}")

        episodes = []
        for scenario in data["scenarios"]:
            if not isinstance(scenario, dict):
                print(f"[WARN][NativeAdapter] Non-dict scenario in {json_path}, skipping.")
                continue

            ep = self._scenario_to_episode(scenario, json_path)
            episodes.append(ep)
        return episodes

    def _scenario_to_episode(
        self, scenario: dict, source_path: Path
    ) -> UnifiedEpisode:
        """Wrap a raw envset scenario dict in a UnifiedEpisode (pass-through)."""
        episode_id = str(scenario.get("id", source_path.stem))

        # Extract minimal metadata for UnifiedEpisode fields
        task = scenario.get("task", {}) if isinstance(scenario.get("task"), dict) else {}
        nav = task.get("navigation", {}) if isinstance(task.get("navigation"), dict) else {}
        instruction = str(nav.get("instruction", ""))

        scene = scenario.get("scene", {}) if isinstance(scenario.get("scene"), dict) else {}
        scene_id = str(scene.get("matterport", {}).get("usd_path", episode_id)).split("/")[-2]

        goal_raw = nav.get("goal_position", [0.0, 0.0, 0.0])
        try:
            goal_pos: tuple = (float(goal_raw[0]), float(goal_raw[1]), float(goal_raw[2]))
        except (TypeError, IndexError, ValueError):
            goal_pos = (0.0, 0.0, 0.0)

        return UnifiedEpisode(
            episode_id=episode_id,
            scene_id=scene_id,
            instruction=instruction,
            task_type="vln",
            scene_usd_path=None,  # Not needed; raw_scenario takes priority
            goal_position=goal_pos,
            extra={"raw_scenario": scenario, "source_file": str(source_path)},
        )
