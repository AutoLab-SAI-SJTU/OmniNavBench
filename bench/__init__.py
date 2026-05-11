"""
OmniNavBench Evaluation Module.

Provides batch benchmark evaluation for VLN (Vision-Language Navigation) tasks.

Usage:
    from bench import BenchRunner, BenchConfig
    from bench.policy import BasePolicy, Action

    class MyPolicy(BasePolicy):
        def act(self, observation):
            return Action(linear_velocity=0.5)

    config = BenchConfig(
        uninav_config=Path("config.yaml"),
        envset_path=Path("scenarios.json"),
        output_dir=Path("results/"),
    )
    runner = BenchRunner(config, MyPolicy())
    results = runner.run()
"""

from .evaluator.bench_runner import BenchRunner, BenchConfig
from .evaluator.episode_runner import EpisodeRunner, EpisodeConfig, EpisodeResult
from .evaluator.termination import (
    TerminationCondition,
    GoalReachedCondition,
    TimeoutCondition,
    StopActionCondition,
    CompositeCondition,
)
from .policy.base import BasePolicy, Observation, Action
from .metrics.navigation import (
    NavigationMetrics,
    compute_success_rate,
    compute_spl,
    compute_navigation_error,
    compute_all_metrics,
)

__all__ = [
    # Core runners
    "BenchRunner",
    "BenchConfig",
    "EpisodeRunner",
    "EpisodeConfig",
    "EpisodeResult",
    # Policy interface
    "BasePolicy",
    "Observation",
    "Action",
    # Termination conditions
    "TerminationCondition",
    "GoalReachedCondition",
    "TimeoutCondition",
    "StopActionCondition",
    "CompositeCondition",
    # Metrics
    "NavigationMetrics",
    "compute_success_rate",
    "compute_spl",
    "compute_navigation_error",
    "compute_all_metrics",
]
