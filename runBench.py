#!/usr/bin/env python3
"""
OmniNavBench Evaluation Runner - Example Script

Usage:
    python runBench.py --config path/to/config.yaml --envset path/to/scenarios.json --output results/

Example:
    python runBench.py \
        --config configs/base_config.yaml \
        --envset envsets/grscenes_test.json \
        --output results/my_policy/ \
        --headless
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

from bench import BenchRunner, BenchConfig, BasePolicy, Observation, Action
from bench.evaluator.bench_runner import BenchResult
from bench.evaluator.episode_runner import EpisodeResult

TEST_MODE_ENV = "OMNINAV_BENCH_TEST_MODE"


def _path_from_env(env_var: str) -> Path | None:
    val = os.environ.get(env_var)
    return Path(val) if val else None


def _instantiate_optional_policy(policy_name: str, module_name: str, class_name: str, **kwargs) -> BasePolicy:
    """Import optional policy modules lazily so unavailable integrations do not break the CLI."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Policy '{policy_name}' is unavailable because optional module '{exc.name}' "
            f"could not be imported. Install or restore the dependencies for {module_name} "
            f"before using --policy {policy_name}."
        ) from exc

    policy_cls = getattr(module, class_name)
    return policy_cls(**kwargs)


# ==============================================================================
# Baseline Policy Implementation
# ==============================================================================


class ForwardPolicy(BasePolicy):
    """Policy that always goes forward - for baseline testing."""

    def __init__(self, speed: float = 0.5):
        super().__init__()
        self._speed = speed

    def act(self, observation: Observation) -> Action:
        return Action(linear_velocity=self._speed)

# ==============================================================================
# Main Entry Point
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OmniNavBench evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=Path, required=True, help="OmniNav YAML config")
    parser.add_argument("--envset", type=Path, help="Envset JSON file or directory. Optional when --omninavbench is set (auto-resolved).")
    parser.add_argument("--output", type=Path, default=Path("results/"), help="Output directory")

    # OmniNavBench shortcut: auto-resolves --envset from data root + mode/style/robot
    parser.add_argument(
        "--omninavbench",
        action="store_true",
        help="Auto-resolve envset from the OmniNavBench data tree using --mode/--robot/--style.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        help="OmniNavBench split. Required with --omninavbench. "
             "Resolves to annotations/<mode>/<style>/<robot>/ under --omninavbench-root. "
             "train has GT for local offline scoring; test is sanitized for leaderboard submission.",
    )
    parser.add_argument(
        "--omninavbench-root",
        type=Path,
        default=_path_from_env("OMNINAV_BENCH_DATASET_ROOT"),
        help="OmniNavBench dataset root. Defaults to $OMNINAV_BENCH_DATASET_ROOT "
             "(set in local_paths.env).",
    )
    parser.add_argument(
        "--robot",
        choices=["h1", "aliengo", "carter"],
        default="h1",
        help="Robot embodiment to evaluate (h1 humanoid / aliengo quadruped / carter wheeled). "
             "Used with --omninavbench. Make sure --config matches this robot.",
    )
    parser.add_argument("--style", choices=["original", "concise", "verbose", "first_person"], default="original",
                        help="Instruction style (used with --omninavbench).")
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=_path_from_env("OMNINAV_SCENE_ROOT"),
        help="Base directory for envset file paths (e.g., Matterport3D + GRScenes assets). "
             "Defaults to $OMNINAV_SCENE_ROOT (set in local_paths.env).",
    )
    parser.add_argument("--scenario", action="append", dest="scenarios", help="Scenario ID to run (repeatable).")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--timeout-multiplier", type=float, default=2.0,
                       help="Timeout multiplier based on expert path time/frames (default: 2.0)")
    parser.add_argument("--success-threshold", type=float, default=2.0, help="Success distance (meters)")
    parser.add_argument("--policy", choices=["forward", "uninavid", "uninavid_waypoint", "uninavid_waypoint_points", "mtu3d", "poliformer", "navila", "omninav"], default="forward")
    parser.add_argument("--no-trajectory", action="store_true", help="Disable trajectory recording")
    parser.add_argument("--no-sort", action="store_true", help="Disable sorting scenarios by scene")
    parser.add_argument("--no-save-per-episode", action="store_true", help="Disable saving per-episode results")
    parser.add_argument("--no-skip", action="store_true", help="Disable skipping completed scenarios (force re-run all).")
    parser.add_argument("--isolate-episodes", action="store_true", help="Run each episode in a separate process (prevents PhysX deadlocks)")

    # Uni-NaVid HTTP policy parameters
    parser.add_argument("--uninavid-server-url", type=str, default=None,
                       help="Uni-NaVid HTTP server URL, e.g. http://localhost:<port>. Required when --policy uninavid.")
    parser.add_argument("--forward-speed", type=float, default=1.0,
                       help="Forward speed for uninavid HTTP policy (m/s)")
    parser.add_argument("--turn-angular-velocity", type=float, default=2.0,
                       help="Angular velocity for uninavid HTTP policy (rad/s)")

    # MTU3D HTTP policy parameters
    parser.add_argument("--mtu3d-server-url", type=str, default=None,
                       help="MTU3D HTTP server URL, e.g. http://localhost:<port>. Required when --policy mtu3d.")
    parser.add_argument("--mtu3d-spin-steps", type=int, default=12,
                       help="Number of rotate goals to complete a 360-degree scan")

    # PoliFormer HTTP policy parameters
    parser.add_argument("--poliformer-server-url", type=str, default=None,
                       help="PoliFormer HTTP server URL, e.g. http://localhost:<port>. Required when --policy poliformer.")

    # NaVILA HTTP policy parameters
    parser.add_argument("--navila-server-url", type=str, default=None,
                       help="NaVILA HTTP server URL, e.g. http://localhost:<port>. Required when --policy navila.")

    # OmniNav HTTP policy parameters
    parser.add_argument("--omninav-server-url", type=str, default=None,
                       help="OmniNav HTTP server URL, e.g. http://localhost:<port>. Required when --policy omninav.")

    # Uni-NaVid Waypoint HTTP policy parameters
    parser.add_argument("--uninavid-waypoint-server-url", type=str, default=None,
                        help="Uni-NaVid Waypoint HTTP server URL, e.g. http://localhost:<port>. Required when --policy uninavid_waypoint.")
    parser.add_argument("--uninavid-waypoint-wall-timeout-s", type=float, default=300.0,
                        help="Wall-clock timeout for uninavid_waypoint policy/server session")

    # Uni-NaVid Waypoint Points HTTP policy parameters
    parser.add_argument("--uninavid-waypoint-points-server-url", type=str, default=None,
                        help="Uni-NaVid Waypoint Points HTTP server URL, e.g. http://localhost:<port>. Required when --policy uninavid_waypoint_points.")
    parser.add_argument("--uninavid-waypoint-points-wall-timeout-s", type=float, default=300.0,
                        help="Wall-clock timeout for uninavid_waypoint_points policy/server session")

    # Video recording parameters
    parser.add_argument("--record-video", action="store_true",
                       help="Record RGB video during evaluation")
    parser.add_argument("--video-fps", type=int, default=30,
                       help="Video frame rate (default: 30)")
    parser.add_argument("--no-depth-video", action="store_true",
                       help="Disable depth video recording")
    parser.add_argument("--record-images", action="store_true",
                       help="Record RGB image frames during evaluation")
    parser.add_argument("--image-interval-s", type=float, default=1.0,
                       help="Sim-time interval between saved RGB frames (default: 1.0s)")

    return parser.parse_args()


def create_policy(policy_name: str) -> BasePolicy:
    """Create policy instance by name."""
    policies = {
        "forward": ForwardPolicy,
    }
    if policy_name not in policies:
        raise ValueError(f"Unknown policy: {policy_name}")
    return policies[policy_name]()

def create_uninavid_policy(
    policy_name: str,
    uninavid_server_url: str = "http://localhost:8000",
    forward_speed: float = 0.25,
    turn_angular_velocity: float = 0.5,
) -> BasePolicy:
    """
    Create policy; reuse existing map and only intercept uninavid for custom wiring.
    """
    if policy_name == "uninavid":
        return _instantiate_optional_policy(
            policy_name,
            "bench.policy.uninavid.uninavid_http_policy",
            "UniNaVidHTTPPolicy",
            server_url=uninavid_server_url,
            forward_speed=forward_speed,
            turn_angular_velocity=turn_angular_velocity,
        )
    return create_policy(policy_name)

def create_mtu3d_policy(
    policy_name: str,
    mtu3d_server_url: str = "http://localhost:8010",
    spin_steps: int = 12,
) -> BasePolicy:
    if policy_name == "mtu3d":
        return _instantiate_optional_policy(
            policy_name,
            "bench.policy.mtu3d.mtu3d_http_policy",
            "MTU3DHTTPPolicy",
            server_url=mtu3d_server_url,
            waypoint_threshold_m=0.1,
            spin_steps=int(spin_steps),
        )
    return create_policy(policy_name)

def create_poliformer_policy(
    policy_name: str,
    poliformer_server_url: str = "http://localhost:8030",
) -> BasePolicy:
    if policy_name == "poliformer":
        return _instantiate_optional_policy(
            policy_name,
            "bench.policy.poliformer.poliformer_http_policy",
            "PoliFormerHTTPPolicy",
            server_url=poliformer_server_url,
        )
    return create_policy(policy_name)

def create_navila_policy(
    policy_name: str,
    navila_server_url: str = "http://localhost:8050",
) -> BasePolicy:
    if policy_name == "navila":
        return _instantiate_optional_policy(
            policy_name,
            "bench.policy.navila.navila_http_policy",
            "NaVILAHTTPPolicy",
            server_url=navila_server_url,
        )
    return create_policy(policy_name)

def create_omninav_policy(
    policy_name: str,
    omninav_server_url: str = "http://localhost:8005",
) -> BasePolicy:
    if policy_name == "omninav":
        return _instantiate_optional_policy(
            policy_name,
            "bench.policy.omninav.omninav_http_policy",
            "OmniNavHTTPPolicy",
            server_url=omninav_server_url,
        )
    return create_policy(policy_name)

def create_uninavid_waypoint_policy(
    policy_name: str,
    uninavid_waypoint_server_url: str = "http://localhost:8001",
    uninavid_waypoint_wall_timeout_s: float | None = 300.0,
) -> BasePolicy:
    """Create Uni-NaVid Waypoint HTTP policy."""
    if policy_name == "uninavid_waypoint":
        return _instantiate_optional_policy(
            policy_name,
            "bench.policy.uninavid_waypoint.uninavid_waypoint_http_policy_action",
            "UniNaVidWaypointHTTPPolicy",
            server_url=uninavid_waypoint_server_url,
            wall_timeout_s=uninavid_waypoint_wall_timeout_s,
        )
    return create_policy(policy_name)

def create_uninavid_waypoint_points_policy(
    policy_name: str,
    uninavid_waypoint_points_server_url: str = "http://localhost:8002",
    uninavid_waypoint_points_wall_timeout_s: float | None = 300.0,
) -> BasePolicy:
    """Create Uni-NaVid Waypoint Points HTTP policy (go_toward_point controller)."""
    if policy_name == "uninavid_waypoint_points":
        return _instantiate_optional_policy(
            policy_name,
            "bench.policy.uninavid_waypoint.uninavid_waypoint_http_policy_points",
            "UniNaVidWaypointPointsHTTPPolicy",
            server_url=uninavid_waypoint_points_server_url,
            wall_timeout_s=uninavid_waypoint_points_wall_timeout_s,
        )
    return create_policy(policy_name)

_ROBOT_TO_DATASET_DIR = {"h1": "human", "aliengo": "dog", "carter": "car"}


def _resolve_omninavbench_envset(args: argparse.Namespace) -> Path:
    """Resolve the OmniNavBench scenarios directory; BenchRunner reads it as
    an envset directory directly (no intermediate file is generated).

    Layout matches the HuggingFace release:
      <root>/annotations/<mode>/<style>/<robot>/<scene>/<episode>.json
    """
    if args.mode is None:
        raise SystemExit("--omninavbench requires --mode {train,test}")
    if args.omninavbench_root is None:
        raise SystemExit(
            "--omninavbench needs the dataset root. Set OMNINAV_BENCH_DATASET_ROOT "
            "in local_paths.env (then `source load_local_paths.sh`), or pass "
            "--omninavbench-root /path/to/OmniNavBench on the command line."
        )

    dataset_dir = _ROBOT_TO_DATASET_DIR[args.robot]
    envset_dir = args.omninavbench_root / "annotations" / args.mode / args.style / dataset_dir

    if not envset_dir.exists():
        raise FileNotFoundError(
            f"OmniNavBench scenario directory not found: {envset_dir}. "
            f"Expected layout under --omninavbench-root: annotations/<mode>/<style>/<robot>/. "
            f"Check --omninavbench-root and that --mode/--style/--robot match the dataset layout."
        )

    print(f"[runBench/OmniNavBench] mode={args.mode} robot={args.robot} style={args.style}")
    print(f"[runBench/OmniNavBench] envset dir: {envset_dir}")
    if args.mode == "test" and not args.record_video:
        print("[runBench/OmniNavBench] test mode: video recording is OFF by default; pass --record-video to override.")
    return envset_dir


def main():
    args = parse_args()

    if args.omninavbench:
        args.envset = _resolve_omninavbench_envset(args)
    elif args.envset is None:
        raise SystemExit("--envset is required (or use --omninavbench --mode {train,test}).")

    _POLICY_SERVER_URL_ATTR = {
        "uninavid": "uninavid_server_url",
        "mtu3d": "mtu3d_server_url",
        "poliformer": "poliformer_server_url",
        "navila": "navila_server_url",
        "omninav": "omninav_server_url",
        "uninavid_waypoint": "uninavid_waypoint_server_url",
        "uninavid_waypoint_points": "uninavid_waypoint_points_server_url",
    }
    if args.policy in _POLICY_SERVER_URL_ATTR:
        attr = _POLICY_SERVER_URL_ATTR[args.policy]
        if not getattr(args, attr, None):
            flag = "--" + attr.replace("_", "-")
            raise SystemExit(
                f"--policy {args.policy} requires the {flag} server-url flag "
                f"(e.g. {flag} http://localhost:<port>); no default is provided."
            )

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not args.envset.exists():
        raise FileNotFoundError(f"Envset not found: {args.envset}")

    args.output.mkdir(parents=True, exist_ok=True)

    # Create policy
    print(f"[runBench] Policy: {args.policy}")
    match args.policy:
        case "uninavid":
            policy = create_uninavid_policy(
                args.policy,
                args.uninavid_server_url,
                args.forward_speed,
                args.turn_angular_velocity
            )
        case "mtu3d":
            policy = create_mtu3d_policy(
                args.policy,
                args.mtu3d_server_url,
                args.mtu3d_spin_steps,
            )
        case "poliformer":
            policy = create_poliformer_policy(
                args.policy,
                args.poliformer_server_url,
            )
        case "navila":
            policy = create_navila_policy(
                args.policy,
                args.navila_server_url,
            )
        case "omninav":
            policy = create_omninav_policy(
                args.policy,
                args.omninav_server_url,
            )
        case "uninavid_waypoint":
            policy = create_uninavid_waypoint_policy(
                args.policy,
                args.uninavid_waypoint_server_url,
                args.uninavid_waypoint_wall_timeout_s,
            )
        case "uninavid_waypoint_points":
            policy = create_uninavid_waypoint_points_policy(
                args.policy,
                args.uninavid_waypoint_points_server_url,
                args.uninavid_waypoint_points_wall_timeout_s,
            )
        case _:
            policy = create_policy(args.policy)

    # Build policy-specific arguments for subprocess mode
    policy_args = {
        "uninavid_server_url": args.uninavid_server_url,
        "forward_speed": args.forward_speed,
        "turn_angular_velocity": args.turn_angular_velocity,
        "mtu3d_server_url": args.mtu3d_server_url,
        "mtu3d_spin_steps": args.mtu3d_spin_steps,
        "poliformer_server_url": args.poliformer_server_url,
        "navila_server_url": args.navila_server_url,
        "omninav_server_url": args.omninav_server_url,
        "uninavid_waypoint_server_url": args.uninavid_waypoint_server_url,
        "uninavid_waypoint_wall_timeout_s": args.uninavid_waypoint_wall_timeout_s,
        "uninavid_waypoint_points_server_url": args.uninavid_waypoint_points_server_url,
        "uninavid_waypoint_points_wall_timeout_s": args.uninavid_waypoint_points_wall_timeout_s,
    }

    # Create benchmark config
    config = BenchConfig(
        uninav_config=args.config.resolve(),
        envset_path=args.envset.resolve(),
        output_dir=args.output.resolve(),
        scene_root=args.scene_root,
        scenario_ids=args.scenarios,
        headless=args.headless,
        timeout_multiplier=args.timeout_multiplier,
        success_threshold=args.success_threshold,
        record_trajectory=not args.no_trajectory,
        save_per_episode=not args.no_save_per_episode,
        sort_by_scene=not args.no_sort,
        skip_completed=not args.no_skip,
        record_video=args.record_video,
        video_fps=args.video_fps,
        save_depth_video=not args.no_depth_video,
        record_images=args.record_images,
        image_interval_s=args.image_interval_s,
        policy_name=args.policy,
        policy_args=policy_args,
        isolate_episodes=args.isolate_episodes,
    )

    print(f"[runBench] Config: {args.config}")
    print(f"[runBench] Envset: {args.envset}")
    print(f"[runBench] Output: {args.output}")

    # Run benchmark (optionally using stub runner for tests)
    results = _execute_benchmark(config, policy)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"{'Scenario':<40} {'Steps':^8} {'Time (s)':^10}")
    print("-" * 70)
    for r in results.results:
        print(f"{r.scenario_id:<40} {r.steps:^8} {r.time_s:^10.2f}")
    print("-" * 70)
    print(f"  Episodes:   {len(results.results)}")
    print(f"  Avg Steps:  {results.avg_steps:.1f}")
    print(f"  Avg Time:   {results.avg_time_s:.2f}s")
    print(f"  Total Time: {results.total_time_s:.1f}s")
    print(f"\nResults saved to: {args.output}")
    print("Scoring is offline: see bench/evaluator/offline_test.py")


def _execute_benchmark(config: BenchConfig, policy: BasePolicy) -> BenchResult:
    """Run benchmark or stubbed version for tests."""
    if os.getenv(TEST_MODE_ENV):
        print("[runBench] Test mode enabled, using stub runner")
        return _run_stub_bench(config)
    runner = BenchRunner(config, policy)
    return runner.run()


def _run_stub_bench(config: BenchConfig) -> BenchResult:
    """Create deterministic outputs for CLI tests without launching Isaac."""
    scenario_id = _get_first_scenario_id(config.envset_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    episode = EpisodeResult(
        scenario_id=scenario_id,
        success=True,
        termination_reason="stub-success",
        steps=1,
        time_s=0.1,
        distance_to_goal=0.0,
        path_length=0.0,
        trajectory=[],
        metrics={"shortest_path": 0.0},
    )
    result = BenchResult(
        results=[episode],
        avg_steps=episode.steps,
        avg_time_s=episode.time_s,
        total_time_s=episode.time_s,
    )
    _write_stub_outputs(config.output_dir, episode, result)
    return result


def _get_first_scenario_id(envset_path: Path) -> str:
    """Load first scenario id from envset for naming outputs."""
    try:
        with envset_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        scenarios = data.get("scenarios", [])
        if scenarios:
            return scenarios[0].get("id", "stub_scenario")
    except Exception:
        pass
    return "stub_scenario"


def _write_stub_outputs(output_dir: Path, episode: EpisodeResult, result: BenchResult):
    """Persist stub summary and per-episode JSON artifacts."""
    episode_file = output_dir / f"{episode.scenario_id}.json"
    summary_file = output_dir / "summary.json"

    episode_payload = {
        "scenario_id": episode.scenario_id,
        "termination_reason": episode.termination_reason,
        "steps": episode.steps,
        "time_s": episode.time_s,
        "path_length": episode.path_length,
        "trajectory": episode.trajectory,
    }
    if episode.extra:
        episode_payload.update(episode.extra)
    summary_payload = {
        "avg_steps": result.avg_steps,
        "avg_time_s": result.avg_time_s,
        "total_time_s": result.total_time_s,
        "num_episodes": len(result.results),
    }

    episode_file.write_text(json.dumps(episode_payload, indent=2), encoding="utf-8")
    summary_file.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
