# pyright: reportMissingImports=false

"""Navigation metrics computation for VLN evaluation.

Standard metrics:
- SR (Success Rate): Fraction of episodes where goal was reached
- SPL (Success weighted by Path Length): Efficiency metric
- NE (Navigation Error): Final distance to goal
- OSR (Oracle Success Rate): Success if goal was ever reached during episode
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
import math
import re

try:
    import carb
    import omni.anim.navigation.core as nav
    NAV_AVAILABLE = True
except Exception:  # pragma: no cover - Isaac Sim may be absent outside the sim env
    carb = None
    nav = None
    NAV_AVAILABLE = False

if TYPE_CHECKING:
    from ..evaluator.episode_runner import EpisodeResult, TrajectoryPoint


@dataclass
class NavigationMetrics:
    """Collection of navigation metrics.

    Attributes:
        success_rate: Fraction of successful episodes (SR)
        spl: Success weighted by Path Length (SPL)
        navigation_error: Average final distance to goal (NE)
        oracle_success_rate: Fraction where goal was ever reached (OSR)
        path_length_ratio: Avg ratio of actual/shortest path
        follow_human_success: Average follow human task success rate
        follow_human_success_ratio: Average follow human success ratio
        csr: Average Completion Success Rate
        softsr: Average Soft Success Rate
        sii: Social Intrusion Index (average social violation ratio)
        num_episodes: Total number of episodes evaluated
    """
    success_rate: float
    spl: float
    navigation_error: float
    oracle_success_rate: float
    path_length_ratio: float
    follow_human_success: float
    follow_human_success_ratio: float
    csr: float
    softsr: float
    sii: float
    num_episodes: int


@dataclass(frozen=True)
class _FollowHumanSample:
    position: Tuple[float, float, float]
    frame: Optional[int] = None
    step: Optional[int] = None
    time_s: Optional[float] = None


def compute_success_rate(results: List["EpisodeResult"]) -> float:
    """Compute success rate (SR).

    SR = (# successful episodes) / (# total episodes)

    Args:
        results: List of episode results

    Returns:
        Success rate in [0, 1]
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r.success) / len(results)


def compute_spl(
    results: List["EpisodeResult"],
    shortest_paths: Optional[List[float]] = None,
) -> float:
    """Compute SPL (Success weighted by Path Length).

    SPL = (1/N) * sum(S_i * L_i / max(P_i, L_i))

    where:
    - S_i = 1 if success, 0 otherwise
    - L_i = shortest geodesic path length
    - P_i = actual path length taken

    Args:
        results: List of episode results
        shortest_paths: Precomputed shortest paths (if available)

    Returns:
        SPL in [0, 1]
    """
    if not results:
        return 0.0

    n = len(results)
    spl_sum = 0.0

    for i, r in enumerate(results):
        if not r.success:
            continue

        # Get shortest path (from precomputed or from metrics)
        if shortest_paths and i < len(shortest_paths):
            shortest = shortest_paths[i]
        else:
            shortest = r.metrics.get("shortest_path", r.path_length)

        if shortest <= 0:
            shortest = r.path_length

        # SPL contribution for this episode
        if r.path_length > 0:
            spl_sum += shortest / max(r.path_length, shortest)
        elif shortest > 0:
            spl_sum += 1.0  # Perfect efficiency if no movement needed

    return spl_sum / n


def compute_spl_offline(
    success: bool,
    path_length: float,
    shortest_path: float,
) -> float:
    """Compute SPL for a single episode (offline).

    SPL = L / max(P, L) if success else 0

    Args:
        success: Whether the episode was successful
        path_length: Actual path length taken
        shortest_path: Shortest path length

    Returns:
        SPL value for this episode in [0, 1]
    """
    if not success:
        return 0.0

    if shortest_path <= 0:
        return 0.0

    if path_length > 0:
        return shortest_path / max(path_length, shortest_path)
    elif shortest_path > 0:
        return 1.0  # Perfect efficiency if no movement needed
    else:
        return 0.0


def compute_navigation_error(results: List["EpisodeResult"]) -> float:
    """Compute average navigation error (NE).

    NE = (1/N) * sum(final_distance_to_goal)

    Args:
        results: List of episode results

    Returns:
        Average navigation error in meters
    """
    if not results:
        return 0.0
    return sum(r.distance_to_goal for r in results) / len(results)


def compute_oracle_success(
    results: List["EpisodeResult"],
    success_threshold: float = 1.0,
) -> float:
    """Compute oracle success rate (OSR).

    OSR = fraction of episodes where the agent was ever within
    success_threshold of the goal during the episode.

    Requires trajectory recording.

    Args:
        results: List of episode results
        success_threshold: Distance threshold for success

    Returns:
        Oracle success rate in [0, 1]
    """
    if not results:
        return 0.0

    oracle_successes = 0
    episodes_with_trajectory = 0

    for r in results:
        # Prefer the per-episode oracle_success metric if available.
        # This allows per-episode logic (e.g., "leave then return" for return tasks)
        # without re-accessing goal positions/trajectories here.
        oracle_success_metric = r.metrics.get("oracle_success")
        if isinstance(oracle_success_metric, (int, float)):
            episodes_with_trajectory += 1
            oracle_successes += 1 if float(oracle_success_metric) >= 1.0 else 0
            continue

        if not r.trajectory:
            # Fall back to final success if no trajectory
            if r.success:
                oracle_successes += 1
            episodes_with_trajectory += 1
            continue

        episodes_with_trajectory += 1

        # Check if any point in trajectory was within threshold
        # Note: We need goal_position which is not in EpisodeResult
        # Use the metric if available, otherwise skip
        min_distance = r.metrics.get("min_distance_to_goal")
        if min_distance is not None:
            if min_distance <= success_threshold:
                oracle_successes += 1
        elif r.success:
            oracle_successes += 1

    if episodes_with_trajectory == 0:
        return 0.0
    return oracle_successes / episodes_with_trajectory


def compute_path_length_ratio(
    results: List["EpisodeResult"],
    shortest_paths: Optional[List[float]] = None,
) -> float:
    """Compute average path length ratio.

    Ratio = actual_path_length / shortest_path_length

    Args:
        results: List of episode results
        shortest_paths: Precomputed shortest paths

    Returns:
        Average path length ratio (>= 1.0 for suboptimal paths)
    """
    if not results:
        return 0.0

    ratios = []
    for i, r in enumerate(results):
        if shortest_paths and i < len(shortest_paths):
            shortest = shortest_paths[i]
        else:
            shortest = r.metrics.get("shortest_path", 0)

        if shortest > 0:
            ratios.append(r.path_length / shortest)

    if not ratios:
        return 1.0
    return sum(ratios) / len(ratios)


def compute_average_follow_human_success(results: List["EpisodeResult"]) -> float:
    """Compute average follow human task success rate.

    Args:
        results: List of episode results

    Returns:
        Average follow human task success rate in [0, 1]
    """
    if not results:
        return 0.0
    
    success_values = []
    for r in results:
        success = r.metrics.get("follow_human_task_success")
        if success is not None:
            try:
                success_values.append(float(success))
            except (TypeError, ValueError):
                pass
    
    if not success_values:
        return 0.0
    return sum(success_values) / len(success_values)


def compute_average_follow_human_success_ratio(results: List["EpisodeResult"]) -> float:
    """Compute average follow human success ratio.

    Args:
        results: List of episode results

    Returns:
        Average follow human success ratio in [0, 1]
    """
    if not results:
        return 0.0
    
    ratio_values = []
    for r in results:
        ratio = r.metrics.get("follow_human_success_ratio")
        if ratio is not None:
            try:
                ratio_values.append(float(ratio))
            except (TypeError, ValueError):
                pass
    
    if not ratio_values:
        return 0.0
    return sum(ratio_values) / len(ratio_values)


def compute_average_csr(results: List["EpisodeResult"]) -> float:
    """Compute average Completion Success Rate (CSR).

    Args:
        results: List of episode results

    Returns:
        Average CSR in [0, 1]
    """
    if not results:
        return 0.0
    
    csr_values = []
    for r in results:
        csr = r.metrics.get("csr")
        if csr is not None:
            try:
                csr_values.append(float(csr))
            except (TypeError, ValueError):
                pass
    
    if not csr_values:
        return 0.0
    return sum(csr_values) / len(csr_values)


def compute_average_softsr(results: List["EpisodeResult"]) -> float:
    """Compute average Soft Success Rate (SoftSR).

    Args:
        results: List of episode results

    Returns:
        Average SoftSR in [0, 1]
    """
    if not results:
        return 0.0
    
    softsr_values = []
    for r in results:
        softsr = r.metrics.get("softsr")
        if softsr is not None:
            try:
                softsr_values.append(float(softsr))
            except (TypeError, ValueError):
                pass
    
    if not softsr_values:
        return 0.0
    return sum(softsr_values) / len(softsr_values)


def compute_average_sii(results: List["EpisodeResult"]) -> float:
    """Compute average Social Intrusion Index (SII).

    SII = average social violation ratio across all episodes.

    Args:
        results: List of episode results

    Returns:
        Average SII in [0, 1]
    """
    if not results:
        return 0.0
    
    sii_values = []
    for r in results:
        sii = r.metrics.get("social_violation_ratio")
        if sii is not None:
            try:
                sii_values.append(float(sii))
            except (TypeError, ValueError):
                pass
    
    if not sii_values:
        return 0.0
    return sum(sii_values) / (len(sii_values) * 2)


def compute_all_metrics(
    results: List["EpisodeResult"],
    shortest_paths: Optional[List[float]] = None,
    success_threshold: float = 1.0,
) -> NavigationMetrics:
    """Compute all navigation metrics.

    Args:
        results: List of episode results
        shortest_paths: Precomputed shortest paths
        success_threshold: Distance threshold for success

    Returns:
        NavigationMetrics with all computed values
    """
    return NavigationMetrics(
        success_rate=compute_success_rate(results),
        spl=compute_spl(results, shortest_paths),
        navigation_error=compute_navigation_error(results),
        oracle_success_rate=compute_oracle_success(results, success_threshold),
        path_length_ratio=compute_path_length_ratio(results, shortest_paths),
        follow_human_success=compute_average_follow_human_success(results),
        follow_human_success_ratio=compute_average_follow_human_success_ratio(results),
        csr=compute_average_csr(results),
        softsr=compute_average_softsr(results),
        sii=compute_average_sii(results),
        num_episodes=len(results),
    )


 
def compute_trajectory_length(trajectory: List["TrajectoryPoint"]) -> float:
    """Compute total length of trajectory."""
    if len(trajectory) < 2:
        return 0.0

    length = 0.0
    for i in range(1, len(trajectory)):
        length += euclidean_distance(
            trajectory[i - 1].position,
            trajectory[i].position,
        )
    return length


_NUMBER_WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
}

_NUMBER_TENS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_NUMBER_UNITS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_NUMBER_WORD_PATTERN = re.compile(r"\b(" + "|".join(_NUMBER_WORD_TO_DIGIT.keys()) + r")\b")


def _normalize_for_eqa(text: str) -> str:
    """Lowercase + numeric normalisation for EQA substring matching.

    Implements the "numeric-normalized substring matching" from the paper's
    Table III footnote: spelled-out numbers are mapped to digits, thousands
    separators are stripped, and trailing decimal zeros are removed.
    """
    s = text.lower()

    # 1) Strip thousands separators inside numbers ("1,000" -> "1000").
    while True:
        new = re.sub(r"(\d),(\d)", r"\1\2", s)
        if new == s:
            break
        s = new

    # 2) Strip trailing zeros after a decimal point ("5.0" -> "5", "5.50" -> "5.5").
    s = re.sub(r"(\d)\.0+(?!\d)", r"\1", s)
    s = re.sub(r"(\d\.\d*?)0+(?!\d)", r"\1", s)

    # 3) Compound number words: "twenty-three" or "twenty three" -> "23".
    for ten in _NUMBER_TENS:
        for unit in _NUMBER_UNITS:
            combined = str(int(_NUMBER_WORD_TO_DIGIT[ten]) + int(_NUMBER_WORD_TO_DIGIT[unit]))
            s = re.sub(rf"\b{ten}[\s-]{unit}\b", combined, s)

    # 4) Standalone number words: "three" -> "3", "twenty" -> "20", ...
    s = _NUMBER_WORD_PATTERN.sub(lambda m: _NUMBER_WORD_TO_DIGIT[m.group(0)], s)

    return s


def compute_eqa(ground_truth: str, model_answer: str) -> bool:
    """Compute the EQA accuracy success indicator.

    Per the paper's Table III, success is decided by *numeric-normalized
    substring matching*: both strings are lowercased, spelled-out numbers
    are mapped to digits, thousands separators are stripped, and trailing
    decimal zeros are removed before checking whether the normalised
    ground-truth string occurs inside the normalised model answer.
    """
    if not isinstance(ground_truth, str) or not isinstance(model_answer, str):
        return False
    return _normalize_for_eqa(ground_truth) in _normalize_for_eqa(model_answer)


def compute_follow_human_task_success(
    human_paths: Dict[str, Any],
    trajectory: List["TrajectoryPoint"],
    distance_threshold: float = 3.0,
    inner_threshold: float = 1.0,
) -> float:
    """Offline overall success of a FOLLOW_HUMAN task (returns 0.0 or 1.0).

    Steps:
    1. Take the full trajectory of one human from `human_paths` (with step ids).
    2. Walk backwards to find the earliest step at which the human stops moving
       (positions[idx:] all equal, idx minimal).
    3. Evaluate the robot pose at that step:
       - 2D distance between robot and human falls in (inner_threshold, distance_threshold];
       - and the human lies within the robot's forward 120 deg cone.
       Heading is read from yaw/quaternion if available; else estimated from neighbouring positions.
    """
    if not isinstance(human_paths, dict) or not trajectory:
        return 0.0

    # Only single-human FOLLOW is supported; pick the first key when multiple are present.
    if not human_paths:
        return 0.0
    human_id, positions = next(iter(human_paths.items()))
    if not isinstance(positions, list) or not positions:
        return 0.0

    samples = _follow_human_samples(positions)
    n = len(samples)
    if n == 0:
        return 0.0

    # Earliest step from which the human's position stops changing.
    stop_idx = _follow_human_stop_index(samples)

    # Robot pose on the same timeline as the human's stop step.
    hx, hy, _ = samples[stop_idx].position

    robot_idx = _trajectory_start_index_for_sample(trajectory, samples[stop_idx])
    if robot_idx is None:
        robot_idx = min(stop_idx, len(trajectory) - 1)
    if robot_idx >= len(trajectory):
        robot_idx = len(trajectory) - 1
    robot_pt = trajectory[robot_idx]
    rx, ry, _ = robot_pt.position

    # Distance check on the xy plane.
    dx = hx - rx
    dy = hy - ry
    dist_xy = float(np.hypot(dx, dy))
    if not (inner_threshold < dist_xy <= distance_threshold):
        return 0.0

    heading = _robot_heading_xy(trajectory, robot_idx)

    # Forward 120 deg cone check.
    if heading is None:
        return 1.0

    vec = np.asarray([dx, dy], dtype=np.float32)
    v_norm = np.linalg.norm(vec)
    if v_norm <= 1e-5:
        return 1.0
    vec /= v_norm
    cos_theta = float(np.clip(np.dot(heading, vec), -1.0, 1.0))
    if cos_theta >= 0.5:  # 120 deg cone, cos(60 deg)
        return 1.0

    return 0.0


def compute_follow_human_success_ratio(
    human_paths: Dict[str, Any],
    trajectory: List["TrajectoryPoint"],
    distance_threshold: float = 3.0,
    inner_threshold: float = 1.0,
) -> float:
    """Offline FOLLOW_HUMAN per-step success ratio inside the follow segment.

    Definition:
    - The follow segment starts at step 1.
    - The follow-end step is the earliest step (scanned from the end) at which the human stops moving.
    - total_steps = follow_end_step.
    - A step is "successful" iff distance in (inner_threshold, distance_threshold] AND human in the
      robot's forward 120 deg cone. Returns success_steps / total_steps.
    """
    if not isinstance(human_paths, dict) or not trajectory:
        return 0.0
    if not human_paths:
        return 0.0

    # Single-human FOLLOW only; first key wins.
    human_id, positions = next(iter(human_paths.items()))
    if not isinstance(positions, list) or not positions:
        return 0.0

    samples = _follow_human_samples(positions)
    n = len(samples)
    if n == 0:
        return 0.0

    stop_idx = _follow_human_stop_index(samples)

    if stop_idx < 0:
        return 0.0

    # Follow starts at step 1, so total = stop_idx + 1.
    total_steps = stop_idx + 1
    if total_steps <= 0:
        return 0.0

    success_steps = 0
    robot_indexes = _trajectory_start_indexes_for_samples(trajectory, samples[: stop_idx + 1])

    for i in range(0, stop_idx + 1):
        hx, hy, _ = samples[i].position

        robot_idx = robot_indexes[i]
        if robot_idx is None:
            robot_idx = i
        if robot_idx >= len(trajectory):
            break
        cur_pt = trajectory[robot_idx]
        rx, ry, _ = cur_pt.position

        dx = hx - rx
        dy = hy - ry
        dist_xy = float(np.hypot(dx, dy))
        if not (inner_threshold < dist_xy <= distance_threshold):
            continue

        heading = _robot_heading_xy(trajectory, robot_idx)

        if heading is None:
            in_front = True
        else:
            vec = np.asarray([dx, dy], dtype=np.float32)
            v_norm = np.linalg.norm(vec)
            if v_norm <= 1e-5:
                in_front = True
            else:
                vec /= v_norm
                cos_theta = float(np.clip(np.dot(heading, vec), -1.0, 1.0))
                in_front = cos_theta >= 0.5  # 120 deg cone, cos(60 deg)

        if in_front:
            success_steps += 1

    return float(success_steps / total_steps) if total_steps > 0 else 0.0


def _follow_human_samples(positions: List[Any]) -> List[_FollowHumanSample]:
    samples: List[_FollowHumanSample] = []
    for item in positions:
        if isinstance(item, dict):
            metadata = item
            xyz = item.get("position")
        else:
            metadata = {}
            xyz = item
        if not isinstance(xyz, (list, tuple)) or len(xyz) < 3:
            continue
        try:
            position = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        except (TypeError, ValueError):
            continue
        samples.append(
            _FollowHumanSample(
                position=position,
                frame=_optional_int(metadata.get("frame")),
                step=_optional_int(metadata.get("step")),
                time_s=_optional_float(metadata.get("time_s")),
            )
        )
    return samples


def _follow_human_stop_index(samples: List[_FollowHumanSample]) -> int:
    stop_idx = len(samples) - 1
    last_xy = np.asarray(samples[-1].position[:2], dtype=np.float32)
    for i in range(len(samples) - 2, -1, -1):
        cur_xy = np.asarray(samples[i].position[:2], dtype=np.float32)
        if not np.allclose(cur_xy, last_xy, atol=1e-3):
            break
        stop_idx = i
    return stop_idx


def _trajectory_start_index_for_sample(
    trajectory: List["TrajectoryPoint"],
    sample: _FollowHumanSample,
) -> Optional[int]:
    sample_value = _sample_timeline_value(sample)
    if sample_value is None:
        return None
    for index, point in enumerate(trajectory):
        point_value = _trajectory_point_timeline_value(point)
        if point_value is not None and point_value >= sample_value:
            return index
    return len(trajectory)


def _trajectory_start_indexes_for_samples(
    trajectory: List["TrajectoryPoint"],
    samples: List[_FollowHumanSample],
) -> List[Optional[int]]:
    values: List[float] = []
    indexes: List[int] = []
    for index, point in enumerate(trajectory):
        point_value = _trajectory_point_timeline_value(point)
        if point_value is not None:
            values.append(point_value)
            indexes.append(index)

    if not values:
        return [None for _ in samples]

    if any(values[index] > values[index + 1] for index in range(len(values) - 1)):
        return [_trajectory_start_index_for_sample(trajectory, sample) for sample in samples]

    result: List[Optional[int]] = []
    for sample in samples:
        sample_value = _sample_timeline_value(sample)
        if sample_value is None:
            result.append(None)
            continue
        position = bisect_left(values, sample_value)
        result.append(indexes[position] if position < len(indexes) else len(trajectory))
    return result


def _sample_timeline_value(sample: _FollowHumanSample) -> Optional[float]:
    if sample.step is not None:
        return float(sample.step)
    return None


def _trajectory_point_timeline_value(point: Any) -> Optional[float]:
    step = getattr(point, "step", None)
    if step is not None:
        try:
            return float(step)
        except (TypeError, ValueError):
            return None
    return None


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _robot_heading_xy(trajectory: List["TrajectoryPoint"], index: int) -> Optional[np.ndarray]:
    if index < 0 or index >= len(trajectory):
        return None

    point = trajectory[index]
    if index <= 0:
        return None
    prev_pt = trajectory[index - 1]
    rx, ry, _ = point.position
    px, py, _ = prev_pt.position
    hdx = float(rx - px)
    hdy = float(ry - py)
    norm = np.hypot(hdx, hdy)
    if norm <= 1e-5:
        return None
    return np.asarray([hdx / norm, hdy / norm], dtype=np.float32)


def _trajectory_point_yaw_rad(point: Any) -> Optional[float]:
    yaw_rad = getattr(point, "yaw_rad", None)
    if yaw_rad is not None:
        try:
            return float(yaw_rad)
        except (TypeError, ValueError):
            return None

    orientation = getattr(point, "orientation", None)
    if not isinstance(orientation, (list, tuple)) or len(orientation) < 4:
        return None
    try:
        w, x, y, z = [float(v) for v in orientation[:4]]
    except (TypeError, ValueError):
        return None
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


# --- Measure infrastructure ------------------------------------------------


@dataclass
class MeasureSetup:
    """Context bundle required to initialize a measure."""

    goal_position: Tuple[float, float, float]
    goal_radius: float
    waypoints: Optional[List[Tuple[float, float, float]]] = None
    shortest_path: Optional[float] = None
    # Object list [(name, (x, y, z)), ...] used by object-proximity measures.
    objects: Optional[List[Tuple[str, Tuple[float, float, float]]]] = None
    # Object proximity threshold; falls back to goal_radius if not set.
    object_threshold: Optional[float] = None
    # Human list [(name, (x, y, z)), ...] used by social-distance measures.
    humans: Optional[List[Tuple[str, Tuple[float, float, float]]]] = None
    # Social-distance threshold (0.5 is an arbitrary default).
    social_distance: Optional[float] = 0.5
    # Room-zone AABBs from the envset's room_zone field, e.g.
    # {"study_doorway": {"aabb_min": [...], "aabb_max": [...]}, ...}
    room_zones: Optional[Dict[str, Dict[str, Any]]] = None
    # NavMesh instance (optional; enables geodesic object-proximity checks).
    navmesh: Optional[Any] = None

def euclidean_distance(
    pos_a: Union[List[float], np.ndarray, Tuple[float, float, float]],
    pos_b: Union[List[float], np.ndarray, Tuple[float, float, float]],
) -> float:
    """3D Euclidean distance; uses numpy so list / ndarray inputs both work."""
    return float(np.linalg.norm(np.asarray(pos_b) - np.asarray(pos_a), ord=2))


# ---------------------------
# NavMesh-based object proximity helpers
# ---------------------------

def _rot2d(vxy: np.ndarray, deg: float) -> np.ndarray:
    """Rotate a 2D vector around the Z axis; returns a unit vector."""
    if not NAV_AVAILABLE:
        return vxy
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    x, y = float(vxy[0]), float(vxy[1])
    out = np.array([ca * x - sa * y, sa * x + ca * y], dtype=float)
    n = np.linalg.norm(out)
    return out / n if n > 1e-9 else out


def _to_carb(p: np.ndarray) -> Any:
    """Convert a numpy array to carb.Float3."""
    if not NAV_AVAILABLE or carb is None:
        return None
    return carb.Float3(float(p[0]), float(p[1]), float(p[2]))


def _to_np(p: Any) -> Optional[np.ndarray]:
    """Convert carb.Float3 / list / tuple / numpy array to a numpy array."""
    if p is None:
        return None

    if isinstance(p, np.ndarray):
        return p

    if isinstance(p, (list, tuple)):
        if len(p) >= 3:
            return np.array([float(p[0]), float(p[1]), float(p[2])], dtype=float)
        return None

    if NAV_AVAILABLE and carb is not None:
        try:
            if hasattr(p, 'x') and hasattr(p, 'y') and hasattr(p, 'z'):
                return np.array([float(p.x), float(p.y), float(p.z)], dtype=float)
        except Exception:
            pass

    return None


def _closest_point_on_island(navmesh: Any, pos_np: np.ndarray, island_id: int) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Project pos onto the NavMesh, returning (projected_point, island_id) or (None, None).

    The actual Isaac Sim API does not support a `search_island_id` parameter, so we
    can only return the closest point. The island_id parameter is kept for interface
    compatibility but is not used.
    """
    if not NAV_AVAILABLE or navmesh is None:
        return None, None
    try:
        # API: query_closest_point(target, area_indices=[], agent_ids=[], obstacle_ids=[])
        # Returns (carb.Float3, int) per docs, but defensive unpacking is needed.
        result = navmesh.query_closest_point(target=_to_carb(pos_np))
        if result is None:
            return None, None

        p_carb = None
        isl = None

        try:
            if isinstance(result, (tuple, list)) and len(result) >= 2:
                p_carb, isl = result[0], result[1]
            elif isinstance(result, tuple) and len(result) == 2:
                p_carb, isl = result
            else:
                p_carb = result
                isl = None
        except (TypeError, ValueError, IndexError):
            return None, None

        if p_carb is None:
            return None, None

        np_pos = _to_np(p_carb)
        if np_pos is None:
            return None, None

        try:
            isl_int = int(isl) if isl is not None else None
        except (TypeError, ValueError):
            isl_int = None

        return np_pos, isl_int
    except Exception as e:
        return None, None


def _is_on_mesh(navmesh: Any, pos_np: np.ndarray, island_id: int, tol: float) -> Tuple[bool, Optional[np.ndarray]]:
    """Return (is_on_mesh, projected_point); on-mesh iff projection error < tol."""
    proj, _ = _closest_point_on_island(navmesh, pos_np, island_id)
    if proj is None:
        return False, None
    return (float(np.linalg.norm(proj - pos_np)) < float(tol)), proj


def _find_edge_candidate(
    navmesh: Any,
    object_probe_np: np.ndarray,
    dir_xy_unit: np.ndarray,
    island_id: int,
    *,
    step: float = 0.05,
    max_dist: float = 6.0,
    tol: float = 0.03,
) -> Optional[np.ndarray]:
    """Sweep outward from the object along `dir_xy_unit`.

    - If the object sits inside a NavMesh hole (e.g. a table or bed), the sweep crosses
      from off-mesh to on-mesh. Return the projected point near the first such edge.
    - If the object is already on the mesh (a ground object), return its direct projection.
    """
    if not NAV_AVAILABLE or navmesh is None:
        return None

    prev_inside, _ = _is_on_mesh(navmesh, object_probe_np, island_id, tol)
    prev_t = 0.0

    t = float(step)
    while t <= float(max_dist):
        pos = object_probe_np + np.array([dir_xy_unit[0] * t, dir_xy_unit[1] * t, 0.0], dtype=float)
        inside, proj = _is_on_mesh(navmesh, pos, island_id, tol)

        # off-mesh -> on-mesh transition: bisect to refine the edge.
        if inside and (not prev_inside):
            lo, hi = prev_t, t
            for _ in range(25):
                mid = 0.5 * (lo + hi)
                mid_pos = object_probe_np + np.array([dir_xy_unit[0] * mid, dir_xy_unit[1] * mid, 0.0], dtype=float)
                inside_mid, _ = _is_on_mesh(navmesh, mid_pos, island_id, tol)
                if inside_mid:
                    hi = mid
                else:
                    lo = mid

            edge_pos = object_probe_np + np.array([dir_xy_unit[0] * hi, dir_xy_unit[1] * hi, 0.0], dtype=float)
            _, edge_proj = _is_on_mesh(navmesh, edge_pos, island_id, tol)
            return edge_proj  # walkable point near the edge

        prev_inside = inside
        prev_t = t
        t += float(step)

    # Starting point already on mesh: return its projection (ground object).
    if prev_inside:
        _, proj0 = _is_on_mesh(navmesh, object_probe_np, island_id, tol)
        return proj0

    return None


def _check_nav_available() -> bool:
    """Runtime check for NavMesh availability; retries the import if it failed at module load."""
    global NAV_AVAILABLE, carb, nav
    if NAV_AVAILABLE:
        return True
    try:
        import carb
        import omni.anim.navigation.core as nav
        NAV_AVAILABLE = True
        return True
    except Exception:
        return False


def compute_object_goal_point(
    navmesh: Any,
    robot_pos: np.ndarray,
    object_pos: np.ndarray,
    *,
    fan_angles_deg: Tuple[float, ...] = (-60.0, -30.0, 0.0, 30.0, 60.0),
    scan_step: float = 0.05,
    scan_max_dist: float = 6.0,
    on_mesh_tol: float = 0.03,
    debug_obj_name: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Compute the NavMesh goal_point used for object-proximity checks.

    The goal point is:
    - Locked to the robot's current NavMesh island (no level/island jumps).
    - Near the edge of the object's hole or obstacle, biased to the robot's side.
    - Selected by shortest geodesic distance among candidates (avoids the far side of, e.g., a bed).

    Args:
        navmesh:        NavMesh instance from omni.anim.navigation.core.
        robot_pos:      Robot position [x, y, z].
        object_pos:     Object position [x, y, z].
        fan_angles_deg: Fan scan angles in degrees relative to the object->robot direction.
        scan_step:      Scan step size in metres.
        scan_max_dist:  Maximum scan distance in metres.
        on_mesh_tol:    Tolerance in metres for considering a point "on the mesh".
        debug_obj_name: Optional object name used in error messages.

    Returns:
        (goal_point [x, y, z], None) on success, (None, error_message) on failure.
    """
    if not _check_nav_available():
        return None, "NavMesh module unavailable (omni.anim.navigation.core could not be imported)"
    if navmesh is None:
        return None, "NavMesh instance is None (not initialised yet, or already invalidated)"

    try:
        # 1) Anchor: snap the robot to the NavMesh and obtain its island.
        # Note: the actual API does not support a search_island_id parameter.
        result = navmesh.query_closest_point(target=_to_carb(robot_pos))
        if result is None:
            return None, f"Robot position ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f}) could not be projected onto NavMesh"

        # Extract position and island_id; query_closest_point returns (carb.Float3, int) per docs,
        # but we unpack defensively.
        robot_on_mesh_carb = None
        robot_island = -1

        try:
            if isinstance(result, (tuple, list)) and len(result) >= 2:
                robot_on_mesh_carb, robot_island = result[0], result[1]
            elif isinstance(result, tuple) and len(result) == 2:
                robot_on_mesh_carb, robot_island = result
            else:
                robot_on_mesh_carb = result
                robot_island = -1
        except (TypeError, ValueError, IndexError) as e:
            return None, f"Robot position ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f}) could not be projected onto NavMesh (unpack failed: {type(result)}, error: {e})"

        if robot_on_mesh_carb is None:
            return None, f"Robot position ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f}) could not be projected onto NavMesh (got None)"

        robot_on_mesh = _to_np(robot_on_mesh_carb)
        if robot_on_mesh is None:
            return None, f"Robot position ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f}) could not be projected onto NavMesh (position conversion failed, type: {type(robot_on_mesh_carb)})"

        try:
            robot_island = int(robot_island) if robot_island is not None else -1
        except (TypeError, ValueError):
            robot_island = -1

        # 2) Object probe point: use the object's xy with the robot's NavMesh z so that
        # tabletop/bed-surface heights don't interfere.
        object_probe = np.array([object_pos[0], object_pos[1], robot_on_mesh[2]], dtype=float)

        # 3) Base direction: object -> robot, on the horizontal plane.
        base_xy = robot_on_mesh[:2] - object_probe[:2]
        n = np.linalg.norm(base_xy)
        if n < 1e-9:
            return robot_on_mesh, None  # robot is directly above the object; return its position
        base_xy = base_xy / n

        # 4) Generate candidates by scanning fan angles for "off-mesh -> on-mesh" edge points.
        candidates = []
        for ang in fan_angles_deg:
            dir_xy = _rot2d(base_xy, float(ang))
            cand = _find_edge_candidate(
                navmesh,
                object_probe_np=object_probe,
                dir_xy_unit=dir_xy,
                island_id=robot_island,
                step=scan_step,
                max_dist=scan_max_dist,
                tol=on_mesh_tol,
            )
            if cand is not None:
                # Re-project to lock the island (more robust against ramps / numerical jitter).
                cand2, _ = _closest_point_on_island(navmesh, cand, robot_island)
                if cand2 is not None:
                    candidates.append(cand2)

        # No candidates: fall back to the closest walkable point on the same island.
        if len(candidates) == 0:
            fallback, _ = _closest_point_on_island(navmesh, object_probe, robot_island)
            if fallback is not None:
                return fallback, None
            obj_name_str = f" '{debug_obj_name}'" if debug_obj_name else ""
            return None, f"Object{obj_name_str} position ({object_pos[0]:.2f}, {object_pos[1]:.2f}, {object_pos[2]:.2f}) could not be projected onto the robot's island {robot_island}, and no walkable scan direction succeeded"

        # 5) Pick the candidate with the smallest geodesic shortest-path cost.
        best_point = candidates[0]
        best_cost = float("inf")
        valid_paths = 0

        for cand in candidates:
            try:
                path = navmesh.query_shortest_path(start_pos=robot_on_mesh_carb, end_pos=_to_carb(cand))
                if path is None:
                    continue
                cost = float(path.length())
                if cost < best_cost:
                    best_cost = cost
                    best_point = cand
                valid_paths += 1
            except Exception as e:
                continue

        if valid_paths == 0:
            obj_name_str = f" '{debug_obj_name}'" if debug_obj_name else ""
            return None, f"Object{obj_name_str}: found {len(candidates)} candidates but no shortest-path query succeeded (likely different islands or blocked path)"

        return best_point, None
    except Exception as e:
        obj_name_str = f" '{debug_obj_name}'" if debug_obj_name else ""
        return None, f"Exception while computing goal_point{obj_name_str}: {type(e).__name__}: {e}"


class Measure:
    """Abstract base class for all measures."""

    cls_uuid: str = "measure"
    dependencies: Tuple[str, ...] = ()

    def __init__(self, manager: "MeasureManager"):
        self.manager = manager
        self._metric: Optional[float] = None

    def reset_metric(self, position: Tuple[float, float, float]):
        raise NotImplementedError

    def update_metric(self, position: Tuple[float, float, float]):
        raise NotImplementedError

    def get_metric(self) -> Optional[float]:
        return self._metric


class MeasureManager:
    """Manager that registers measures, validates dependencies, and handles lifecycle."""

    def __init__(self, setup: MeasureSetup):
        self.setup = setup
        self.measures: Dict[str, Measure] = {}
        self._order: List[str] = []

    def register_measure(self, measure_cls: type[Measure]):
        for dependency in measure_cls.dependencies:
            if dependency not in self.measures:
                raise ValueError(
                    f"Measure {measure_cls.cls_uuid} missing dependency: {dependency}"
                )
        instance = measure_cls(self)
        self.measures[measure_cls.cls_uuid] = instance
        self._order.append(measure_cls.cls_uuid)

    def get_measure(self, measure_uuid: str) -> Optional[Measure]:
        return self.measures.get(measure_uuid)

    def reset_measures(self, position: Tuple[float, float, float]):
        for uuid in self._order:
            self.measures[uuid].reset_metric(position)

    def update_measures(self, position: Tuple[float, float, float]):
        for uuid in self._order:
            self.measures[uuid].update_metric(position)

    def get_measurements(self) -> Dict[str, float]:
        measurements: Dict[str, float] = {}
        for uuid, measure in self.measures.items():
            metric = measure.get_metric()
            if metric is not None:
                # Allow measures to return a dict (e.g. object-subtask status); otherwise stay float-compatible.
                if isinstance(metric, dict):
                    measurements[uuid] = metric
                else:
                    measurements[uuid] = float(metric)

        # Backward-compat aliases for legacy metric names.
        one = measurements.get("oracle_navigation_error")
        if one is not None:
            measurements.setdefault("min_distance_to_goal", one)

        shortest = self.setup.shortest_path
        if shortest is not None:
            measurements.setdefault("shortest_path", shortest)

        return measurements


class PathLength(Measure):
    """Cumulative path length (PL)."""

    cls_uuid = "path_length"

    def __init__(self, manager: "MeasureManager"):
        super().__init__(manager)
        self._previous_position: Optional[Tuple[float, float, float]] = None

    def reset_metric(self, position: Tuple[float, float, float]):
        self._previous_position = position
        self._metric = 0.0

    def update_metric(self, position: Tuple[float, float, float]):
        if self._previous_position is not None:
            self._metric = float(
                (self._metric or 0.0)
                + euclidean_distance(position, self._previous_position)
            )
        self._previous_position = position


class DistanceToGoal(Measure):
    """2D Euclidean distance to the goal (ignores z)."""

    cls_uuid = "distance_to_goal"

    def __init__(self, manager: "MeasureManager"):
        super().__init__(manager)
        self._goal_position = tuple(manager.setup.goal_position)

    def reset_metric(self, position: Tuple[float, float, float]):
        self._metric = self._compute_distance(position)

    def update_metric(self, position: Tuple[float, float, float]):
        self._metric = self._compute_distance(position)

    def _compute_distance(self, position: Tuple[float, float, float]) -> float:
        dx = float(position[0] - self._goal_position[0])
        dy = float(position[1] - self._goal_position[1])
        return float(np.hypot(dx, dy))


class Success(Measure):
    """Access-to-goal: success iff distance is within the goal radius."""

    cls_uuid = "success"
    dependencies = (DistanceToGoal.cls_uuid,)

    def reset_metric(self, position: Tuple[float, float, float]):
        self.update_metric(position)

    def update_metric(self, position: Tuple[float, float, float]):
        distance_measure = self.manager.get_measure(DistanceToGoal.cls_uuid)
        if distance_measure is None or distance_measure.get_metric() is None:
            raise RuntimeError("Success depends on DistanceToGoal which has not been initialised")

        success_threshold = float(self.manager.setup.goal_radius)
        self._metric = float(
            1.0 if distance_measure.get_metric() <= success_threshold else 0.0
        )


class OracleNavigationError(Measure):
    """Oracle Navigation Error (ONE)。"""

    cls_uuid = "oracle_navigation_error"
    dependencies = (DistanceToGoal.cls_uuid,)

    def reset_metric(self, position: Tuple[float, float, float]):
        distance_measure = self.manager.get_measure(DistanceToGoal.cls_uuid)
        if distance_measure is None or distance_measure.get_metric() is None:
            raise RuntimeError("OracleNavigationError requires a DistanceToGoal value")
        self._metric = float(distance_measure.get_metric())

    def update_metric(self, position: Tuple[float, float, float]):
        distance_measure = self.manager.get_measure(DistanceToGoal.cls_uuid)
        if distance_measure is None or distance_measure.get_metric() is None:
            raise RuntimeError("OracleNavigationError requires a DistanceToGoal value")
        current_distance = float(distance_measure.get_metric())
        self._metric = current_distance if self._metric is None else float(min(self._metric, current_distance))


class OracleSuccess(Measure):
    """Oracle Success: agent was ever within goal radius.

    Special case for return-to-start style episodes:
    if the agent starts within goal_radius, we require that it first leaves
    beyond 2.0m at least once, and only then count a later
    re-entry into goal_radius as oracle success.
    """

    cls_uuid = "oracle_success"
    dependencies = (DistanceToGoal.cls_uuid,)

    def reset_metric(self, position: Tuple[float, float, float]):
        self._ever_left_goal_area = False
        success_threshold = float(self.manager.setup.goal_radius)
        dx = float(position[0] - self.manager.setup.goal_position[0])
        dy = float(position[1] - self.manager.setup.goal_position[1])
        start_distance = float(np.hypot(dx, dy))
        self._require_leave_first = bool(start_distance <= success_threshold)
        self._metric = 0.0
        self.update_metric(position)

    def update_metric(self, position: Tuple[float, float, float]):
        distance_measure = self.manager.get_measure(DistanceToGoal.cls_uuid)
        if distance_measure is None or distance_measure.get_metric() is None:
            raise RuntimeError("OracleSuccess requires DistanceToGoal")

        success_threshold = float(self.manager.setup.goal_radius)
        current_distance = float(distance_measure.get_metric())

        if self._require_leave_first:
            if current_distance > 2.0:
                self._ever_left_goal_area = True
            current_success = float(
                1.0
                if (self._ever_left_goal_area and current_distance <= success_threshold)
                else 0.0
            )
        else:
            current_success = float(1.0 if current_distance <= success_threshold else 0.0)

        self._metric = float(max(float(self._metric or 0.0), current_success))

class ObjectReachStatus(Measure):
    """Object-proximity status; tracks per-step approach to each object.

    Uses NavMesh-based goal_point computation to avoid issues where the geometric centre
    of a large object (e.g. a bed) is unreachable. NavMesh is mandatory: if it is unavailable
    or fails, distances are reported as infinity (object considered unreachable). The metric
    deliberately does NOT fall back to plain 3D Euclidean distance.
    """

    cls_uuid = "object_reach_status"
    _THRESHOLDS_M: Tuple[float, float, float] = (0.36, 1.0, 3.0)

    def __init__(self, manager: "MeasureManager"):
        super().__init__(manager)
        objects = manager.setup.objects or []
        # Drop entries whose name starts with "Human" (handled by the social measure).
        self._objects: List[Tuple[str, Tuple[float, float, float]]] = [
            (name, tuple(pos))
            for name, pos in objects
            if not name.startswith("Human")
        ]
        # Fixed 1 m success threshold for object proximity.
        self._threshold = 1.0
        self._navmesh = manager.setup.navmesh
        self._status: Dict[str, Dict[str, float | int]] = {}
        self._step_idx = 0
        # Per-object goal_point cache: {obj_name: (goal_point, last_robot_pos, last_step)}.
        self._goal_points_cache: Dict[str, Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float]], int]] = {}
        self._cache_update_interval = 10  # refresh interval, in steps
        # One-shot fallback warning per object so we don't spam every step.
        self._fallback_warned: Dict[str, bool] = {}
        self._navmesh_missing_warned = False
        if self._navmesh is None and self._objects:
            print(f"[ObjectReachStatus][WARN] NavMesh instance is None; object proximity will be unavailable (objects: {len(self._objects)})")

    def _compute_distance_to_object(
        self, 
        robot_pos: Tuple[float, float, float], 
        obj_pos: Tuple[float, float, float],
        obj_name: str
    ) -> float:
        """Distance from robot to object; uses NavMesh goal_point when possible."""
        # NavMesh may only become available at runtime; retry the import if needed.
        nav_available = _check_nav_available()
        if self._navmesh is None and nav_available:
            try:
                inav = nav.acquire_interface()
                if inav is not None:
                    retry_navmesh = inav.get_navmesh()
                    if retry_navmesh is not None:
                        self._navmesh = retry_navmesh
            except Exception:
                pass

        # Without NavMesh, treat all objects as unreachable (returns inf).
        if self._navmesh is None:
            if not self._navmesh_missing_warned:
                print(f"[ObjectReachStatus][WARN] NavMesh missing; object proximity unavailable, all objects treated as not-reached")
                self._navmesh_missing_warned = True
            return float("inf")

        if self._navmesh is not None:
            goal_point = None
            error_msg = None
            use_cache = False
            if obj_name in self._goal_points_cache:
                cached_goal, cached_robot_pos, cached_step = self._goal_points_cache[obj_name]
                # Reuse the cached goal_point when:
                #   1) the last update was within `cache_update_interval` steps, AND
                #   2) the robot has moved less than 0.5 m since then.
                if cached_step is not None and cached_robot_pos is not None:
                    steps_since_update = self._step_idx - cached_step
                    robot_pos_change = euclidean_distance(robot_pos, cached_robot_pos)
                    if steps_since_update < self._cache_update_interval and robot_pos_change < 0.5:
                        use_cache = True
                        goal_point = cached_goal

            if not use_cache:
                goal_point, error_msg = compute_object_goal_point(
                    navmesh=self._navmesh,
                    robot_pos=np.asarray(robot_pos, dtype=float),
                    object_pos=np.asarray(obj_pos, dtype=float),
                    debug_obj_name=obj_name,
                )
                if goal_point is not None:
                    self._goal_points_cache[obj_name] = (goal_point, robot_pos, self._step_idx)
                else:
                    self._goal_points_cache.pop(obj_name, None)

            if goal_point is not None:
                # 3D Euclidean distance to the goal_point. Callers that need xy-only should drop z.
                return euclidean_distance(robot_pos, tuple(goal_point))

            # goal_point failed; warn once and return unreachable. No Euclidean fallback by design.
            if obj_name not in self._fallback_warned:
                error_detail = error_msg if error_msg else "unknown"
                print(f"[ObjectReachStatus][WARN] NavMesh goal_point failed for object '{obj_name}': {error_detail}")
                print(f"  - robot position: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f})")
                print(f"  - object position: ({obj_pos[0]:.2f}, {obj_pos[1]:.2f}, {obj_pos[2]:.2f})")
                print(f"  - not falling back to 3D Euclidean; object treated as unreachable.")
                self._fallback_warned[obj_name] = True
            return float("inf")

    def reset_metric(self, position: Tuple[float, float, float]):
        self._step_idx = 0
        self._goal_points_cache.clear()
        self._fallback_warned.clear()
        self._status = {}
        for name, obj_pos in self._objects:
            dist = self._compute_distance_to_object(position, obj_pos, name)
            ts_by_thr = {str(thr): -1 for thr in self._THRESHOLDS_M}
            for thr in self._THRESHOLDS_M:
                if dist <= thr and ts_by_thr[str(thr)] < 0:
                    ts_by_thr[str(thr)] = self._step_idx
            self._status[name] = {
                "reached": int(ts_by_thr["1.0"] >= 0),
                "min_distance": float(dist),
                "timestamp": int(ts_by_thr["1.0"]),
                "timestamp_by_threshold": ts_by_thr,
            }
        self._metric = dict(self._status)

    def update_metric(self, position: Tuple[float, float, float]):
        self._step_idx += 1
        if not self._objects:
            self._metric = {}
            return
        
        for name, obj_pos in self._objects:
            dist = self._compute_distance_to_object(position, obj_pos, name)
            obj_stat = self._status.setdefault(
                name,
                {
                    "reached": 0,
                    "min_distance": float(dist),
                    "timestamp": -1,
                    "timestamp_by_threshold": {str(thr): -1 for thr in self._THRESHOLDS_M},
                },
            )
            obj_stat["min_distance"] = float(min(obj_stat["min_distance"], dist))
            ts_by_thr = obj_stat.get("timestamp_by_threshold")
            if not isinstance(ts_by_thr, dict):
                ts_by_thr = {str(thr): -1 for thr in self._THRESHOLDS_M}
                obj_stat["timestamp_by_threshold"] = ts_by_thr

            for thr in self._THRESHOLDS_M:
                key = str(thr)
                if dist <= thr and int(ts_by_thr.get(key, -1)) < 0:
                    ts_by_thr[key] = int(self._step_idx)

            # Backward-compatible fields for the "object" threshold (1.0m)
            obj_stat["reached"] = int(int(ts_by_thr.get("1.0", -1)) >= 0)
            obj_stat["timestamp"] = int(ts_by_thr.get("1.0", -1))
        # Return a copy so external code can't mutate internal state.
        self._metric = {
            name: {
                "reached": int(stat["reached"]),
                "min_distance": float(stat["min_distance"]),
                "timestamp": int(stat["timestamp"]),
                "timestamp_by_threshold": dict(stat.get("timestamp_by_threshold") or {}),
            }
            for name, stat in self._status.items()
        }


class HumanPersonalSpace(Measure):
    """Personal-space violation counter: tracks per-step Human* proximity violations."""

    cls_uuid = "human_personal_space"

    def __init__(self, manager: "MeasureManager"):
        super().__init__(manager)
        humans = manager.setup.humans or []
        self._humans: List[Tuple[str, Tuple[float, float, float]]] = [
            (name, tuple(pos)) for name, pos in humans if name.startswith("Human")
        ]
        self._social_distance = (
            manager.setup.social_distance
            if manager.setup.social_distance is not None
            else manager.setup.goal_radius
        )
        self._step_idx = 0
        self._violation_steps = 0
        self._has_dynamic = False
        try:
            from OmniNavExt.envset.agent_manager import AgentManager
            am = AgentManager.get_instance()
            if am is not None and am.get_all_agent_names():
                self._has_dynamic = True
        except Exception:
            pass
        self._is_social = bool(self._humans) or self._has_dynamic

    def reset_metric(self, position: Tuple[float, float, float]):
        self._step_idx = 0
        self._violation_steps = 0
        self._update_status(position)

    def update_metric(self, position: Tuple[float, float, float]):
        self._step_idx += 1
        self._update_status(position)

    def _update_status(self, position: Tuple[float, float, float]):
        if not self._is_social:
            self._metric = {"is_social": False, "violation_steps": 0}
            return

        violated = False
        # Static humans (fixed position from navigation.objects)
        for _, human_pos in self._humans:
            if euclidean_distance(position, human_pos) <= self._social_distance:
                violated = True
                break

        # Dynamic humans (per-frame position from AgentManager)
        if not violated and self._has_dynamic:
            try:
                from OmniNavExt.envset.agent_manager import AgentManager
                am = AgentManager.get_instance()
                if am is not None:
                    for name in am.get_all_agent_names():
                        dyn_pos = am.get_agent_pos_by_name(name)
                        if dyn_pos is not None:
                            dist = euclidean_distance(position, tuple(dyn_pos))
                            if dist <= self._social_distance:
                                violated = True
                                break
            except Exception:
                pass

        if violated:
            self._violation_steps += 1
        self._metric = {
            "is_social": self._is_social,
            "violation_steps": int(self._violation_steps),
        }


class RoomZoneStatus(Measure):
    """Room-zone entry status: records whether the robot ever entered each room_zone."""

    cls_uuid = "room_zone_status"

    def __init__(self, manager: "MeasureManager"):
        super().__init__(manager)
        raw_zones = manager.setup.room_zones or {}
        self._zones: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name, cfg in raw_zones.items():
            if not isinstance(cfg, dict):
                continue
            aabb_min = cfg.get("aabb_min")
            aabb_max = cfg.get("aabb_max")
            if (
                isinstance(aabb_min, (list, tuple))
                and isinstance(aabb_max, (list, tuple))
                and len(aabb_min) >= 3
                and len(aabb_max) >= 3
            ):
                # Normalise to numpy vectors and enforce min <= max.
                vmin = np.asarray(aabb_min[:3], dtype=np.float32)
                vmax = np.asarray(aabb_max[:3], dtype=np.float32)
                real_min = np.minimum(vmin, vmax)
                real_max = np.maximum(vmin, vmax)
                self._zones[str(name)] = (real_min, real_max)

        # State: {room_name: {"entered": 0/1, "timestamp": int}}.
        self._status: Dict[str, Dict[str, int]] = {}
        self._step_idx = 0

    def reset_metric(self, position: Tuple[float, float, float]):
        self._step_idx = 0
        self._status = {name: {"entered": 0, "timestamp": -1} for name in self._zones.keys()}
        self._update_status(position)

    def update_metric(self, position: Tuple[float, float, float]):
        self._step_idx += 1
        self._update_status(position)

    def _update_status(self, position: Tuple[float, float, float]):
        if not self._zones:
            self._metric = {}
            return

        pos = np.asarray(position[:3], dtype=np.float32)
        for name, (vmin, vmax) in self._zones.items():
            stat = self._status.setdefault(name, {"entered": 0, "timestamp": -1})
            if stat["entered"]:
                # Already entered: timestamp records the first entry only.
                continue
            inside = bool(np.all(pos >= vmin) and np.all(pos <= vmax))
            if inside:
                stat["entered"] = 1
                stat["timestamp"] = self._step_idx

        # Return a copy so external code can't mutate internal state.
        self._metric = {
            name: {"entered": int(stat["entered"]), "timestamp": int(stat["timestamp"])}
            for name, stat in self._status.items()
        }


MEASURE_REGISTRY: Dict[str, type[Measure]] = {
    PathLength.cls_uuid: PathLength,
    DistanceToGoal.cls_uuid: DistanceToGoal,
    Success.cls_uuid: Success,
    OracleNavigationError.cls_uuid: OracleNavigationError,
    OracleSuccess.cls_uuid: OracleSuccess,
    ObjectReachStatus.cls_uuid: ObjectReachStatus,
    HumanPersonalSpace.cls_uuid: HumanPersonalSpace,
    RoomZoneStatus.cls_uuid: RoomZoneStatus,
}


def compute_subtask_progress(subtask_type: str, info: Dict[str, Any], threshold: float) -> float:
    """Continuous subtask progress in [0, 1] (a softer success signal than binary).

    Used by object-proximity, room-entry, and follow-human subtasks.

    Args:
        subtask_type: one of GOTO_OBJECT, GOTO_POINT, RETURN_TO, GOTO_ROOM, FOLLOW_HUMAN.
        info:         subtask status dict.
        threshold:    success threshold (distance or radius).
    """
    if subtask_type in ["GOTO_OBJECT", "GOTO_POINT", "RETURN_TO", "GOTO_LANDMARK"]:
        # Progress is continuous; success is defined elsewhere.
        min_dist = float(info.get("min_distance", float("inf")))
        thr = float(threshold)
        if thr <= 0 or not np.isfinite(thr):
            raise ValueError(f"Invalid threshold for subtask progress: {threshold!r}")
        if min_dist <= thr:
            return 1.0
        return float(thr / min_dist)

    elif subtask_type == "GOTO_ROOM":
        # Room entry is binary in the current implementation; could be softened
        # later if we add distance-to-boundary info.
        if info.get("entered", False):
            return 1.0
        return 0.0

    elif subtask_type == "FOLLOW_HUMAN":
        return float(info.get("success_ratio", 0.0))

    return 0.0


def add_measurement(
    setup: MeasureSetup,
    measure_names: Optional[List[str]] = None,
) -> MeasureManager:
    """Instantiate and register the named measures."""

    manager = MeasureManager(setup)
    names = measure_names or list(MEASURE_REGISTRY.keys())
    for name in names:
        measure_cls = MEASURE_REGISTRY.get(name)
        if measure_cls is None:
            raise KeyError(f"Unknown measure: {name}")
        manager.register_measure(measure_cls)
    return manager
