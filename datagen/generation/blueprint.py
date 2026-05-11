from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from datagen.config import FollowConfig
from datagen.core.registry import ObjectEntry, ObjectRegistry
from datagen.core.geometry import GeometryInterface


def _log_info(msg: str) -> None:
    try:
        import carb  # type: ignore

        carb.log_info(msg)
    except Exception:
        print(msg)


def _log_warn(msg: str) -> None:
    try:
        import carb  # type: ignore

        carb.log_warn(msg)
    except Exception:
        print(f"[WARN] {msg}")

_OBJECT_TASK_TYPES = {"vln", "objectnav", "objnav", "eqa"}
_POINTNAV_TASK_TYPES = {"pointnav", "point_nav", "point-nav"}
_BLUEPRINT_PATH_SAMPLE_DIST = 0.1
_OBJECT_GOAL_ANGLE_OFFSETS = [
    0.0,
    np.pi / 4.0,
    -np.pi / 4.0,
    np.pi / 2.0,
    -np.pi / 2.0,
    3.0 * np.pi / 4.0,
    -3.0 * np.pi / 4.0,
    np.pi,
]

@dataclass
class TaskNode:
    """Represents a single segment in a task chain."""
    task_type: str # "VLN", "Follow"
    start_pos: np.ndarray
    end_pos: np.ndarray
    target_object: Optional[ObjectEntry] = None
    waypoints: List[np.ndarray] = field(default_factory=list) # Intermediate points for tortuosity
    path_points: List[np.ndarray] = field(default_factory=list)
    follow_human_name: Optional[str] = None
    follow_route_commands: List[str] = field(default_factory=list)
    eqa_question: Optional[str] = None
    eqa_answer: Optional[str] = None


@dataclass(frozen=True)
class VirtualHumansContext:
    """Scenario-derived virtual human info used for FOLLOW blueprinting."""

    names: Sequence[str]
    routes_by_name: Dict[str, List[str]]
    spawn_by_name: Dict[str, np.ndarray]
    units_in_meters: float = 1.0

@dataclass
class TaskChain:
    """A sequence of connected tasks."""
    nodes: List[TaskNode] = field(default_factory=list)

class ChainSampler:
    """
    Samples valid task chains (A -> B -> C) ensuring connectivity and logical flow.
    """

    def __init__(
        self,
        registry: ObjectRegistry,
        geometry: GeometryInterface,
        *,
        rng: random.Random,
        follow_cfg: Optional[FollowConfig] = None,
    ):
        self._registry = registry
        self._geo = geometry
        self._rng = rng
        self._follow_cfg = follow_cfg or FollowConfig()

    def build_start_pool(
        self,
        num_episodes: int,
        min_clearance: float,
        min_dist: float,
        *,
        task_types: Optional[Sequence[str]] = None,
        object_start_min_dist: float = 0.0,
    ) -> List[np.ndarray]:
        """Delegate to :class:`StartPoolBuilder` for reusability."""
        from datagen.generation.start_pool import StartPoolBuilder

        builder = StartPoolBuilder(self._geo, self._registry, self._rng)
        return builder.build(
            num_episodes=num_episodes,
            min_clearance=min_clearance,
            min_dist=min_dist,
            task_types=task_types,
            object_start_min_dist=object_start_min_dist,
        )


        if grid_size is None or float(grid_size) <= 0:
            return np.asarray(position, dtype=np.float32)
        size = float(grid_size)
        snapped = np.asarray(
            [
                round(float(position[0]) / size) * size,
                round(float(position[1]) / size) * size,
                float(position[2]),
            ],
            dtype=np.float32,
        )
        nav_snapped = self._geo.snap_point(snapped)
        tol = size * 0.2
        if float(np.linalg.norm((nav_snapped - snapped)[:2])) > tol:
            return None
        snapped[2] = float(nav_snapped[2])
        return snapped

    def _resample_path(self, points: List[np.ndarray], ds: float) -> List[np.ndarray]:
        if not points:
            return []
        if ds <= 0:
            return [np.asarray(p, dtype=np.float32) for p in points]
        new_points = [np.asarray(points[0], dtype=np.float32)]
        for i in range(len(points) - 1):
            p0 = np.asarray(points[i], dtype=np.float32)
            p1 = np.asarray(points[i + 1], dtype=np.float32)
            seg_len = float(np.linalg.norm(p1 - p0))
            if seg_len < 1e-6:
                continue
            direction = (p1 - p0) / seg_len
            dist = float(ds)
            while dist < seg_len:
                next_p = np.asarray(p0 + direction * dist, dtype=np.float32)
                if float(np.linalg.norm(next_p - new_points[-1])) > 1e-6:
                    new_points.append(next_p)
                dist += float(ds)
            if float(np.linalg.norm(p1 - new_points[-1])) > 1e-6:
                new_points.append(p1)
        return new_points

    def _query_path(
        self, start: np.ndarray, end: np.ndarray, min_clearance: Optional[float]
    ) -> Optional[List[np.ndarray]]:
        path, _reason = self._query_path_with_reason(start, end, min_clearance)
        return path

    def _query_path_with_reason(
        self, start: np.ndarray, end: np.ndarray, min_clearance: Optional[float]
    ) -> Tuple[Optional[List[np.ndarray]], str]:
        path = self._geo.query_path(start, end)
        if not path:
            return None, "no_navmesh_path"
        path = [np.asarray(p, dtype=np.float32) for p in path]
        if path:
            path[0] = np.asarray(start, dtype=np.float32)
            path[-1] = np.asarray(end, dtype=np.float32)
            path = self._resample_path(path, _BLUEPRINT_PATH_SAMPLE_DIST)
        return path, "ok"

    def sample_chain(
        self,
        chain_length: int,
        task_types: Optional[List[str]] = None,
        *,
        start_pos: np.ndarray,
        virtual_humans: Optional[VirtualHumansContext] = None,
        min_clearance: Optional[float] = None,
        grid_size: Optional[float] = None,
        pointnav_steps: int = 4,
        pointnav_step_min: float = 1.0,
        pointnav_step_max: float = 2.0,
        pointnav_step_attempts: int = 30,
        object_goal_min: float = 0.5,
        object_goal_max: float = 1.5,
        object_start_min_dist: float = 0.0,
    ) -> TaskChain:
        if task_types is None:
            task_types = ["VLN"]

        chain = TaskChain()
        task_types_norm = {str(t).strip().lower() for t in task_types if t}
        all_objects = self._registry.query()
        if not all_objects and any(t in _OBJECT_TASK_TYPES for t in task_types_norm):
            raise RuntimeError("Registry is empty. Cannot sample object-based tasks.")

        start = np.asarray(start_pos, dtype=np.float32)
        _log_info(
            "[Blueprint] sample_chain start: "
            f"chain_length={chain_length} task_types={list(task_types)} "
            f"start=({float(start[0]):.2f}, {float(start[1]):.2f}, {float(start[2]):.2f})"
        )
        current_start_pos = start
        for i in range(chain_length):
            t_type = str(task_types[i % len(task_types)])
            _log_info(
                "[Blueprint] sample_chain segment start: "
                f"index={i} task_type={t_type} "
                f"start=({float(current_start_pos[0]):.2f}, {float(current_start_pos[1]):.2f}, {float(current_start_pos[2]):.2f})"
            )
            node = self._sample_segment(
                current_start_pos,
                t_type,
                all_objects,
                virtual_humans=virtual_humans,
                min_clearance=min_clearance,
                grid_size=grid_size,
                pointnav_steps=pointnav_steps,
                pointnav_step_min=pointnav_step_min,
                pointnav_step_max=pointnav_step_max,
                pointnav_step_attempts=pointnav_step_attempts,
                object_goal_min=object_goal_min,
                object_goal_max=object_goal_max,
                object_start_min_dist=object_start_min_dist,
            )
            if not node:
                _log_warn(f"Failed to sample segment {i} starting at {current_start_pos}. Terminating chain early.")
                break
            _log_info(
                "[Blueprint] sample_chain segment done: "
                f"index={i} task_type={t_type} "
                f"end=({float(node.end_pos[0]):.2f}, {float(node.end_pos[1]):.2f}, {float(node.end_pos[2]):.2f}) "
                f"path_points={len(node.path_points)}"
            )
            chain.nodes.append(node)
            current_start_pos = node.end_pos

        if not chain.nodes:
            raise RuntimeError("Failed to generate any valid task nodes.")
        return chain

    def _sample_segment(
        self,
        start_pos: np.ndarray,
        task_type: str,
        candidates: List[ObjectEntry],
        max_attempts: int = 20,
        *,
        virtual_humans: Optional[VirtualHumansContext] = None,
        min_clearance: Optional[float] = None,
        grid_size: Optional[float] = None,
        pointnav_steps: int = 4,
        pointnav_step_min: float = 0.5,
        pointnav_step_max: float = 2.0,
        pointnav_step_attempts: int = 30,
        object_goal_min: float = 0.5,
        object_goal_max: float = 1.5,
        object_start_min_dist: float = 0.0,
    ) -> Optional[TaskNode]:
        """Samples a single reachable task segment."""
        task_type_norm = str(task_type).strip().lower()

        if task_type_norm in {"follow", "follow_human", "followhuman"}:
            node = self._sample_follow_segment(
                start_pos,
                virtual_humans=virtual_humans,
                min_clearance=min_clearance,
            )
            return node

        if task_type_norm in _POINTNAV_TASK_TYPES:
            return self._sample_pointnav_segment(
                start_pos=start_pos,
                task_type=task_type,
                steps=pointnav_steps,
                step_min=pointnav_step_min,
                step_max=pointnav_step_max,
                step_attempts=pointnav_step_attempts,
                min_clearance=min_clearance,
                grid_size=grid_size,
            )

        if task_type_norm not in _OBJECT_TASK_TYPES:
            return self._sample_random_segment(
                start_pos=start_pos,
                task_type=task_type,
                step_min=pointnav_step_min,
                step_max=pointnav_step_max,
                step_attempts=pointnav_step_attempts,
                min_clearance=min_clearance,
                grid_size=grid_size,
            )

        for _ in range(max_attempts):
            target_obj = self._rng.choice(candidates)
            node = self._sample_object_segment(
                start_pos=start_pos,
                task_type=task_type,
                target_obj=target_obj,
                object_goal_min=object_goal_min,
                object_goal_max=object_goal_max,
                object_start_min_dist=object_start_min_dist,
                min_clearance=min_clearance,
                grid_size=grid_size,
            )
            if node is None:
                continue
            return node

        return None

    def _sample_pointnav_segment(
        self,
        *,
        start_pos: np.ndarray,
        task_type: str,
        steps: int,
        step_min: float,
        step_max: float,
        step_attempts: int,
        min_clearance: Optional[float],
        grid_size: Optional[float],
    ) -> Optional[TaskNode]:
        steps = max(1, int(steps))
        step_min = float(step_min)
        step_max = float(step_max)
        if step_min > step_max:
            step_min, step_max = step_max, step_min

        current = np.asarray(start_pos, dtype=np.float32)
        path_points: List[np.ndarray] = []
        waypoints: List[np.ndarray] = []

        for _ in range(steps):
            step_end, step_path = self._sample_step(
                start_pos=current,
                step_min=step_min,
                step_max=step_max,
                step_attempts=step_attempts,
                min_clearance=min_clearance,
                grid_size=grid_size,
            )
            if step_end is None or not step_path:
                return None
            if path_points:
                step_path = step_path[1:]
            path_points.extend(step_path)
            waypoints.append(step_end)
            current = step_end

        return TaskNode(
            task_type=task_type,
            start_pos=start_pos,
            end_pos=current,
            waypoints=waypoints[:-1] if len(waypoints) > 1 else [],
            path_points=path_points,
        )

    def _sample_step(
        self,
        *,
        start_pos: np.ndarray,
        step_min: float,
        step_max: float,
        step_attempts: int,
        min_clearance: Optional[float],
        grid_size: Optional[float],
    ) -> Tuple[Optional[np.ndarray], Optional[List[np.ndarray]]]:
        for _ in range(int(step_attempts)):
            angle = self._rng.uniform(0.0, 2.0 * np.pi)
            dist = self._rng.uniform(float(step_min), float(step_max))
            raw = np.asarray(
                [
                    float(start_pos[0]) + float(np.cos(angle)) * dist,
                    float(start_pos[1]) + float(np.sin(angle)) * dist,
                    float(start_pos[2]),
                ],
                dtype=np.float32,
            )
            cand = self._snap_to_grid(raw, grid_size)
            if cand is None:
                continue
            dist_xy = float(np.linalg.norm((cand - start_pos)[:2]))
            if dist_xy < float(step_min) or dist_xy > float(step_max):
                continue
            path = self._query_path(start_pos, cand, min_clearance)
            if not path:
                continue
            return cand, path
        return None, None

    def _build_object_goal_candidates(
        self,
        *,
        center: np.ndarray,
        desired_r: float,
        base_angle: float,
    ) -> List[np.ndarray]:
        candidates: List[np.ndarray] = []
        for angle_offset in _OBJECT_GOAL_ANGLE_OFFSETS:
            angle = float(base_angle) + float(angle_offset)
            raw = np.asarray(
                [
                    float(center[0]) + float(np.cos(angle)) * float(desired_r),
                    float(center[1]) + float(np.sin(angle)) * float(desired_r),
                    float(center[2]),
                ],
                dtype=np.float32,
            )
            candidates.append(self._geo.snap_point(raw))
        return candidates

    def _sample_object_segment(
        self,
        start_pos: np.ndarray,
        *,
        task_type: str,
        target_obj: ObjectEntry,
        object_goal_min: float,
        object_goal_max: float,
        object_start_min_dist: float,
        min_clearance: Optional[float],
        grid_size: Optional[float],
    ) -> Optional[TaskNode]:
        min_r = float(object_goal_min)
        max_r = float(object_goal_max)
        if min_r > max_r:
            min_r, max_r = max_r, min_r
        center = np.asarray(target_obj.position, dtype=np.float32)
        if float(np.linalg.norm((np.asarray(start_pos, dtype=np.float32) - center)[:2])) < float(object_start_min_dist):
            return None
        target_label = str(target_obj.object_id or target_obj.category)
        _log_info(
            "[Blueprint] ObjectNav sampling start: "
            f"target={target_label} category={target_obj.category} "
            f"start=({float(start_pos[0]):.2f}, {float(start_pos[1]):.2f}, {float(start_pos[2]):.2f}) "
            f"center=({float(center[0]):.2f}, {float(center[1]):.2f}, {float(center[2]):.2f}) "
            f"goal_radius=[{min_r:.2f}, {max_r:.2f}]"
        )

        desired_r = 0.5 * (min_r + max_r) if max_r > 0 else 0.0
        desired_r = max(0.0, desired_r)

        offset_vec = np.asarray(start_pos[:2], dtype=np.float32) - np.asarray(center[:2], dtype=np.float32)
        offset_norm = float(np.linalg.norm(offset_vec))
        if offset_norm > 1e-6:
            base_angle = float(np.arctan2(offset_vec[1], offset_vec[0]))
        else:
            base_angle = 0.0

        path: Optional[List[np.ndarray]] = None
        reason = "no_navmesh_path"
        cand = self._geo.snap_point(center)
        for cand in self._build_object_goal_candidates(center=center, desired_r=desired_r, base_angle=base_angle):
            path, reason = self._query_path_with_reason(start_pos, cand, min_clearance)
            if path:
                break

        if not path:
            _log_warn(
                "[Blueprint] ObjectNav sampling failed: "
                f"target={target_label} candidate=({float(cand[0]):.2f}, {float(cand[1]):.2f}, {float(cand[2]):.2f}) "
                f"reason={reason}"
            )
            return None

        _log_info(
            "[Blueprint] ObjectNav sampling success: "
            f"target={target_label} candidate=({float(cand[0]):.2f}, {float(cand[1]):.2f}, {float(cand[2]):.2f}) "
            f"path_points={len(path)} simple_mode=True"
        )
        return TaskNode(
            task_type=task_type,
            start_pos=start_pos,
            end_pos=cand,
            target_object=target_obj,
            path_points=path,
        )

    def _sample_random_segment(
        self,
        *,
        start_pos: np.ndarray,
        task_type: str,
        step_min: float,
        step_max: float,
        step_attempts: int,
        min_clearance: Optional[float],
        grid_size: Optional[float],
    ) -> Optional[TaskNode]:
        end_pos, path = self._sample_step(
            start_pos=start_pos,
            step_min=step_min,
            step_max=step_max,
            step_attempts=step_attempts,
            min_clearance=min_clearance,
            grid_size=grid_size,
        )
        if end_pos is None or not path:
            return None
        return TaskNode(
            task_type=task_type,
            start_pos=start_pos,
            end_pos=end_pos,
            path_points=path,
        )

    def _sample_follow_segment(
        self,
        start_pos: np.ndarray,
        *,
        virtual_humans: Optional[VirtualHumansContext],
        min_clearance: Optional[float] = None,
    ) -> Optional[TaskNode]:
        if virtual_humans is None or not virtual_humans.names:
            return None

        # Pick a virtual human and use the last GoTo target (if available) as a rough episode end anchor.
        agent_name = str(self._rng.choice(list(virtual_humans.names)))
        commands = list(virtual_humans.routes_by_name.get(agent_name) or [])
        if not commands:
            # Minimal requirement: sample a few points and issue GoTo commands (+ optional Idle 1.0).
            human_start = virtual_humans.spawn_by_name.get(agent_name)
            if human_start is None:
                # Fallback to using robot start position as seed for sampling.
                human_start = np.asarray(start_pos, dtype=np.float32)
            commands = self._generate_follow_route_commands(
                agent_name=agent_name,
                seed_pos=np.asarray(human_start, dtype=np.float32),
                units_in_meters=float(virtual_humans.units_in_meters or 1.0),
                min_clearance=min_clearance,
            )
            if not commands:
                return None

        targets = _extract_goto_targets(commands, agent_name=agent_name)
        if not targets:
            return None

        # End near the final GoTo target.
        end_pos = self._geo.snap_point(np.asarray(targets[-1], dtype=np.float32))
        if not self._geo.is_reachable(start_pos, end_pos):
            return None

        return TaskNode(
            task_type="Follow",
            start_pos=start_pos,
            end_pos=end_pos,
            follow_human_name=agent_name,
            follow_route_commands=commands,
        )

    def _generate_follow_route_commands(
        self,
        *,
        agent_name: str,
        seed_pos: np.ndarray,
        units_in_meters: float,
        min_clearance: Optional[float] = None,
    ) -> List[str]:
        cfg = self._follow_cfg
        units_in_meters = float(units_in_meters) if float(units_in_meters) > 0 else 1.0

        def m_to_units(val_m: float) -> float:
            return float(val_m) / units_in_meters

        n_min = max(1, int(cfg.route_points_min))
        n_max = max(n_min, int(cfg.route_points_max))
        n = self._rng.randint(n_min, n_max)

        min_step = max(0.1, m_to_units(float(cfg.route_min_step_m)))
        max_step = max(min_step, m_to_units(float(cfg.route_max_step_m)))
        idle_prob = float(np.clip(float(cfg.route_idle_probability), 0.0, 1.0))
        idle_dur = max(0.0, float(cfg.route_idle_duration_s))

        seed = self._geo.snap_point(np.asarray(seed_pos, dtype=np.float32))
        cur = seed
        cmds: List[str] = []

        for _ in range(n):
            tgt = None
            # Try a handful of samples to find a reachable point for the human.
            for _attempt in range(40):
                if self._rng.random() < 0.5:
                    objs = self._registry.query()
                    if objs:
                        obj = self._rng.choice(objs)
                        base = np.asarray(obj.position, dtype=np.float32)
                    else:
                        base = seed
                else:
                    base = seed
                angle = self._rng.uniform(0.0, 2.0 * np.pi)
                dist = self._rng.uniform(min_step, max_step)
                cand = np.asarray([base[0] + np.cos(angle) * dist, base[1] + np.sin(angle) * dist, base[2]], dtype=np.float32)
                cand = self._geo.snap_point(cand)
                # Ensure sequential reachability for the route.
                if not self._geo.is_reachable(cur, cand):
                    continue
                if float(np.linalg.norm((cand - cur)[:2])) < float(min_step) * 0.5:
                    continue
                tgt = cand
                break
            if tgt is None:
                break
            cmds.append(f"{agent_name} GoTo {tgt[0]:.2f} {tgt[1]:.2f} {tgt[2]:.2f} _")
            cur = tgt
            if self._rng.random() < idle_prob and idle_dur > 0.0:
                cmds.append(f"{agent_name} Idle {idle_dur:.1f}")

        return cmds

def _extract_goto_targets(commands: Sequence[str], *, agent_name: str) -> List[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    for raw in commands:
        if not raw:
            continue
        tokens = str(raw).strip().split()
        if len(tokens) < 5:
            continue
        if str(tokens[0]) != str(agent_name):
            continue
        cmd = str(tokens[1]).strip().lower()
        if not (cmd == "goto" or cmd.startswith("goto")):
            continue
        try:
            x = float(tokens[2])
            y = float(tokens[3])
            z = float(tokens[4])
        except Exception:
            continue
        out.append((x, y, z))
    return out
