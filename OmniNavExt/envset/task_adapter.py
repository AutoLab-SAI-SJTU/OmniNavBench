from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict, List

from OmniNavExt.envset.unit_utils import resolve_env_unit_scale, scale_env_position

from .scenario_types import EnvsetScenarioData, RobotSpec

from OmniNavExt.configs.sensors import RepCameraCfg

# Import pre-defined controller configurations for complex robots
try:
    from OmniNavExt.configs.robots.aliengo import (
        move_by_speed_cfg as aliengo_move_by_speed_cfg,
        move_to_point_cfg as aliengo_move_to_point_cfg,
        move_along_path_cfg as aliengo_move_along_path_cfg,
        go_toward_point_cfg as aliengo_go_toward_point_cfg,
        rotate_cfg as aliengo_rotate_cfg,
        AliengoRobotCfg,
    )
    from OmniNavExt.configs.robots.carter_v1 import (
        move_by_speed_cfg as carter_v1_move_by_speed_cfg,
        move_to_point_cfg as carter_v1_move_to_point_cfg,
        move_along_path_cfg as carter_v1_move_along_path_cfg,
        go_toward_point_cfg as carter_v1_go_toward_point_cfg,
        rotate_cfg as carter_v1_rotate_cfg,
    )
    from OmniNavExt.configs.robots.h1 import (
        move_to_point_cfg as h1_move_to_point_cfg,
        move_by_speed_cfg as h1_move_by_speed_cfg,
        move_along_path_cfg as h1_move_along_path_cfg,
        go_toward_point_cfg as h1_go_toward_point_cfg,
        rotate_cfg as h1_rotate_cfg,
        H1RobotCfg,
    )
    from OmniNavExt.configs.robots.g1 import (
        move_by_speed_cfg as g1_move_by_speed_cfg,
        move_to_point_cfg as g1_move_to_point_cfg,
        move_along_path_cfg as g1_move_along_path_cfg,
        rotate_cfg as g1_rotate_cfg,
    )
    from OmniNavExt.configs.robots.gr1 import (
        move_by_speed_cfg as gr1_move_by_speed_cfg,
        move_to_point_cfg as gr1_move_to_point_cfg,
        move_along_path_cfg as gr1_move_along_path_cfg,
        rotate_cfg as gr1_rotate_cfg,
    )
    from OmniNavExt.configs.robots.kitt15 import (
        move_by_speed_cfg as kitt15_move_by_speed_cfg,
        move_to_point_cfg as kitt15_move_to_point_cfg,
        move_along_path_cfg as kitt15_move_along_path_cfg,
        go_toward_point_cfg as kitt15_go_toward_point_cfg,
        rotate_cfg as kitt15_rotate_cfg,
        KITT15RobotCfg,
    )
    from OmniNavExt.configs.robots.h1_with_hand import H1WithHandRobotCfg
    from OmniNavExt.configs.robots.carter_v1 import CarterV1RobotCfg
    from OmniNavExt.configs.robots.jetbot import JetbotRobotCfg
except ImportError as e:
    print(f"[EnvsetTaskAugmentor] WARNING: Failed to import robot configs: {e}")
    aliengo_move_by_speed_cfg = None
    aliengo_move_to_point_cfg = None
    aliengo_move_along_path_cfg = None
    aliengo_go_toward_point_cfg = None
    aliengo_rotate_cfg = None
    carter_v1_move_by_speed_cfg = None
    carter_v1_move_to_point_cfg = None
    carter_v1_move_along_path_cfg = None
    carter_v1_go_toward_point_cfg = None
    carter_v1_rotate_cfg = None
    h1_move_to_point_cfg = None
    h1_move_by_speed_cfg = None
    h1_move_along_path_cfg = None
    h1_go_toward_point_cfg = None
    h1_rotate_cfg = None
    g1_move_by_speed_cfg = None
    g1_move_to_point_cfg = None
    g1_move_along_path_cfg = None
    g1_rotate_cfg = None
    gr1_move_by_speed_cfg = None
    gr1_move_to_point_cfg = None
    gr1_move_along_path_cfg = None
    gr1_rotate_cfg = None
    AliengoRobotCfg = None
    H1RobotCfg = None
    H1WithHandRobotCfg = None
    CarterV1RobotCfg = None
    JetbotRobotCfg = None
    kitt15_move_by_speed_cfg = None
    kitt15_move_to_point_cfg = None
    kitt15_move_along_path_cfg = None
    kitt15_go_toward_point_cfg = None
    kitt15_rotate_cfg = None
    KITT15RobotCfg = None


class EnvsetTaskAugmentor:
    @staticmethod
    def apply(config: Dict[str, Any], scenario_data: EnvsetScenarioData, scenario_id: str = None) -> Dict[str, Any]:
        tasks = config.get("task_configs")
        if not isinstance(tasks, list):
            return config
        payload = EnvsetTaskAugmentor._build_envset_payload(scenario_data, scenario_id)
        for idx, task in enumerate(tasks):
            EnvsetTaskAugmentor._inject_task(task, payload, scenario_data, robot_prefix=f"envset_{idx}")
        return config

    @staticmethod
    def _build_envset_payload(scenario_data: EnvsetScenarioData, scenario_id: str = None) -> Dict[str, Any]:
        payload = {
            "scene": scenario_data.scene.raw,
            "navmesh": scenario_data.navmesh.raw if scenario_data.navmesh else None,
            "logging": scenario_data.logging,
            "scenario_id": scenario_id,  # Add scenario id to payload
        }
        if scenario_data.virtual_humans:
            vh = scenario_data.virtual_humans
            routes_payload = [EnvsetTaskAugmentor._route_to_dict(rt) for rt in vh.routes]
            payload["virtual_humans"] = {
                "category": vh.category,
                "count": vh.count,
                "name_sequence": list(vh.name_sequence),
                "assets": vh.assets,
                "asset_root": vh.asset_root,
                "spawn_points": [EnvsetTaskAugmentor._spawn_point_to_dict(sp) for sp in vh.spawn_points],
                # Backward-compatible name kept for existing runtime hooks,
                # while also emitting the new alias used by merged episodes.
                "routes": routes_payload,
                "move_routes": routes_payload,
                "options": vh.options,
            }
        robots_payload = [EnvsetTaskAugmentor._robot_to_payload(rb) for rb in scenario_data.robots]
        payload["robots"] = robots_payload

        raw_scenario = scenario_data.raw if isinstance(scenario_data.raw, dict) else {}
        raw_task = raw_scenario.get("task") if isinstance(raw_scenario.get("task"), dict) else {}
        raw_nav = raw_task.get("navigation") if isinstance(raw_task.get("navigation"), dict) else {}

        # Support merged-episode schema: objects/room_zone/answer live under task.navigation.
        # Keep emitting them at envset root for backwards compatibility (e.g., EpisodeLogger init).
        if "objects" in raw_nav:
            payload["objects"] = raw_nav.get("objects")
        elif "objects" in raw_scenario:
            payload["objects"] = raw_scenario.get("objects")

        if "room_zone" in raw_nav:
            payload["room_zone"] = raw_nav.get("room_zone")
        elif "room_zone" in raw_scenario:
            payload["room_zone"] = raw_scenario.get("room_zone")

        if "answer" in raw_nav:
            payload["answer"] = raw_nav.get("answer")
        elif "answer" in raw_scenario:
            payload["answer"] = raw_scenario.get("answer")

        return payload

    @staticmethod
    def _inject_task(task: Dict[str, Any], payload: Dict[str, Any], scenario_data: EnvsetScenarioData, robot_prefix: str):
        envset_entry = task.setdefault("envset", {})
        envset_entry.update(payload)
        # Skip injecting scene_asset_path for Matterport/MP3D; scene will be handled by importer.
        if not EnvsetTaskAugmentor._is_matterport_scene(scenario_data.scene.raw):
            scene_usd = scenario_data.scene.usd_path
            if scene_usd:
                task["scene_asset_path"] = scene_usd

        env_unit_scale = resolve_env_unit_scale(
            scenario_data.scene.units_in_meters,
            context="scenario.scene.units_in_meters",
        )
        task["scene_units_in_meters"] = env_unit_scale
        EnvsetTaskAugmentor._inject_robots(task, scenario_data.robots, robot_prefix, env_unit_scale)

    @staticmethod
    def _is_matterport_scene(scene_raw: Dict[str, Any]) -> bool:
        """Return True if envset marks the scene as MP3D/Matterport."""
        if not scene_raw:
            return False
        try:
            category = str(scene_raw.get("category") or "").lower()
            if "mp3d" in category or "matterport" in category:
                return True
        except Exception:
            pass
        use_mp = scene_raw.get("use_matterport")
        return bool(use_mp)

    @staticmethod
    def _spawn_point_to_dict(spec) -> Dict[str, Any]:
        return {
            "name": spec.name,
            "position": list(spec.position),
            "orientation_deg": spec.orientation_deg,
        }

    @staticmethod
    def _route_to_dict(spec) -> Dict[str, Any]:
        return {
            "name": spec.name,
            "commands": list(spec.commands),
        }

    @staticmethod
    def _robot_to_payload(spec: RobotSpec) -> Dict[str, Any]:
        payload = asdict(spec)
        # remove nested dataclass conversion for control raw data already handled
        return payload

    @staticmethod
    def _inject_robots(
        task: Dict[str, Any],
        robots: tuple[RobotSpec, ...],
        robot_prefix: str,
        env_unit_scale: float,
    ):
        """Inject robot configurations into task dict as RobotCfg objects."""
        if not robots:
            return
        robot_list = task.setdefault("robots", [])

        # Extract existing names (handle both dict and RobotCfg objects for compatibility)
        existing_names = set()
        for entry in robot_list:
            if isinstance(entry, dict):
                existing_names.add(str(entry.get("name")))
            else:
                existing_names.add(str(entry.name))

        for idx, spec in enumerate(robots):
            robot_cfg = EnvsetTaskAugmentor._build_robot_entry(spec, robot_prefix, idx, env_unit_scale)
            if not robot_cfg:
                continue

            # Check for name conflicts and resolve
            if robot_cfg.name in existing_names:
                robot_cfg.name = f"{robot_cfg.name}_{idx}"

            existing_names.add(robot_cfg.name)
            # Debug: log controllers attached to each robot for verification
            try:
                controller_names = [c.name for c in (robot_cfg.controllers or [])]
                print(f"[EnvsetTaskAugmentor] robot={robot_cfg.name} controllers={controller_names}")
            except Exception:
                pass
            robot_list.append(robot_cfg)  # Append RobotCfg object directly

    @staticmethod
    def _build_robot_entry(spec: RobotSpec, robot_prefix: str, idx: int, env_unit_scale: float):
        """Build RobotCfg object from envset RobotSpec.

        Returns: RobotCfg object (Pydantic model) or None
        """
        from OmniNav.core.config import RobotCfg

        robot_type = EnvsetTaskAugmentor._resolve_robot_type(spec)
        if not robot_type:
            return None

        name = spec.label or f"{robot_prefix}_{idx}"
        controllers = EnvsetTaskAugmentor._build_robot_controllers(spec, name)
        sensors = EnvsetTaskAugmentor._build_robot_sensors(spec)

        # Build RobotCfg object directly
        position = scale_env_position(spec.initial_position, env_unit_scale)
        robot_cfg = RobotCfg(
            name=name,
            type=robot_type,
            prim_path=spec.spawn_path or f"/World/Robots/{name}",
            usd_path=spec.usd_path,
            scale=spec.scale,
            position=position,
            orientation=EnvsetTaskAugmentor._orientation_from_deg(spec.initial_orientation_deg),
            controllers=controllers if controllers else None,
            sensors=sensors if sensors else None,
        )

        # Store envset control metadata if needed (via extra fields - Pydantic allows this)
        if spec.control and hasattr(robot_cfg, '__pydantic_extra__'):
            robot_cfg.__pydantic_extra__ = {"envset_control": asdict(spec.control)}

        return robot_cfg

    @staticmethod
    def _build_robot_sensors(spec: RobotSpec):
        """Return None to let policy's configure_robot_sensors() control sensor configuration.
        
        Previously this method returned default sensors from RobotCfg classes (e.g., AliengoRobotCfg),
        but this prevented policy-specific sensor configurations from taking effect because
        _apply_policy_robot_config() was called after sensors were already set.
        
        Now, sensors are configured by:
        1. Policy's configure_robot_sensors() in bench/policy/<policy>/robot_config.py
        2. If no policy config exists, sensors remain None (robot USD's built-in camera is used)
        """
        return None

    @staticmethod
    def _resolve_robot_type(spec: RobotSpec) -> str | None:
        type_name = (spec.type or "").lower()

        # Differential drive robots
        if type_name == "carter_v1":
            return "CarterV1Robot"
        if type_name == "kitt15":
            return "KITT15Robot"
        if type_name in {"carter", "jetbot", "differential_drive"}:
            return "JetbotRobot"

        # Quadruped robots
        if type_name in {"aliengo"}:
            return "AliengoRobot"

        # Humanoid robots
        if type_name in {"h1", "human"}:
            return "H1Robot"
        if type_name in {"g1"}:
            return "G1Robot"
        if type_name in {"gr1"}:
            return "GR1Robot"

        # Manipulation robots
        if type_name in {"franka"}:
            return "FrankaRobot"

        # Fallback to JetbotRobot for backward compatibility
        return "JetbotRobot" if spec.control else None

    @staticmethod
    def _build_robot_controllers(spec: RobotSpec, name: str) -> List:  # Returns List[ControllerCfg] objects
        """Build controller configuration objects for the robot."""
        from OmniNavExt.configs.controllers import (
            MoveToPointBySpeedControllerCfg,
            MoveAlongPathPointsControllerCfg,
            RotateControllerCfg,
            DifferentialDriveMoveBySpeedControllerCfg,
            GoTowardPointControllerCfg,
        )

        params = spec.control.params if spec.control else {}
        control_mode = (spec.control.mode or "").lower() if spec.control and spec.control.mode else ""
        robot_type = (spec.type or "").lower()

        if robot_type == "carter_v1":
            controllers = []
            if carter_v1_move_by_speed_cfg is not None:
                controllers.append(
                    EnvsetTaskAugmentor._clone_and_override_controller(carter_v1_move_by_speed_cfg, params)
                )
            if carter_v1_move_to_point_cfg is not None:
                controllers.append(
                    EnvsetTaskAugmentor._clone_and_override_controller(carter_v1_move_to_point_cfg, params)
                )
            if carter_v1_move_along_path_cfg is not None:
                controllers.append(
                    EnvsetTaskAugmentor._clone_and_override_controller(carter_v1_move_along_path_cfg, params)
                )
            if carter_v1_go_toward_point_cfg is not None:
                controllers.append(
                    EnvsetTaskAugmentor._clone_and_override_controller(carter_v1_go_toward_point_cfg, params)
                )
            if carter_v1_rotate_cfg is not None:
                controllers.append(
                    EnvsetTaskAugmentor._clone_and_override_controller(carter_v1_rotate_cfg, params)
                )
            if controllers:
                return controllers

        if robot_type == "kitt15":
            controllers = []
            for cfg in [kitt15_move_by_speed_cfg, kitt15_move_to_point_cfg,
                        kitt15_move_along_path_cfg, kitt15_go_toward_point_cfg,
                        kitt15_rotate_cfg]:
                if cfg is not None:
                    controllers.append(
                        EnvsetTaskAugmentor._clone_and_override_controller(cfg, params)
                    )
            if controllers:
                return controllers

        # Differential drive robots (jetbot, carter, etc.)
        if robot_type in {"carter", "carter_v1", "jetbot", "differential_drive"}:
            # Carter V1 has different default wheel parameters than Jetbot
            if robot_type == "carter_v1":
                default_wheel_radius = 0.24
                default_wheel_base = 0.54
            else:
                default_wheel_radius = 0.03
                default_wheel_base = 0.1125

            wheel_radius = EnvsetTaskAugmentor._safe_float(params.get("wheel_radius"), fallback=default_wheel_radius)
            wheel_base = EnvsetTaskAugmentor._safe_float(params.get("track_width"), fallback=default_wheel_base)
            forward_speed = EnvsetTaskAugmentor._safe_float(params.get("base_velocity"), fallback=1.0)
            rotation_speed = EnvsetTaskAugmentor._safe_float(params.get("base_turn_rate"), fallback=1.0)

            # NOTE: OmniNavBench expects a velocity controller named "move_by_speed"
            # (taking (forward, lateral, angular)) and a rotation controller named "rotate"
            # for STEP_ACTION mode. Differential drive robots ignore the lateral component.
            move_by_speed_cfg = DifferentialDriveMoveBySpeedControllerCfg(
                name="move_by_speed",
                wheel_radius=wheel_radius,
                wheel_base=wheel_base,
                forward_speed=forward_speed,
                rotation_speed=rotation_speed,
            )
            move_to_point_cfg = MoveToPointBySpeedControllerCfg(
                name="move_to_point",
                forward_speed=forward_speed,
                rotation_speed=rotation_speed,
                threshold=0.2,
                sub_controllers=[move_by_speed_cfg],
            )
            move_along_path_cfg = MoveAlongPathPointsControllerCfg(
                name="move_along_path",
                forward_speed=forward_speed,
                rotation_speed=rotation_speed,
                threshold=0.2,
                sub_controllers=[move_to_point_cfg],
            )
            go_toward_point_cfg = GoTowardPointControllerCfg(
                name="go_toward_point",
                forward_speed=forward_speed,
                rotation_speed=rotation_speed,
                yaw_threshold=0.02,
                dist_threshold=0.02,
                sub_controllers=[move_by_speed_cfg],
            )
            rotate_cfg = RotateControllerCfg(
                name="rotate",
                rotation_speed=rotation_speed,
                threshold=0.02,
                sub_controllers=[move_by_speed_cfg],
            )
            return [
                move_by_speed_cfg,
                move_to_point_cfg,
                move_along_path_cfg,
                go_toward_point_cfg,
                rotate_cfg,
            ]

        # Legged and humanoid robots - use pre-defined configurations with parameter overrides
        if robot_type == "aliengo":
            controllers = []
            # Always expose move_by_speed for velocity-based policies/teleop.
            if aliengo_move_by_speed_cfg is not None:
                controllers.append(EnvsetTaskAugmentor._clone_and_override_controller(aliengo_move_by_speed_cfg, params))
            # Also expose high-level controllers used by STEP_ACTION execution.
            if aliengo_move_to_point_cfg is not None:
                controllers.append(EnvsetTaskAugmentor._clone_and_override_controller(aliengo_move_to_point_cfg, params))
            # Add move_along_path for trajectory-based policies (e.g., OmniNav)
            if aliengo_move_along_path_cfg is not None:
                controllers.append(EnvsetTaskAugmentor._clone_and_override_controller(aliengo_move_along_path_cfg, params))
            if aliengo_go_toward_point_cfg is not None:
                controllers.append(EnvsetTaskAugmentor._clone_and_override_controller(aliengo_go_toward_point_cfg, params))
            if aliengo_rotate_cfg is not None:
                controllers.append(EnvsetTaskAugmentor._clone_and_override_controller(aliengo_rotate_cfg, params))
            if not controllers:
                print(f"[_build_robot_controllers] WARNING: No aliengo controller config available!")
            return controllers

        if robot_type in {"h1", "human"}:
            if h1_move_to_point_cfg is None or h1_move_by_speed_cfg is None:
                raise RuntimeError("H1 controller configs are missing; cannot build controllers.")

            # Expose move_by_speed at the top level so it matches the keyboard / bench action keys.
            controllers = [
                EnvsetTaskAugmentor._clone_and_override_controller(h1_move_by_speed_cfg, params)
            ]
            # Also keep the high-level move_to_point (it uses move_by_speed as a sub-controller).
            controllers.append(EnvsetTaskAugmentor._clone_and_override_controller(h1_move_to_point_cfg, params))
            # Add move_along_path for trajectory-based policies (e.g., OmniNav)
            if h1_move_along_path_cfg is not None:
                controllers.append(EnvsetTaskAugmentor._clone_and_override_controller(h1_move_along_path_cfg, params))
            if h1_go_toward_point_cfg is not None:
                controllers.append(EnvsetTaskAugmentor._clone_and_override_controller(h1_go_toward_point_cfg, params))
            # STEP_ACTION's rotation step dispatches the 'rotate' controller name, so expose it explicitly.
            if h1_rotate_cfg is not None:
                controllers.append(EnvsetTaskAugmentor._clone_and_override_controller(h1_rotate_cfg, params))
            return controllers

        if robot_type == "g1":
            controllers = []
            if g1_move_by_speed_cfg is not None:
                controllers.append(EnvsetTaskAugmentor._clone_and_override_controller(g1_move_by_speed_cfg, params))
            if g1_move_to_point_cfg is not None:
                controllers.append(EnvsetTaskAugmentor._clone_and_override_controller(g1_move_to_point_cfg, params))
            # Add move_along_path for trajectory-based policies (e.g., OmniNav)
            if g1_move_along_path_cfg is not None:
                controllers.append(EnvsetTaskAugmentor._clone_and_override_controller(g1_move_along_path_cfg, params))
            if g1_rotate_cfg is not None:
                controllers.append(EnvsetTaskAugmentor._clone_and_override_controller(g1_rotate_cfg, params))
            return controllers

        if robot_type == "gr1":
            controllers = []
            if gr1_move_by_speed_cfg is not None:
                controllers.append(EnvsetTaskAugmentor._clone_and_override_controller(gr1_move_by_speed_cfg, params))
            if gr1_move_to_point_cfg is not None:
                controllers.append(EnvsetTaskAugmentor._clone_and_override_controller(gr1_move_to_point_cfg, params))
            # Add move_along_path for trajectory-based policies (e.g., OmniNav)
            if gr1_move_along_path_cfg is not None:
                controllers.append(EnvsetTaskAugmentor._clone_and_override_controller(gr1_move_along_path_cfg, params))
            if gr1_rotate_cfg is not None:
                controllers.append(EnvsetTaskAugmentor._clone_and_override_controller(gr1_rotate_cfg, params))
            return controllers

        # Manipulation robots (franka) - no controllers
        if robot_type in {"franka"}:
            return []

        # Fallback: return empty
        return []

    @staticmethod
    def _clone_and_override_controller(controller_cfg, params: Dict[str, Any]):
        """Clone controller config and override parameters from envset.

        Returns: ControllerCfg object (Pydantic model)
        """
        # Deep clone to avoid modifying the original predefined config
        cloned = controller_cfg.model_copy(deep=True)

        EnvsetTaskAugmentor._override_controller_fields(cloned, params)

        return cloned

    @staticmethod
    def _override_controller_fields(controller_cfg, params: Dict[str, Any]) -> None:
        """Override common controller fields (in-place), including sub-controllers."""
        # Override speed parameters from envset if provided
        forward_speed = EnvsetTaskAugmentor._safe_float(params.get("base_velocity"), fallback=None)
        rotation_speed = EnvsetTaskAugmentor._safe_float(params.get("base_turn_rate"), fallback=None)

        if forward_speed is not None and hasattr(controller_cfg, "forward_speed"):
            controller_cfg.forward_speed = forward_speed
        if rotation_speed is not None and hasattr(controller_cfg, "rotation_speed"):
            controller_cfg.rotation_speed = rotation_speed

        # Differential-drive wheel geometry overrides (if the controller config exposes them)
        wheel_radius = EnvsetTaskAugmentor._safe_float(params.get("wheel_radius"), fallback=None)
        wheel_base = EnvsetTaskAugmentor._safe_float(
            params.get("track_width", params.get("wheel_base")), fallback=None
        )
        if wheel_radius is not None and hasattr(controller_cfg, "wheel_radius"):
            controller_cfg.wheel_radius = wheel_radius
        if wheel_base is not None and hasattr(controller_cfg, "wheel_base"):
            controller_cfg.wheel_base = wheel_base

        sub_controllers = getattr(controller_cfg, "sub_controllers", None)
        if not sub_controllers:
            return
        for sub in sub_controllers:
            EnvsetTaskAugmentor._override_controller_fields(sub, params)

    @staticmethod
    def _orientation_from_deg(yaw_deg: float | None):
        if yaw_deg is None:
            return None
        rad = math.radians(yaw_deg)
        half = rad / 2.0
        return (math.cos(half), 0.0, 0.0, math.sin(half))

    @staticmethod
    def _safe_float(value: Any, fallback: float | None) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            if fallback is None:
                return None
            return float(fallback)
