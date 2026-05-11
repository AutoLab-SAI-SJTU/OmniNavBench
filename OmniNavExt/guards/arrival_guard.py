try:
    import carb
    import omni.usd
    import omni.kit.app
    from pxr import UsdGeom
    ISAAC_AVAILABLE = True
except ImportError:
    carb = None
    omni = None
    ISAAC_AVAILABLE = False

from typing import Optional

from OmniNavExt.envset.agent_manager import AgentManager
from OmniNavExt.envset.settings import PrimPaths
from OmniNavExt.envset.stage_util import UnitScaleService


class ArrivalGuard:
    """External guard that finishes character GoTo-like commands once within a meter-based tolerance.

    Design goals:
    - Only active for GRScenes (or centimeter-scale stages, mpu <= 0.02) to avoid affecting MP3D/Builtin.
    - Non-invasive: does not modify omni.anim.people internals; uses Behavior API end_current_command().
    - Horizontal-distance check in meters; target parsed from the current command parameters.
    - Per-frame subscription via update event stream (no asyncio tasks to avoid Script Editor conflicts).
    """

    def __init__(self):
        self._sub = None
        self._enabled = False
        self._tol_m = 0.5
        self._category = ""

    def is_active(self) -> bool:
        return self._enabled and self._sub is not None

    def enable_if_grscenes(self, scene_category: Optional[str], tolerance_m: float = 0.5):
        """Enable guard only if scene category is GRScenes (do not affect MP3D / built-in scenes)."""
        category_lc = (scene_category or "").strip().lower()
        if category_lc != "grscenes":
            return

        self._tol_m = float(tolerance_m)
        self._category = scene_category or ""
        if self._sub is None:
            app = omni.kit.app.get_app()
            self._sub = app.get_update_event_stream().create_subscription_to_pop(
                self._on_update, name="ira/arrival_guard/grscenes_only"
            )
        self._enabled = True
        carb.log_info(
            f"[ArrivalGuard] Enabled for category='{self._category}' with tolerance={self._tol_m} m"
        )

    def disable(self):
        if self._sub is not None:
            try:
                if hasattr(self._sub, "unsubscribe"):
                    self._sub.unsubscribe()
            except Exception:
                pass
            self._sub = None
        self._enabled = False

    # --------------------- internals ---------------------
    def _on_update(self, _e):
        if not self._enabled:
            return
        mgr = AgentManager.get_instance()
        parent_path = PrimPaths.characters_parent_path()
        try:
            mpu = float(UnitScaleService.get_meters_per_unit())
        except Exception:
            try:
                stage = omni.usd.get_context().get_stage()
                mpu = float(UsdGeom.GetStageMetersPerUnit(stage)) if stage else 1.0
            except Exception:
                mpu = 1.0

        # Iterate all registered agents and handle only character behaviors
        for agent_name in list(mgr.get_all_agent_names()):
            inst = mgr.get_agent_script_instance_by_name(agent_name)
            if inst is None:
                continue
            prim_path = str(getattr(inst, "prim_path", ""))
            if not prim_path.startswith(parent_path):
                # not a character
                continue

            cur_cmd = getattr(inst, "current_command", None)
            if cur_cmd is None:
                continue

            # Determine command name (prefer API, fallback to raw list)
            cmd_name = None
            if hasattr(cur_cmd, "get_command_name"):
                try:
                    cmd_name = str(cur_cmd.get_command_name())
                except Exception:
                    cmd_name = None
            if not cmd_name:
                try:
                    raw = getattr(cur_cmd, "command", None)
                    if isinstance(raw, (list, tuple)) and len(raw) > 0:
                        cmd_name = str(raw[0])
                except Exception:
                    cmd_name = None

            if not cmd_name:
                continue
            cmd_name_lc = cmd_name.lower()
            # Only act on GoTo-like commands
            if not (cmd_name_lc == "goto" or cmd_name_lc.startswith("gotoblend")):
                continue

            # Extract target from command tokens: ['GoTo*', x, y, z, ...]
            target = None
            try:
                tokens = getattr(cur_cmd, "command", None)
                if isinstance(tokens, (list, tuple)) and len(tokens) >= 4:
                    tx = float(tokens[1]); ty = float(tokens[2]); tz = float(tokens[3])
                    target = (tx, ty, tz)
            except Exception:
                target = None
            if target is None:
                continue

            # Current position (from Behavior API, fallback to USD xform)
            try:
                cur = inst.get_current_position()
                px, py = float(cur[0]), float(cur[1])
            except Exception:
                try:
                    prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
                    mat = omni.usd.get_world_transform_matrix(prim)
                    pos = mat.ExtractTranslation()
                    px, py = float(pos[0]), float(pos[1])
                except Exception:
                    continue

            dx = px - float(target[0]); dy = py - float(target[1])
            dist_m = ((dx * dx + dy * dy) ** 0.5) * mpu
            if dist_m <= self._tol_m:
                # Cleanly finish the current command; next update will pop it.
                try:
                    if hasattr(inst, "end_current_command"):
                        inst.end_current_command()
                        carb.log_info(
                            f"[ArrivalGuard] {agent_name} reached within {self._tol_m} m; ending current command."
                        )
                except Exception:
                    pass
