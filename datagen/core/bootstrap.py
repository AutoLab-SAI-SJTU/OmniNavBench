from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def _log_info(msg: str) -> None:
    try:
        import carb  # type: ignore

        carb.log_info(msg)
    except Exception:
        print(msg)


def _log_error(msg: str) -> None:
    try:
        import carb  # type: ignore

        carb.log_error(msg)
    except Exception:
        print(f"[ERROR] {msg}")


class SceneBootstrapper:
    """
    Handles the initialization of the stage, scene loading, and strict NavMesh validation.
    """

    def __init__(self, scene_usd: Path, strict_navmesh: bool = True, wait_frames: int = 240):
        self._scene_usd = Path(scene_usd)
        self._strict_navmesh = bool(strict_navmesh)
        self._wait_frames = int(wait_frames)
        self._stage = None
        self._navmesh_interface = None

    def bootstrap(self, simulation_app: Optional[Any] = None) -> Any:
        """
        Main entry point to prepare the scene.

        Returns:
            The loaded USD Stage object.
            
        Raises:
            RuntimeError: If scene fails to load or NavMesh is missing.
        """
        self._load_scene(simulation_app=simulation_app)
        self._validate_navmesh()
        return self._stage

    def _load_scene(self, simulation_app: Optional[Any]) -> None:
        """Loads the USD file into the stage."""
        try:
            import omni.usd  # type: ignore
        except Exception as exc:
            raise RuntimeError("Failed to import omni.usd; must run inside Isaac Sim") from exc

        _log_info(f"[DataGen] Opening USD stage: {self._scene_usd}")
        omni.usd.get_context().open_stage(str(self._scene_usd))

        for _ in range(max(1, self._wait_frames)):
            if simulation_app is not None:
                try:
                    simulation_app.update()
                except Exception:
                    pass
            try:
                stage = omni.usd.get_context().get_stage()
            except Exception:
                stage = None
            if stage is not None:
                self._stage = stage
                return

        raise RuntimeError(f"Stage failed to load after {self._wait_frames} frames: {self._scene_usd}")

    def _validate_navmesh(self):
        """
        Strictly checks for existing NavMesh.
        
        Raises:
            RuntimeError: If NavMesh is not found. We DO NOT bake on the fly.
        """
        import omni.anim.navigation.core as nav  # type: ignore

        self._navmesh_interface = nav.acquire_interface()
        navmesh = self._navmesh_interface.get_navmesh()

        if not navmesh:
            if self._strict_navmesh:
                error_msg = (
                    f"[DataGen] Critical Error: No pre-baked NavMesh found in {self._scene_usd}. "
                    "strict_navmesh=True, refusing to bake on the fly."
                )
                _log_error(error_msg)
                raise RuntimeError(error_msg)
            _log_error("[DataGen] No NavMesh found (strict_navmesh=False); continuing without NavMesh.")

        _log_info("[DataGen] NavMesh validated successfully.")

    def get_navmesh_interface(self):
        return self._navmesh_interface
