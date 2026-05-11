from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _log_error(msg: str) -> None:
    try:
        import carb  # type: ignore

        carb.log_error(msg)
    except Exception:
        print(f"[ERROR] {msg}")

class VideoRecorder:
    """
    Manages frame capture for a specific camera.
    Uses Omni Replicator to save RGB frames to disk.
    """
    
    def __init__(
        self,
        output_dir: Path,
        camera_prim_path: str,
        resolution: Tuple[int, int] = (1024, 1024),
        writer_name: str = "BasicWriter",
        writer_params: Optional[Dict[str, Any]] = None,
    ):
        self._output_dir = Path(output_dir)
        self._camera_path = str(camera_prim_path)
        self._resolution = tuple(int(v) for v in resolution)
        self._writer_name = str(writer_name)
        self._writer_params = dict(writer_params or {})
        self._frame_count = 0
        self._writer = None
        self._rep = None
        self._setup_writer()

    def _setup_writer(self):
        """Initializes the Replicator Writer."""
        self._output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import omni.replicator.core as rep  # type: ignore
        except Exception as exc:
            raise RuntimeError("Failed to import omni.replicator.core; must run inside Isaac Sim") from exc

        self._rep = rep
        self._writer = rep.WriterRegistry.get(self._writer_name)
        if not self._writer:
            raise RuntimeError(f"Writer not found via Replicator Registry: {self._writer_name}")

        params = {
            "output_dir": str(self._output_dir),
            "rgb": True,
            "semantic_segmentation": False,
            "instance_id_segmentation": False,
        }
        params.update(self._writer_params)
        try:
            self._writer.initialize(**params)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize replicator writer {self._writer_name}: {exc}") from exc

        render_product = rep.create.render_product(self._camera_path, self._resolution)
        self._writer.attach([render_product])

    def capture_frame(self):
        """
        Trigger a write for the current frame.

        When manually stepping physics, call this after `runner.step(..., render=True)`.
        """
        if self._rep is None:
            return
        try:
            # Orchestrator step triggers writer execution once.
            self._rep.orchestrator.step()
        except Exception as exc:
            _log_error(f"[VideoRecorder] orchestrator.step failed: {exc}")
        self._frame_count += 1

    def get_keyframes(self, interval_sec: float) -> List[str]:
        """
        Returns the list of generated file paths based on naming convention.
        """
        if interval_sec <= 0:
            interval_sec = 0.5
        # Best-effort scan; actual naming depends on writer.
        frames = sorted(str(p) for p in self._output_dir.rglob("*.png"))
        return frames

    def shutdown(self):
        """Stops the writer."""
        if self._writer:
            try:
                self._writer.detach()
            except Exception as exc:
                _log_error(f"[VideoRecorder] writer.detach failed: {exc}")
