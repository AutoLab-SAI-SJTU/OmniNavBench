"""MTU3D HTTP Server (PQ3DModel)

Run this server inside the `conda activate mtu3d` environment.

This server wraps `MTU3D/hm3d-online/data_utils.py::PQ3DModel` and exposes a strict
HTTP API for OmniNavBench policies.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)

_model: Any | None = None
_model_lock = threading.Lock()


def _b64_to_bytes(data_b64: str) -> bytes:
    try:
        return base64.b64decode(data_b64.encode("utf-8"), validate=True)
    except Exception as e:
        raise ValueError("Invalid base64 payload") from e


def _decode_rgb(rgb_b64: str) -> np.ndarray:
    """Decode RGB image from base64-encoded PNG/JPEG bytes to uint8 (H, W, 3)."""
    raw = _b64_to_bytes(rgb_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _decode_depth_npy(depth_npy_b64: str) -> np.ndarray:
    """Decode depth map from base64-encoded .npy bytes to float32 (H, W), meters."""
    raw = _b64_to_bytes(depth_npy_b64)
    arr = np.load(io.BytesIO(raw), allow_pickle=False)
    if arr.ndim != 2:
        raise ValueError(f"depth must be a 2D array, got shape={arr.shape}")
    return np.asarray(arr, dtype=np.float32)


@dataclass
class _SensorState:
    position: np.ndarray
    rotation: Any  # quaternion.quaternion


@dataclass
class _AgentState:
    sensor_states: Dict[str, _SensorState]
    position: np.ndarray


def _make_agent_state(sensor_pos: List[float], sensor_quat_wxyz: List[float]) -> _AgentState:
    if len(sensor_pos) != 3:
        raise ValueError("sensor_pos must be [x, y, z]")
    if len(sensor_quat_wxyz) != 4:
        raise ValueError("sensor_quat_wxyz must be [w, x, y, z]")
    try:
        import quaternion  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError("Missing dependency: quaternion (numpy-quaternion)") from e

    pos = np.array(sensor_pos, dtype=np.float32)
    w, x, y, z = [float(v) for v in sensor_quat_wxyz]
    rot = quaternion.quaternion(w, x, y, z)
    sensor_state = _SensorState(position=pos, rotation=rot)
    return _AgentState(sensor_states={"color_sensor": sensor_state}, position=pos)


def load_mtu3d_model(mtu3d_root: Path, stage1_dir: Path, stage2_dir: Path, min_decision_num: int) -> Any:
    """Load PQ3DModel with MTU3D's expected working directory/layout."""
    hm3d_online = mtu3d_root / "hm3d-online"
    if not hm3d_online.exists():
        raise FileNotFoundError(f"Expected MTU3D hm3d-online at: {hm3d_online}")
    if not stage1_dir.exists():
        raise FileNotFoundError(f"stage1_dir not found: {stage1_dir}")
    if not stage2_dir.exists():
        raise FileNotFoundError(f"stage2_dir not found: {stage2_dir}")

    # MTU3D code uses relative paths like:
    # - FastSAM('./hm3d-online/FastSAM/FastSAM-x.pt')
    # - hydra config_path = '../configs/...'
    # These assume CWD is `hm3d-online`.
    os.chdir(str(hm3d_online))
    sys.path.insert(0, str(hm3d_online))
    sys.path.insert(0, str(mtu3d_root))  # Also add MTU3D root for data imports

    from data_utils import PQ3DModel  # type: ignore

    return PQ3DModel(stage1_dir=str(stage1_dir), stage2_dir=str(stage2_dir), min_decision_num=min_decision_num)


@app.get("/health")
def health():
    return jsonify({"ok": True, "loaded": _model is not None})


@app.post("/reset")
def reset():
    if _model is None:
        raise RuntimeError("Model not loaded")
    with _model_lock:
        _model.reset()
    return jsonify({"ok": True})


@app.post("/decision")
def decision():
    """Run one MTU3D decision.

    Request JSON (strict):
      - sentence: str
      - decision_num: int
      - frontier_waypoints: list[[x,y,z], ...]
      - frames: list of:
          - rgb_b64: base64(PNG/JPEG bytes)
          - depth_npy_b64: base64(.npy bytes), depth in meters
          - sensor_pos: [x,y,z]
          - sensor_quat_wxyz: [w,x,y,z]
    """
    if _model is None:
        raise RuntimeError("Model not loaded")

    payload = request.get_json(force=True, silent=False)
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")

    sentence = payload.get("sentence")
    if not isinstance(sentence, str) or not sentence:
        raise ValueError("Missing required field: sentence (non-empty string)")

    decision_num = payload.get("decision_num")
    try:
        decision_num_i = int(decision_num)
    except Exception as e:
        raise ValueError("Missing/invalid field: decision_num (int)") from e

    frontier_waypoints = payload.get("frontier_waypoints")
    if frontier_waypoints is None:
        frontier_waypoints = []
    if not isinstance(frontier_waypoints, list):
        raise ValueError("frontier_waypoints must be a list")

    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Missing required field: frames (non-empty list)")

    camera_params_list = payload.get("camera_params_list")
    if camera_params_list is None:
        camera_params_list = []
    if not isinstance(camera_params_list, list):
        raise ValueError("camera_params_list must be a list")

    print(f"[MTU3DServer] Decision #{decision_num_i}: Received {len(camera_params_list)} camera params")

    color_list: List[np.ndarray] = []
    depth_list: List[np.ndarray] = []
    agent_state_list: List[_AgentState] = []

    for i, fr in enumerate(frames):
        if not isinstance(fr, dict):
            raise ValueError(f"frames[{i}] must be an object")
        rgb_b64 = fr.get("rgb_b64")
        depth_npy_b64 = fr.get("depth_npy_b64")
        sensor_pos = fr.get("sensor_pos")
        sensor_quat = fr.get("sensor_quat_wxyz")
        if not isinstance(rgb_b64, str) or not rgb_b64:
            raise ValueError(f"frames[{i}].rgb_b64 must be a non-empty base64 string")
        if not isinstance(depth_npy_b64, str) or not depth_npy_b64:
            raise ValueError(f"frames[{i}].depth_npy_b64 must be a non-empty base64 string")
        if not isinstance(sensor_pos, list):
            raise ValueError(f"frames[{i}].sensor_pos must be [x,y,z]")
        if not isinstance(sensor_quat, list):
            raise ValueError(f"frames[{i}].sensor_quat_wxyz must be [w,x,y,z]")

        color_list.append(_decode_rgb(rgb_b64))
        depth_list.append(_decode_depth_npy(depth_npy_b64))
        agent_state_list.append(_make_agent_state(sensor_pos, sensor_quat))

    frontier_np: List[np.ndarray] = []
    for j, wp in enumerate(frontier_waypoints):
        if not (isinstance(wp, (list, tuple)) and len(wp) == 3):
            raise ValueError(f"frontier_waypoints[{j}] must be [x,y,z]")
        frontier_np.append(np.array([float(wp[0]), float(wp[1]), float(wp[2])], dtype=np.float32))

    with _model_lock:
        target_position, is_object_decision = _model.decision(
            color_list=color_list,
            depth_list=depth_list,
            agent_state_list=agent_state_list,
            frontier_waypoints=frontier_np,
            sentence=sentence,
            decision_num=decision_num_i,
            camera_params_list=camera_params_list,
        )

    decision_type = "Object Approach" if is_object_decision else "Frontier Exploration"
    target_coords = [float(target_position[0]), float(target_position[1]), float(target_position[2])]

    print(f"[MTU3DServer] Decision #{decision_num_i}: {decision_type}")
    print(f"  Target: ({target_coords[0]:.3f}, {target_coords[1]:.3f}, {target_coords[2]:.3f})")
    print(f"  Sentence: '{sentence}'")

    return jsonify({
        "target_position": target_coords,
        "is_object_decision": bool(is_object_decision),
        "decision_type": decision_type
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="MTU3D PQ3DModel HTTP server")
    parser.add_argument("--mtu3d_path", type=str, required=True, help="Path to MTU3D repo root")
    parser.add_argument("--stage1_dir", type=str, required=True, help="Path to stage1 checkpoint dir (contains pytorch_model.bin)")
    parser.add_argument("--stage2_dir", type=str, required=True, help="Path to stage2 checkpoint dir (contains pytorch_model.bin)")
    parser.add_argument("--min_decision_num", type=int, default=3, help="MTU3D min decision num")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True,
                        help="Server port (required; pick any free TCP port)")
    args = parser.parse_args()

    mtu3d_root = Path(args.mtu3d_path).expanduser().resolve()
    stage1_dir = Path(args.stage1_dir).expanduser().resolve()
    stage2_dir = Path(args.stage2_dir).expanduser().resolve()

    global _model
    print(f"[MTU3DServer] Loading MTU3D from: {mtu3d_root}")
    _model = load_mtu3d_model(mtu3d_root, stage1_dir, stage2_dir, min_decision_num=int(args.min_decision_num))
    print("[MTU3DServer] ✓ Model loaded")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

