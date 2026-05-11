"""Uni-NaVid Waypoint HTTP Server for OmniNavBench (Waypoints Output).

This server loads the Uni-NaVid waypoint prediction model and returns
waypoints in robot-centric local frame for go_toward_point controller.

================================================================================
How to Run
================================================================================

1. Start the server (terminal 1, ``uni-navid`` env):

   conda activate uni-navid
   cd /path/to/Uni-NaVid_waypoints
   python -m bench.policy.uninavid_waypoint.uninavid_waypoint_server_points \
       --uninavid_path /path/to/Uni-NaVid_waypoints \
       --model_path /path/to/Uni-NaVid_waypoints/model_zoo/omninav_waypoint_lora \
       --model_base /path/to/Uni-NaVid_waypoints/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2 \
       --port 8002 \
       --debug

2. Run the benchmark (terminal 2, ``isaaclab`` env):

   conda activate isaaclab
   cd $OMNINAV_REPO_ROOT

   python runBench.py \
       --config configs/aliengoh1_test.yaml \
       --scene-root $OMNINAV_SCENE_ROOT \
       --envset $OMNINAV_REPO_ROOT/model_test_episode_aliengo.json \
       --output results/uninavid_waypoint_points_test/ \
       --policy uninavid_waypoint_points \
       --uninavid-waypoint-points-server-url http://localhost:8002 \
       --headless

================================================================================

Response format (/act):
    {
        "waypoints": [[x, y, yaw], ...],  # 5 waypoints in robot-centric frame
        "arrive_probs": [0.1, 0.2, ...],  # arrival probability for each waypoint
        "step": N,
        "inference_time": 0.xx
    }

Coordinate System:
    - Robot-centric local frame: x=forward (positive), y=left (positive)
    - Waypoints are incremental (need cumsum for absolute positions)
    - Yaw is relative to robot's current orientation
"""

from __future__ import annotations

import argparse
import base64
import sys
import os
import io
import threading
import time
from pathlib import Path
from typing import Any, Optional, List

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# ============================================================================
# Global State
# ============================================================================

_agent = None  # UniNaVid_Agent instance
_model_path = None
_model_base = None
_uninavid_path = None
_inference_lock = threading.Lock()
_last_instruction = ""

# Debug settings
_debug_enabled = False
_debug_dir = None
_debug_interval = 10
_step_count = 0
_default_wall_timeout_s: Optional[float] = 300.0
_episode_wall_timeout_s: Optional[float] = None
_episode_start_monotonic: Optional[float] = None


# ============================================================================
# Model Loading
# ============================================================================

def load_waypoint_model(model_path: str, model_base: str, uninavid_path: Path) -> Any:
    """Load Uni-NaVid Waypoint model.

    Args:
        model_path: Path to waypoint model checkpoint
        model_base: Path to full pretrained model
        uninavid_path: Path to Uni-NaVid_waypoints project root

    Returns:
        UniNaVid_Waypoint_Agent instance for waypoint prediction
    """
    print(f"[WaypointServerPoints] Loading waypoint model...")
    print(f"[WaypointServerPoints]   Model path: {model_path}")
    print(f"[WaypointServerPoints]   Base model: {model_base}")
    print(f"[WaypointServerPoints]   Project root: {uninavid_path}")

    # Add to sys.path
    if str(uninavid_path) not in sys.path:
        sys.path.insert(0, str(uninavid_path))
        print(f"[WaypointServerPoints] Added to sys.path: {uninavid_path}")

    # Change working directory
    original_cwd = os.getcwd()
    try:
        os.chdir(str(uninavid_path))
        print(f"[WaypointServerPoints] Changed working directory to: {uninavid_path}")

        # Import UniNaVid_Waypoint_Agent from offline_eval_waypoints.py
        import importlib.util
        offline_eval_path = uninavid_path / "offline_eval_waypoints.py"
        if not offline_eval_path.exists():
            raise FileNotFoundError(f"offline_eval_waypoints.py not found at: {offline_eval_path}")

        spec = importlib.util.spec_from_file_location("offline_eval_waypoints", str(offline_eval_path))
        offline_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(offline_module)
        UniNaVid_Waypoint_Agent = offline_module.UniNaVid_Waypoint_Agent

        print("[WaypointServerPoints] Successfully imported UniNaVid_Waypoint_Agent")

        # Create agent instance
        agent = UniNaVid_Waypoint_Agent(model_path, model_base)

        print("[WaypointServerPoints] Model loaded successfully")
        return agent

    finally:
        os.chdir(original_cwd)


# ============================================================================
# Debug Utilities
# ============================================================================

def _save_debug_image(rgb_array: np.ndarray, waypoints: List[List[float]], step: int):
    """Save debug image with waypoints visualization."""
    if not _debug_enabled or _debug_dir is None:
        return

    try:
        import cv2

        img = rgb_array.copy()
        h, w = img.shape[:2]

        # Robot position (bottom center)
        origin_x, origin_y = w // 2, int(h * 0.9)
        scale = 50  # pixels per meter

        # Draw robot position
        cv2.circle(img, (origin_x, origin_y), 6, (0, 0, 255), -1)

        # Colors for waypoints
        colors = [(0, 255, 0), (0, 255, 255), (0, 165, 255), (255, 0, 255), (255, 0, 0)]

        # Accumulate positions (waypoints are incremental)
        acc_x, acc_y = 0.0, 0.0
        prev_pt = (origin_x, origin_y)

        for i, wp in enumerate(waypoints):
            if len(wp) < 2:
                continue

            acc_x += float(wp[0])  # forward
            acc_y += float(wp[1])  # left

            # Convert to image coordinates: x=forward (up), y=left (left)
            img_x = int(origin_x - acc_y * scale)
            img_y = int(origin_y - acc_x * scale)
            img_x = max(0, min(w-1, img_x))
            img_y = max(0, min(h-1, img_y))

            color = colors[i % len(colors)]
            cv2.line(img, prev_pt, (img_x, img_y), color, 2)
            cv2.circle(img, (img_x, img_y), 5, color, -1)
            cv2.putText(img, str(i+1), (img_x+5, img_y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            prev_pt = (img_x, img_y)

        # Add step info
        cv2.putText(img, f"Step {step}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Save image
        save_path = os.path.join(_debug_dir, f"server_step_{step:04d}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    except Exception as e:
        print(f"[WaypointServerPoints] Debug image save failed: {e}")


# ============================================================================
# HTTP Endpoints
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": _agent is not None,
        "model_path": _model_path,
        "model_base": _model_base,
        "model_type": "waypoint_points",
        "debug_enabled": _debug_enabled,
        "wall_timeout_s": _episode_wall_timeout_s if _episode_wall_timeout_s is not None else _default_wall_timeout_s,
    })


@app.route('/reset', methods=['POST'])
def reset():
    """Reset agent state for a new episode."""
    global _agent, _last_instruction, _step_count
    global _episode_wall_timeout_s, _episode_start_monotonic

    if _agent is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json() or {}
        instruction = data.get("instruction", "")
        task_type = data.get("task_type", "vln")
        wall_timeout_raw = data.get("wall_timeout_s")
        wall_timeout_s = _default_wall_timeout_s if wall_timeout_raw is None else float(wall_timeout_raw)

        with _inference_lock:
            _last_instruction = instruction
            _step_count = 0
            _episode_wall_timeout_s = wall_timeout_s if wall_timeout_s and wall_timeout_s > 0 else None
            _episode_start_monotonic = time.monotonic()
            _agent.reset(task_type=task_type)

        print(
            f"[WaypointServerPoints] Reset: instruction='{instruction[:50]}...' "
            f"task_type={task_type} wall_timeout_s={_episode_wall_timeout_s}"
        )

        return jsonify({
            "status": "reset",
            "instruction": instruction,
            "task_type": task_type,
            "wall_timeout_s": _episode_wall_timeout_s,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/act', methods=['POST'])
def act():
    """Get waypoint prediction from model.

    Request body:
        {
            "instruction": "navigation instruction",
            "image": "base64_encoded_png_image",
            "image_shape": [height, width, 3]
        }

    Response:
        {
            "waypoints": [[x, y, yaw], ...],  # robot-centric, incremental
            "arrive_probs": [0.1, 0.2, ...],
            "step": int,
            "inference_time": float
        }
    """
    global _agent, _last_instruction, _step_count

    if _agent is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        instruction = data.get("instruction", "")
        image_base64 = data.get("image")
        image_shape = data.get("image_shape", [480, 640, 3])

        if not image_base64:
            return jsonify({"error": "No image provided"}), 400

        timeout_result = _build_timeout_response()
        if timeout_result is not None:
            return jsonify(timeout_result)

        # Decode image
        rgb_array = _decode_image(image_base64, image_shape)

        inference_time = 0.0
        waypoints = []
        arrive_probs = []

        with _inference_lock:
            _step_count += 1

            # Update instruction if changed
            if instruction and instruction != _last_instruction:
                _last_instruction = instruction

            start_time = time.time()

            # Call agent.act() - expects waypoint output
            result = _agent.act({
                "instruction": instruction,
                "observations": rgb_array
            })

            inference_time = time.time() - start_time

            # Extract waypoints from result
            # Expected format: {"waypoints": [[x,y,yaw], ...], "arrive_probs": [...]}
            waypoints = result.get("waypoints", [])
            arrive_probs = result.get("arrive_probs", [])

            # If model returns empty waypoints, create a default "stop" response
            if not waypoints:
                print(f"[WaypointServerPoints] Step {_step_count}: No waypoints returned, signaling stop")
                return jsonify({
                    "waypoints": [],
                    "arrive_probs": [1.0],  # High arrive prob to trigger stop
                    "step": _step_count,
                    "inference_time": inference_time,
                    "stop": True,
                    "timed_out": False,
                    "reason": "no waypoints returned",
                })

        # Log waypoints
        if waypoints:
            first_wp = waypoints[0]
            print(f"[WaypointServerPoints] Step {_step_count}: first_waypoint=({first_wp[0]:.2f}, {first_wp[1]:.2f}), "
                  f"arrive_prob={arrive_probs[0] if arrive_probs else 'N/A':.3f}")

        # Debug output
        if _step_count % _debug_interval == 0:
            _save_debug_image(rgb_array, waypoints, _step_count)

        return jsonify({
            "waypoints": waypoints,
            "arrive_probs": arrive_probs,
            "step": _step_count,
            "inference_time": inference_time,
            "stop": False,
            "timed_out": False,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Helper Functions
# ============================================================================

def _decode_image(image_base64: str, image_shape: list) -> np.ndarray:
    """Decode base64 image to numpy array (uint8, RGB)."""
    img_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    rgb_array = np.array(img)
    if rgb_array.dtype != np.uint8:
        rgb_array = rgb_array.astype(np.uint8)
    return rgb_array


def _build_timeout_response() -> Optional[dict[str, Any]]:
    if _episode_wall_timeout_s is None or _episode_start_monotonic is None:
        return None
    elapsed_s = time.monotonic() - _episode_start_monotonic
    if elapsed_s < _episode_wall_timeout_s:
        return None
    reason = f"server wall timeout reached ({elapsed_s:.2f}s >= {_episode_wall_timeout_s:.2f}s)"
    print(f"[WaypointServerPoints] {reason}")
    return {
        "waypoints": [],
        "arrive_probs": [],
        "step": _step_count,
        "inference_time": 0.0,
        "stop": True,
        "timed_out": True,
        "reason": reason,
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Uni-NaVid Waypoint HTTP Server (Points Output)")
    parser.add_argument("--model_path", type=str,
                       default="model_zoo/omninav_waypoint_lora",
                       help="Path to waypoint model checkpoint")
    parser.add_argument("--model_base", type=str,
                       default="model_zoo/uninavid-7b-full-224-video-fps-1-grid-2",
                       help="Path to full pretrained model")
    parser.add_argument("--uninavid_path", type=str, required=True,
                       help="Path to Uni-NaVid_waypoints project root")
    parser.add_argument("--port", type=int, default=8002, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--debug_dir", type=str, default="debug_waypoint_server_points",
                       help="Directory for debug outputs")
    parser.add_argument("--debug_interval", type=int, default=10,
                       help="Save debug info every N steps")
    parser.add_argument("--wall-timeout-s", type=float, default=300.0,
                       help="Wall-clock timeout per episode/session")

    args = parser.parse_args()

    global _agent, _model_path, _model_base, _uninavid_path
    global _debug_enabled, _debug_dir, _debug_interval, _default_wall_timeout_s

    _model_path = args.model_path
    _model_base = args.model_base
    _uninavid_path = Path(args.uninavid_path)
    _debug_enabled = args.debug
    _debug_interval = args.debug_interval
    _default_wall_timeout_s = args.wall_timeout_s

    if _debug_enabled:
        _debug_dir = args.debug_dir
        os.makedirs(_debug_dir, exist_ok=True)
        print(f"[WaypointServerPoints] Debug enabled, saving to: {_debug_dir}")

    if not _uninavid_path.exists():
        raise FileNotFoundError(f"Uni-NaVid path does not exist: {_uninavid_path}")

    try:
        _agent = load_waypoint_model(args.model_path, args.model_base, _uninavid_path)
        print(f"[WaypointServerPoints] Model loaded successfully")
    except Exception as e:
        print(f"[WaypointServerPoints] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"[WaypointServerPoints] Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
