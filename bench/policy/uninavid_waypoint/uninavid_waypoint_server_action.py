"""Uni-NaVid Action HTTP Server for OmniNavBench.

This server loads the Uni-NaVid action prediction model (with LoRA support)
and exposes HTTP endpoints for inference.

================================================================================
How to Run
================================================================================

1. Start the server (terminal 1, ``uni-navid`` env):

   conda activate uni-navid
   cd /path/to/Uni-NaVid_waypoints
   python -m bench.policy.uninavid_waypoint.uninavid_waypoint_server_action \
       --uninavid_path /path/to/Uni-NaVid_waypoints \
       --model_path /path/to/Uni-NaVid_waypoints/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2 \
       --lora_path /path/to/Uni-NaVid_waypoints/model_zoo/omninav_action_lora \
       --port 8001 \
       --debug

2. Run the benchmark (terminal 2, ``isaaclab`` env):

   conda activate isaaclab
   cd $OMNINAV_REPO_ROOT

   # Single JSON file
   python runBench.py \
       --config configs/aliengoh1_test.yaml \
       --scene-root $OMNINAV_SCENE_ROOT \
       --envset $OMNINAV_REPO_ROOT/model_test_episode_aliengo.json \
       --output results/uninavid_action_test/ \
       --policy uninavid_waypoint \
       --uninavid-waypoint-server-url http://localhost:8001 \
       --headless

   # Or pass a directory (all JSON files inside are iterated)
   python runBench.py \
       --config configs/aliengoh1_test.yaml \
       --scene-root $OMNINAV_SCENE_ROOT \
       --envset /path/to/dataset/dog \
       --output results/uninavid_action_test/ \
       --policy uninavid_waypoint \
       --uninavid-waypoint-server-url http://localhost:8001 \
       --headless

Optional arguments:
    --debug              Enable debug mode (saves input images).
    --debug_interval N   Save a debug image every N steps (default: 10).
    --debug_dir DIR      Output directory for debug images (default: debug_waypoint_server).

================================================================================

Model Loading:
    - model_path: Base/full model checkpoint directory
    - lora_path: LoRA checkpoint directory (contains adapter_model.bin, non_lora_trainables.bin)

Endpoints:
    GET  /health  - Health check, returns {"status": "ok", "model_loaded": true}
    POST /reset   - Reset episode state, body: {"instruction": "...", "task_type": "vln"}
    POST /act     - Get action prediction, body: {"instruction": "...", "image": base64, "image_shape": [H,W,3]}

Response format (/act):
    {
        "action": "forward",  # one of: forward, left, right, wait, stop
        "actions": ["forward", "left", ...],  # full action sequence from model
        "step": N,
        "inference_time": 0.xx
    }
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
from typing import Any, Optional

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# ============================================================================
# Global State
# ============================================================================

_agent = None  # UniNaVid_Agent instance
_model_path = None
_lora_path = None
_uninavid_path = None
_inference_lock = threading.Lock()
_last_instruction = ""

# Debug settings
_debug_enabled = False
_debug_dir = None
_debug_interval = 10  # Save debug info every N steps
_step_count = 0
_default_wall_timeout_s: Optional[float] = 300.0
_episode_wall_timeout_s: Optional[float] = None
_episode_start_monotonic: Optional[float] = None


# ============================================================================
# Model Loading
# ============================================================================

def load_waypoint_model(model_path: str, uninavid_path: Path, lora_path: Optional[str] = None) -> Any:
    """Load Uni-NaVid Action model by importing UniNaVid_Agent.

    Args:
        model_path: Base/full model checkpoint directory.
        uninavid_path: Path to Uni-NaVid_waypoints project root
        lora_path: Optional LoRA checkpoint directory.

    Returns:
        UniNaVid_Agent instance
    """
    print(f"[WaypointServer] Loading action model...")
    print(f"[WaypointServer]   Model path: {model_path}")
    print(f"[WaypointServer]   LoRA checkpoint: {lora_path}")
    print(f"[WaypointServer]   Project root: {uninavid_path}")

    # Add to sys.path
    if str(uninavid_path) not in sys.path:
        sys.path.insert(0, str(uninavid_path))
        print(f"[WaypointServer] Added to sys.path: {uninavid_path}")

    # Change working directory
    original_cwd = os.getcwd()
    try:
        os.chdir(str(uninavid_path))
        print(f"[WaypointServer] Changed working directory to: {uninavid_path}")

        # Import UniNaVid_Agent from offline_eval_uninavid.py
        import importlib.util
        offline_eval_path = uninavid_path / "offline_eval_uninavid.py"
        if not offline_eval_path.exists():
            raise FileNotFoundError(f"offline_eval_uninavid.py not found at: {offline_eval_path}")

        spec = importlib.util.spec_from_file_location("offline_eval_uninavid", str(offline_eval_path))
        offline_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(offline_module)
        UniNaVid_Agent = offline_module.UniNaVid_Agent

        print("[WaypointServer] Successfully imported UniNaVid_Agent")

        if lora_path:
            # Same runtime path as UniNaVidServer: inherit act()/predict_inference()
            # and only replace model initialization to load base + LoRA.
            class UniNaVidLoRAAgent(UniNaVid_Agent):
                def __init__(self, base_model_path: str, adapter_path: str):
                    print("Initialize UniNaVid Action LoRA")

                    self.conv_mode = "vicuna_v1"
                    self.model_name = offline_module.get_model_name_from_path(base_model_path)
                    self.tokenizer, self.model, self.image_processor, self.context_len = (
                        offline_module.load_pretrained_model(
                            adapter_path,
                            base_model_path,
                            self.model_name,
                        )
                    )

                    assert self.image_processor is not None

                    print("Initialization Complete")

                    self.promt_template = (
                        "Imagine you are a robot programmed for navigation tasks. "
                        "You have been given a video of historical observations and "
                        "an image of the current observation <image>. Your assigned "
                        "task is: '{}'. Analyze this series of images to determine "
                        "your next four actions. The predicted action should be one "
                        "of the following: forward, left, right, wait, or stop."
                    )
                    self.rgb_list = []
                    self.count_id = 0
                    self.reset()

            agent = UniNaVidLoRAAgent(base_model_path=model_path, adapter_path=lora_path)
        else:
            agent = UniNaVid_Agent(model_path)

        print("[WaypointServer] Model loaded successfully")
        return agent

    finally:
        os.chdir(original_cwd)


# ============================================================================
# Debug Utilities
# ============================================================================

def _save_debug_image(rgb_array: np.ndarray, action: str, step: int):
    """Save debug image with action annotation."""
    if not _debug_enabled or _debug_dir is None:
        return

    try:
        import cv2

        # Draw action on image
        img = rgb_array.copy()
        h, w = img.shape[:2]

        # Add action text
        cv2.putText(img, f"Step {step}: {action}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Save image
        save_path = os.path.join(_debug_dir, f"server_step_{step:04d}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    except Exception as e:
        print(f"[WaypointServer] Debug image save failed: {e}")


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
        "lora_path": _lora_path,
        "model_type": "waypoint",
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
            f"[WaypointServer] Reset: instruction='{instruction}...' "
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
    """Get action prediction from model.

    Each call runs inference and returns the first valid action.

    Request body:
        {
            "instruction": "navigation instruction",
            "image": "base64_encoded_png_image",
            "image_shape": [height, width, 3]
        }

    Response:
        {
            "action": "forward",  # first action: forward, left, right, wait, stop
            "actions": ["forward", "left", ...],  # full action sequence from model
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

        # Decode image (needed for model inference)
        rgb_array = _decode_image(image_base64, image_shape)

        inference_time = 0.0
        all_actions = []

        with _inference_lock:
            _step_count += 1

            # Update instruction if changed
            if instruction and instruction != _last_instruction:
                _last_instruction = instruction

            start_time = time.time()

            # Call agent.act()
            result = _agent.act({
                "instruction": instruction,
                "observations": rgb_array
            })

            inference_time = time.time() - start_time

            # Extract action list from result
            all_actions = result.get("actions", [])

            # Filter all valid actions
            valid_actions = []
            for act_str in all_actions:
                act_str = act_str.lower().strip()
                if act_str in ["forward", "left", "right", "wait", "stop"]:
                    valid_actions.append(act_str)

            # First valid action for backward compatibility
            current_action = valid_actions[0] if valid_actions else "wait"

        # Always print action for debugging
        print(f"[WaypointServer] Step {_step_count}: action={current_action}, valid_actions={valid_actions}")

        # Debug output (every N steps)
        if _step_count % _debug_interval == 0:
            _save_debug_image(rgb_array, current_action, _step_count)

        return jsonify({
            "action": current_action,
            "actions": valid_actions if valid_actions else [current_action],
            "step": _step_count,
            "inference_time": inference_time,
            "stop": current_action == "stop",
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
    print(f"[WaypointServer] {reason}")
    return {
        "action": "stop",
        "actions": [],
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
    parser = argparse.ArgumentParser(description="Uni-NaVid Waypoint HTTP Server")
    parser.add_argument("--model_path", type=str,
                       default="model_zoo/uninavid-7b-full-224-video-fps-1-grid-2",
                       help="Path to Uni-NaVid base/full model checkpoint")
    parser.add_argument("--lora_path", type=str,
                       default=None,
                       help="Optional path to LoRA checkpoint directory")
    parser.add_argument("--uninavid_path", type=str, required=True,
                       help="Path to Uni-NaVid_waypoints project root")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--debug_dir", type=str, default="debug_waypoint_server",
                       help="Directory for debug outputs")
    parser.add_argument("--debug_interval", type=int, default=10,
                       help="Save debug info every N steps")
    parser.add_argument("--wall-timeout-s", type=float, default=300.0,
                       help="Wall-clock timeout per episode/session")

    args = parser.parse_args()

    global _agent, _model_path, _lora_path, _uninavid_path
    global _debug_enabled, _debug_dir, _debug_interval, _default_wall_timeout_s

    _model_path = args.model_path
    _lora_path = args.lora_path
    _uninavid_path = Path(args.uninavid_path)
    _debug_enabled = args.debug
    _debug_interval = args.debug_interval
    _default_wall_timeout_s = args.wall_timeout_s

    if _debug_enabled:
        _debug_dir = args.debug_dir
        os.makedirs(_debug_dir, exist_ok=True)
        print(f"[WaypointServer] Debug enabled, saving to: {_debug_dir}")

    if not _uninavid_path.exists():
        raise FileNotFoundError(f"Uni-NaVid path does not exist: {_uninavid_path}")

    try:
        _agent = load_waypoint_model(args.model_path, _uninavid_path, lora_path=args.lora_path)
        print(f"[WaypointServer] Model loaded successfully")
    except Exception as e:
        print(f"[WaypointServer] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"[WaypointServer] Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
