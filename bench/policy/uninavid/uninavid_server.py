"""Uni-NaVid HTTP Server for OmniNavBench.

This server loads the original Uni-NaVid model and exposes HTTP endpoints
for inference. It is designed to run in the Uni-NaVid conda environment.

Usage:
    conda activate uninavid
    cd /path/to/Uni-NaVid
    python -m bench.policy.uninavid.uninavid_server \
        --model_path /path/to/Uni-NaVid/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2 \
        --lora_path /path/to/Uni-NaVid/model_zoo/vln_action_text_lora_xxx \
        --uninavid_path /path/to/Uni-NaVid \
        --port 8000

Endpoints:
    GET  /health  - Health check
    POST /reset   - Reset episode state
    POST /act     - Get navigation action

The server directly uses the original UniNaVid_Agent class from
offline_eval_uninavid.py, preserving the original inference logic.
"""

from __future__ import annotations

import argparse
import base64
from collections import deque
import sys
import os
import io
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# ============================================================================
# Global State
# ============================================================================

_model = None  # UniNaVid_Agent instance
_model_path = None
_lora_path = None
_uninavid_path = None
_inference_lock = threading.Lock()

# Pending action queue.
# Default queue length is 1 to match the real-robot client, which only executes actions[0].
_pending_actions = deque()
_last_pred_actions = []
_last_instruction = ""
_MAX_PENDING_ACTIONS = 1

# Valid actions
_ALLOWED_ACTIONS = {"forward", "left", "right", "stop"}
_ACTION_TO_INDEX = {"stop": 0, "forward": 1, "left": 2, "right": 3}




# ============================================================================
# Model Loading (Directly imports original UniNaVid_Agent)
# ============================================================================

def _resolve_model_arg(path_value: Optional[str], uninavid_path: Path) -> Optional[str]:
    """Resolve a CLI path argument relative to the Uni-NaVid project root."""
    if path_value is None:
        return None
    resolved = Path(path_value).expanduser()
    if not resolved.is_absolute():
        resolved = uninavid_path / resolved
    return str(resolved)


def load_uninavid_model(model_path: str, uninavid_path: Path, lora_path: Optional[str] = None) -> Any:
    """Load Uni-NaVid model by importing original UniNaVid_Agent.

    This function dynamically imports the UniNaVid_Agent class from
    offline_eval_uninavid.py to preserve the original implementation.

    Args:
        model_path: Path to base model checkpoint (relative or absolute)
        uninavid_path: Path to Uni-NaVid project root
        lora_path: Optional path to LoRA checkpoint directory

    Returns:
        UniNaVid_Agent instance
    """
    print(f"[UniNaVidServer] Loading model from: {model_path}")
    if lora_path:
        print(f"[UniNaVidServer] Loading LoRA from: {lora_path}")
    print(f"[UniNaVidServer] Uni-NaVid path: {uninavid_path}")

    # Add Uni-NaVid to sys.path
    if str(uninavid_path) not in sys.path:
        sys.path.insert(0, str(uninavid_path))
        print(f"[UniNaVidServer] Added to sys.path: {uninavid_path}")

    # Change working directory to Uni-NaVid (for relative paths in model)
    original_cwd = os.getcwd()
    try:
        os.chdir(str(uninavid_path))
        print(f"[UniNaVidServer] Changed working directory to: {uninavid_path}")

        # Dynamically import UniNaVid_Agent from offline_eval_uninavid.py
        offline_eval_path = uninavid_path / "offline_eval_uninavid.py"
        if not offline_eval_path.exists():
            raise FileNotFoundError(f"offline_eval_uninavid.py not found at: {offline_eval_path}")

        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "offline_eval_uninavid",
            str(offline_eval_path)
        )
        offline_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(offline_module)
        UniNaVid_Agent = offline_module.UniNaVid_Agent

        print("[UniNaVidServer] Successfully imported UniNaVid_Agent")

        if lora_path:
            # Reuse the original inference methods while swapping in the
            # base-model + LoRA initialization path used by Uni-NaVid_waypoints.
            class UniNaVidLoRAAgent(UniNaVid_Agent):
                def __init__(self, model_path: str, lora_path: str):
                    print("Initialize UniNaVid")

                    self.conv_mode = "vicuna_v1"
                    self.model_name = offline_module.get_model_name_from_path(model_path)
                    self.tokenizer, self.model, self.image_processor, self.context_len = (
                        offline_module.load_pretrained_model(
                            lora_path,
                            model_path,
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
                        "of the following: forward, left, right, or stop."
                    )
                    self.rgb_list = []
                    self.count_id = 0
                    self.reset()

            model = UniNaVidLoRAAgent(model_path=model_path, lora_path=lora_path)
        else:
            # Create model instance exactly as in the original code.
            # offline_eval_uninavid.py:285: agent = UniNaVid_Agent(...)
            model = UniNaVid_Agent(model_path)

        print("[UniNaVidServer] Model loaded successfully")
        return model

    finally:
        os.chdir(original_cwd)


# ============================================================================
# HTTP Endpoints
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": _model is not None,
        "model_path": _model_path,
        "lora_path": _lora_path,
        "uninavid_path": str(_uninavid_path) if _uninavid_path else None,
        "pending_actions": len(_pending_actions),
    })


@app.route('/reset', methods=['POST'])
def reset():
    """Reset model state for a new episode.

    Request body:
        {
            "instruction": "navigation instruction text"
        }

    This directly calls the original reset() method from
    offline_eval_uninavid.py:141-155
    """
    global _model, _pending_actions, _last_pred_actions, _last_instruction

    if _model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json() or {}
        instruction = data.get("instruction", "")

        with _inference_lock:
            _last_instruction = instruction
            _pending_actions.clear()
            _last_pred_actions = []

            # Call original reset() method
            # offline_eval_uninavid.py:141-155
            _model.reset()

        return jsonify({
            "status": "reset",
            "instruction": instruction
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/act', methods=['POST'])
def act():
    """Get navigation action from model.

    Request body:
        {
            "instruction": "navigation instruction",
            "image": "base64_encoded_png_image",
            "image_shape": [height, width, 3]
        }

    Response:
        {
            "action": "forward" | "left" | "right" | "stop",
            "actions": ["forward"],
            "action_index": 1,
            "queue_len": 0,
            "pred_actions": ["forward", "left", ...],
            "source": "inference" | "queue"
        }

    This follows the original act() logic from:
    - offline_eval_uninavid.py:158-194
    - NaVid-VLN-CE/agent_uninavid.py:260-311 (pending action queue)

    IMPORTANT: The key insight from VLN-CE agent is:
    1. ALWAYS accumulate frames to rgb_list first (line 264)
    2. THEN check pending_action_list (line 270)
    3. Only run inference when queue is empty (line 284)

    This ensures the model sees the full video history for context.
    """
    global _model, _pending_actions, _last_pred_actions, _last_instruction

    if _model is None:
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

        # Decode image
        rgb_array = _decode_image(image_base64, image_shape)

        # Debug: log image info
        print(f"[UniNaVidServer] Image received: shape={rgb_array.shape}, dtype={rgb_array.dtype}, "
              f"min={rgb_array.min()}, max={rgb_array.max()}, mean={rgb_array.mean():.1f}")

        with _inference_lock:
            # Clear queue if instruction changes (new episode)
            if instruction and instruction != _last_instruction:
                _pending_actions.clear()
                _last_pred_actions = []
                _last_instruction = instruction

            # CRITICAL FIX: Always accumulate frame to model's rgb_list first!
            # This matches VLN-CE agent behavior (agent_uninavid.py:264):
            #   rgb = observations["rgb"]
            #   self.rgb_list.append(rgb)
            # The model needs video history for proper navigation decisions.
            _model.rgb_list.append(rgb_array)

            # Debug: log accumulated frame count
            frame_count = len(_model.rgb_list)
            print(f"[UniNaVidServer] Accumulated frames: {frame_count}")

            # Fast path: return pending action if available
            # agent_uninavid.py:270-278
            if _pending_actions:
                action_str = str(_pending_actions.popleft()).lower()
                if action_str not in _ALLOWED_ACTIONS:
                    action_str = "stop"

                return jsonify({
                    "action": action_str,
                    "actions": [action_str],
                    "action_index": _ACTION_TO_INDEX.get(action_str, 0),
                    "queue_len": len(_pending_actions),
                    "pred_actions": _last_pred_actions,
                    "accumulated_frames": frame_count,
                    "source": "queue",
                })

            # Run inference - model will use accumulated rgb_list
            # Note: predict_inference() in offline_eval_uninavid.py (line 111-112):
            #   imgs = self.process_images(self.rgb_list)
            #   self.rgb_list = []  # clears after processing
            # So the model processes all accumulated frames and then clears the list.
            #
            # Save frame_count before inference since rgb_list will be cleared
            frames_used_for_inference = frame_count

            navigation_qs = _model.promt_template.format(instruction)

            # Debug: print the full prompt sent to model
            print(f"[UniNaVidServer] Instruction: {instruction}")
            print(f"[UniNaVidServer] Full prompt: {navigation_qs[:200]}...")

            with torch.no_grad():
                navigation = _model.predict_inference(navigation_qs)

            # Debug: print raw model output
            print(f"[UniNaVidServer] Model raw output: {navigation}")

            # Parse action list from model output (space-separated string)
            action_list = navigation.split(" ")

            # Increment step counter BEFORE building result (matching original behavior)
            # See offline_eval_uninavid.py:190-192:
            #   self.executed_steps += 1
            #   self.latest_action = {"step": self.executed_steps, ...}
            _model.executed_steps += 1

            # Build trajectory from actions (matching original act() behavior)
            # See offline_eval_uninavid.py:170-184
            traj = [[0.0, 0.0, 0.0]]
            for action in action_list:
                if action == "stop":
                    traj = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                    break
                elif action == "forward":
                    waypoint = [x + y for x, y in zip(traj[-1], [0.5, 0.0, 0.0])]
                    traj.append(waypoint)
                elif action == "left":
                    waypoint = [x + y for x, y in zip(traj[-1], [0.0, 0.0, -np.deg2rad(30)])]
                    traj.append(waypoint)
                elif action == "right":
                    waypoint = [x + y for x, y in zip(traj[-1], [0.0, 0.0, np.deg2rad(30)])]
                    traj.append(waypoint)

            # Build result dict matching original act() method exactly
            # See offline_eval_uninavid.py:192
            result = {
                "step": _model.executed_steps,
                "path": [traj],
                "actions": action_list
            }

            # Update model's latest_action for compatibility with visualization/trajectory modules
            # See offline_eval_uninavid.py:192
            _model.latest_action = result.copy()

            # Debug: print parsed result
            print(f"[UniNaVidServer] Model result: {result}")

            # Extract action list from result
            pred_actions = _extract_action_list(result)
            _last_pred_actions = pred_actions

            # Queue only the first action by default so server-side execution semantics
            # match the real-robot client, while still allowing a larger queue via CLI.
            # agent_uninavid.py:293-306
            _pending_actions.clear()
            for action in pred_actions:
                _pending_actions.append(action)
                if len(_pending_actions) >= _MAX_PENDING_ACTIONS:
                    break

            # Pop first action to return
            action_str = str(_pending_actions.popleft()).lower()
            if action_str not in _ALLOWED_ACTIONS:
                action_str = "stop"

            return jsonify({
                "action": action_str,
                "actions": [action_str],
                "action_index": _ACTION_TO_INDEX.get(action_str, 0),
                "queue_len": len(_pending_actions),
                "pred_actions": pred_actions,
                "accumulated_frames": frames_used_for_inference,  # Use saved count, rgb_list is cleared after inference
                "source": "inference",
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/predict_text', methods=['POST'])
def predict_text():
    """Generate text answer for EQA task given observation.

    Request body:
        {
            "instruction": "EQA question (e.g., 'What color is the chair?')",
            "image": "base64_encoded_png_image",
            "image_shape": [height, width, 3]
        }

    Response:
        {
            "answer": "generated text answer"
        }
    """
    global _model

    if _model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        instruction = data.get("instruction", "")
        image_base64 = data.get("image")
        image_shape = data.get("image_shape", [336, 336, 3])

        if not instruction:
            return jsonify({"error": "No instruction/question provided"}), 400
        if not image_base64:
            return jsonify({"error": "No image provided"}), 400

        # Decode image
        rgb_array = _decode_image(image_base64, image_shape)

        # For text prediction, we need to prepare the model's rgb_list
        # similar to how predict_inference works in the original code
        with _inference_lock:
            # Set up the model's image buffer for text generation
            # Clear any previous images and add the current one (similar to act method)
            if hasattr(_model, 'rgb_list'):
                _model.rgb_list.clear()  # Clear first
                _model.rgb_list.append(rgb_array)  # Then append

            # Initialize model state for text generation (similar to reset)
            if hasattr(_model, 'model'):
                if hasattr(_model.model, 'config') and hasattr(_model.model.config, 'run_type'):
                    _model.model.config.run_type = "eval"

                if hasattr(_model.model, 'get_model'):
                    model_core = _model.model.get_model()
                    if hasattr(model_core, 'initialize_online_inference_nav_feat_cache'):
                        model_core.initialize_online_inference_nav_feat_cache()
                    if hasattr(model_core, 'new_frames'):
                        model_core.new_frames = 0  # Keep as initialized

            # Call predict_inference for text generation
            with torch.no_grad():
                answer = _model.predict_inference(instruction)

            return jsonify({
                "answer": str(answer)
            })

    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"[UniNaVidServer] Error in predict_text: {error_msg}")
        print(f"[UniNaVidServer] Traceback: {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500
# ============================================================================
# Helper Functions
# ============================================================================

def _decode_image(image_base64: str, image_shape: list) -> np.ndarray:
    """Decode base64 image to numpy array.

    Args:
        image_base64: Base64 encoded PNG image
        image_shape: Expected shape [H, W, 3]

    Returns:
        RGB image as numpy array (uint8, shape [H, W, 3])
    """
    # Decode base64
    img_bytes = base64.b64decode(image_base64)

    # Load as PIL Image
    img = Image.open(io.BytesIO(img_bytes))

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Convert to numpy array
    rgb_array = np.array(img)

    # Ensure uint8
    if rgb_array.dtype != np.uint8:
        rgb_array = rgb_array.astype(np.uint8)

    return rgb_array


def _normalize_action(token: Any) -> str:
    """Normalize action token to standard format."""
    text = str(token).strip().lower()
    # Strip common punctuation
    text = text.strip().strip("\"'`")
    text = text.strip(" \t\r\n,.;:!?")
    return text


def _extract_action_list(result: Any) -> list:
    """Extract action list from model result.

    The original act() method returns:
        {"step": int, "path": [...], "actions": ["forward", "left", ...]}

    Args:
        result: Model output dict

    Returns:
        List of normalized action strings
    """
    if not isinstance(result, dict):
        raise ValueError(f"Model result must be dict, got {type(result)}")

    actions = result.get("actions")
    if isinstance(actions, str):
        # Handle space-separated string
        actions = [a for a in actions.split(" ") if a]

    if not isinstance(actions, list) or not actions:
        raise ValueError(f"Model result missing non-empty 'actions': {result}")

    # Normalize and filter valid actions
    normalized = []
    for a in actions:
        token = _normalize_action(a)
        if token in _ALLOWED_ACTIONS:
            normalized.append(token)

    if not normalized:
        # Fallback if no valid actions found
        print(f"[UniNaVidServer] Warning: No valid actions found in {actions}, using 'stop'")
        normalized = ["stop"]

    return normalized


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(
        description="Uni-NaVid HTTP Server for OmniNavBench"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model_zoo/uninavid-7b-full-224-video-fps-1-grid-2",
        help="Path to Uni-NaVid base model checkpoint (relative to uninavid_path)"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Optional path to LoRA checkpoint directory (relative to uninavid_path)"
    )
    parser.add_argument(
        "--uninavid_path",
        type=str,
        required=True,
        help="Path to Uni-NaVid project root"
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Server port (required; pick any free TCP port)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--max_pending_actions",
        type=int,
        default=1,
        help="Max pending actions to queue (default: 1, matches real-robot client)"
    )

    args = parser.parse_args()

    # Convert to Path
    uninavid_path = Path(args.uninavid_path)
    if not uninavid_path.exists():
        raise FileNotFoundError(f"Uni-NaVid path does not exist: {uninavid_path}")

    resolved_model_path = _resolve_model_arg(args.model_path, uninavid_path)
    if resolved_model_path is None or not Path(resolved_model_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {resolved_model_path}")

    resolved_lora_path = _resolve_model_arg(args.lora_path, uninavid_path)
    if resolved_lora_path is not None and not Path(resolved_lora_path).exists():
        raise FileNotFoundError(f"LoRA path does not exist: {resolved_lora_path}")

    # Set global config
    global _model, _model_path, _lora_path, _uninavid_path, _MAX_PENDING_ACTIONS
    _model_path = resolved_model_path
    _lora_path = resolved_lora_path
    _uninavid_path = uninavid_path
    _MAX_PENDING_ACTIONS = max(1, args.max_pending_actions)

    # Load model
    try:
        _model = load_uninavid_model(
            model_path=resolved_model_path,
            uninavid_path=uninavid_path,
            lora_path=resolved_lora_path,
        )
        print(f"[UniNaVidServer] Model loaded successfully")
        print(f"[UniNaVidServer] Max pending actions: {_MAX_PENDING_ACTIONS}")
    except Exception as e:
        print(f"[UniNaVidServer] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Start server
    print(f"[UniNaVidServer] Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
