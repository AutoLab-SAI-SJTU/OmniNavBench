"""OmniNav HTTP Server

Run this server in the OmniNav environment to serve waypoint prediction requests.
This server implements the same inference logic as the original OmniNav waypoint_agent.py

Usage:
python omninav_server.py \
    --model_path /path/to/omninav/checkpoint \
    --omninav_path /path/to/OmniNav \
    --port 8005 \
    --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import base64
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import deque
import io
import copy
import time

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import torch
from scipy.spatial.transform import Rotation as R
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration, \
    Qwen2_5_VLForConditionalGeneration, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from safetensors.torch import load_file
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Global instances and state
_model = None
_tokenizer = None
_processor = None
_inference_lock = None

# Model parameters (from original OmniNav) - optimized for memory
INPUT_IMG_SIZE = (640, 569)  # Native OmniNav input size.
HISTORY_RESIZE_RATIO = 1 / 4
MAX_HISTORY_FRAMES = 20
PREDICT_SCALE = 0.3
NUM_CURRENT_IMAGE = 3
NUM_ACTION_TRUNK = 5
flow_match = False

# Session state management (per episode)
_session_states = {}


class SessionState:
    """Manage per-episode state for OmniNav inference."""

    def __init__(self):
        self.rgb_history = []  # Use list like original OmniNav code
        self.pose_history = []  # Use list like original OmniNav code
        self.image_indices = []  # Use list like original OmniNav code
        self.total_frame_count = 0
        self.instruction = ""

    def reset(self, instruction: str = ""):
        """Reset session state for new episode."""
        self.rgb_history.clear()
        self.pose_history.clear()
        self.image_indices.clear()
        self.total_frame_count = 0
        self.instruction = instruction

    def add_frame(self, rgbs: List[np.ndarray], pose: dict):
        """Add new frame to history (copied from original waypoint_agent.py)."""
        # Convert pose to habitat format if needed
        if isinstance(pose, dict) and 'position' in pose:
            pose_habitat = pose
        else:
            # Convert from OmniNav format to habitat format
            pose_habitat = self._convert_pose_to_habitat(pose)

        # Process images
        rgbs_new = []
        for rgb in rgbs:
            if isinstance(rgb, np.ndarray):
                rgb_img = Image.fromarray(rgb)
                rgb = rgb_img.resize(INPUT_IMG_SIZE)
            else:
                rgb = rgb.resize(INPUT_IMG_SIZE)
            rgbs_new.append(rgb)

        # Remove old frames from current image slots if needed
        if len(self.rgb_history) >= NUM_CURRENT_IMAGE:
            for _ in range(NUM_CURRENT_IMAGE - 1):
                if len(self.rgb_history) > 0:
                    self.rgb_history.pop(-2)
                    self.pose_history.pop(-2)
                    self.image_indices.pop(-2)

        self.rgb_history.extend(rgbs_new)
        self.pose_history.extend([pose_habitat] * len(rgbs_new))
        self.image_indices.extend([self.total_frame_count] * len(rgbs_new))
        self.total_frame_count += 1

        # Resize older images
        if len(self.rgb_history) > NUM_CURRENT_IMAGE:
            self.rgb_history[-1 - NUM_CURRENT_IMAGE] = self.rgb_history[-1 - NUM_CURRENT_IMAGE].resize(
                (int(INPUT_IMG_SIZE[0] * HISTORY_RESIZE_RATIO), int(INPUT_IMG_SIZE[1] * HISTORY_RESIZE_RATIO)))

        # Resample if too many frames
        if len(self.rgb_history) > MAX_HISTORY_FRAMES + NUM_CURRENT_IMAGE:
            min_interval_idx = np.argmin(np.diff(self.image_indices[:-NUM_CURRENT_IMAGE]))
            self.rgb_history.pop(min_interval_idx + 1)
            self.pose_history.pop(min_interval_idx + 1)
            self.image_indices.pop(min_interval_idx + 1)

    def _convert_pose_to_habitat(self, pose):
        """Convert pose to habitat format."""
        if isinstance(pose, dict) and 'position' in pose:
            return pose
        # Assume it's a tuple/list format and convert
        return {
            'position': list(pose[:3]),
            'rotation': list(pose[3:]) if len(pose) > 3 else [1.0, 0.0, 0.0, 0.0]
        }

    def generate_infer_prompt(self) -> List[Dict]:
        """Generate inference prompt (copied from original waypoint_agent.py)."""
        cur_prompt = self._get_prompt_template()

        input_poses = copy.deepcopy(list(self.pose_history))
        local_poses = self._transform_poses_to_local(self.pose_history[-1], input_poses)

        input_positions = [[pose[0, 3], pose[2, 3]] for pose in local_poses]
        images = list(self.rgb_history)

        history_pose_strings = ['<{:.3f},{:.3f}>'.format(pose[0], pose[1]) for pose in input_positions]
        history_pose_string = ",".join(history_pose_strings)

        history_img_string = ''
        current_img_string = "Your current observations is leftside: , rightside: , frontside: "

        cur_prompt = cur_prompt.format(
            instruction=self.instruction,
            history_pose_string=history_pose_string,
            step_scale=PREDICT_SCALE,
            num_action_trunck=NUM_ACTION_TRUNK,
            current_img_string=current_img_string,
            history_img_string=history_img_string
        )

        return self._qwen_data_pack(images, cur_prompt)

    def _get_prompt_template(self) -> str:
        """Get the appropriate prompt template."""
        if flow_match:
            return """You are an autonomous navigation robot. You will get a task with historical pictures and current pictures you see.
Based on these information, you need to decide your next {num_action_trunck} actions, which could involve <|left|>,<|right|>,<|forward|>. If you finish your mission, output <|stop|>. Here are some examples: <|left|><|forward|><|forward|><|stop|>, <|forward|><|forward|><|forward|><|left|><|forward|> or <|stop|>
# Your historical pictures are: {history_img_string}
# {current_img_string}
# Your mission is: {instruction}<|NAV|>"""
        else:
            return """You are an autonomous navigation robot. You will get a task with historical pictures and current pictures you see.
Based on these information, you need to decide your next {num_action_trunck} actions, which could involve <|left|>,<|right|>,<|forward|>. If you finish your mission, output <|stop|>. Here are some examples: <|left|><|forward|><|forward|><|left|><|forward|> or <|stop|>
# Your historical pictures are: {history_img_string}
# {current_img_string}
# Your mission is: {instruction}<|NAV|>\nOutput the waypoint"""

    def _transform_poses_to_local(self, current_pose, input_poses):
        """Transform poses to local coordinate system."""
        if isinstance(current_pose, dict):
            current_pos = np.array(current_pose['position'])
            current_rot = np.array(current_pose['rotation'])
            current_rot_matrix = R.from_quat(current_rot[[1, 2, 3, 0]]).as_matrix()
        else:
            current_rot_matrix = current_pose[:3, :3]
            current_pos = current_pose[:3, 3]

        rot_normal_raw = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        current_rot_matrix = current_rot_matrix @ rot_normal_raw
        current_pose_matrix = np.eye(4)
        current_pose_matrix[:3, :3] = current_rot_matrix
        current_pose_matrix[:3, 3] = current_pos
        current_pose_inv = np.linalg.inv(current_pose_matrix)

        output_poses = []
        for pose in input_poses:
            if isinstance(pose, dict):
                pos = np.array(pose['position'])
                rot = np.array(pose['rotation'])
                rot_matrix = R.from_quat(rot[[1, 2, 3, 0]]).as_matrix()
            else:
                rot_matrix = pose[:3, :3]
                pos = pose[:3, 3]

            rot_matrix = rot_matrix @ rot_normal_raw
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = rot_matrix
            pose_matrix[:3, 3] = pos
            local_pose = current_pose_inv @ pose_matrix
            output_poses.append(local_pose)

        return output_poses

    def _qwen_data_pack(self, images, user_content):
        """Pack data for Qwen model (copied from original)."""
        content = []
        for idx, image in enumerate(images):
            if idx >= len(images) - NUM_CURRENT_IMAGE:
                cur_json = {
                    "type": "image",
                    "image": image,
                    "resized_height": INPUT_IMG_SIZE[1],
                    "resized_width": INPUT_IMG_SIZE[0],
                }
            else:
                cur_json = {
                    "type": "image",
                    "image": image,
                    "resized_height": INPUT_IMG_SIZE[1] * HISTORY_RESIZE_RATIO,
                    "resized_width": INPUT_IMG_SIZE[0] * HISTORY_RESIZE_RATIO,
                }
            content.append(cur_json)
        content.append({
            "type": "text",
            "text": user_content,
        })
        messages = [
            {
                "role": "user",
                "content": content
            },
        ]
        return messages


def get_session_state(session_id: str) -> SessionState:
    """Get or create session state."""
    if session_id not in _session_states:
        _session_states[session_id] = SessionState()
    return _session_states[session_id]


def qwen_infer(messages):
    """Run Qwen inference with memory optimization (copied from original waypoint_agent.py)."""
    global _model, _processor, _tokenizer

    try:
        # Clear cache before inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text + "<|im_end|>"

        nav_version = 'special_token'
        if nav_version == 'special_token':
            text = text.replace('<|vision_start|><|image_pad|><|vision_end|>', '')
            num_image = len(messages[0]['content']) - 1
            num_current_image = 3
            num_history_image = num_image - num_current_image

            history_img_str = ''.join(['<|vision_start|><|image_pad|><|vision_end|>'] * num_history_image)
            history_str_pos = text.rfind('Your historical pictures are: ') + len('Your historical pictures are: ')
            text = text[:history_str_pos] + history_img_str + text[history_str_pos:]

            text = text.replace('leftside: ', 'leftside: <|vision_start|><|image_pad|><|vision_end|>')
            text = text.replace('rightside: ', 'rightside: <|vision_start|><|image_pad|><|vision_end|>')
            text = text.replace('frontside: ', 'frontside: <|vision_start|><|image_pad|><|vision_end|>')

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = _processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")

        # Use torch.no_grad() for inference to save memory
        with torch.no_grad():
            if flow_match:
                norm = [{"min": [
                                 [-0.49142804741859436, -0.018926994875073433, -0.5000011853675626, 0.8660246981163404, 0.0],
                                 [-0.8506758809089646, -0.11684392392635345, -0.5176391471000088, -0.36602701582911296, 0.0],
                                 [-0.9391180276870728, -0.262770414352417, -0.5176390363234377, -0.5000015591363245, 0.0],
                                 [-0.9319084137678146, -0.5872985124588013, -0.5176390363234377, -0.5176391890893195, 0.0],
                                 [-0.9333658218383789, -0.8579317331314087, -0.5176390363233605, -0.5176391431200632, 0.0]],
                         "max": [[0.8222980499267578, 1.1485368013381958, 0.5000012222510074, 1.0, 1.0],
                                     [0.8579317331314087, 1.0390557050704985, 0.5176391335634103, 0.13397477820902748, 1.0],
                                     [0.9584183096885622, 0.9541159868240356, 0.5176391335632885, 0.3660255949191993, 1.0],
                                     [0.9442337155342072, 0.9441415071487427, 0.5176391335631186, 0.5000004173778672, 1.0],
                                     [0.9610724449157715, 0.9491362571716309, 0.5176391335630393, 0.5176390878671062, 1.0]]}]
                wp_pred, arrive_pred, sin_angle, cos_angle = _model.forward(**inputs.to(_model.device), norm=norm, action_former=True,
                                                                            gt_waypoints=0, train=False,
                                                                            train_branch=['continue'])
            else:
                wp_pred, arrive_pred, sin_angle, cos_angle = _model.forward(**inputs.to(_model.device), action_former=True,
                                                                            gt_waypoints=0, train=False,
                                                                            train_branch=['continue'])

        # Clear cache after inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return wp_pred * PREDICT_SCALE, arrive_pred, sin_angle, cos_angle

    except torch.cuda.OutOfMemoryError as e:
        # Clear cache on OOM and re-raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.error(f"CUDA OOM during inference: {e}")
        raise e
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise e


def load_omninav_model(args, action_mode: bool = False) -> Any:
    """Load OmniNav model components directly."""
    global _model, _tokenizer, _processor, _inference_lock, _flow_match

    # Set the global flow_match flag
    _flow_match = action_mode

    try:
        # Set memory optimization environment variables
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:512')

        # Clear any existing GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Load tokenizer and processor
        _tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        _processor = AutoProcessor.from_pretrained(args.model_path)

        # Load model with memory optimizations
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",  # Use bfloat16 for memory efficiency
            device_map="auto",
            attn_implementation="flash_attention_2",  # Changed from flash_attention_2 to eager for compatibility
            low_cpu_mem_usage=True,
        )

        # Load safetensors weights if available
        if flow_match:
            _model = _model.cuda()
            for name in os.listdir(args.model_path):
                if name.endswith('safetensors'):
                    safe_model_path = os.path.join(args.model_path, name)
                    state_dict = load_file(safe_model_path)
                    _model.load_state_dict(state_dict, strict=False)
        else:
            _model = _model.cuda()
            for name in os.listdir(args.model_path):
                if name.endswith('safetensors'):
                    safe_model_path = os.path.join(args.model_path, name)
                    state_dict = load_file(safe_model_path)
                    _model.load_state_dict(state_dict, strict=False)

        # Set model to evaluation mode and disable gradients
        _model.eval()
        _model.requires_grad_(False)

        # Additional memory optimizations
        if torch.cuda.is_available():
            # Enable TF32 for better performance
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        _inference_lock = torch.multiprocessing.Lock() if hasattr(torch, 'multiprocessing') else None

        # Print memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[OmniNavServer] ✓ Model loaded successfully. GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        else:
            print("[OmniNavServer] ✓ Model loaded successfully (CPU mode)")

        return _model

    except Exception as e:
        raise RuntimeError(f"Failed to load OmniNav model: {e}")


def _decode_image(image_base64: str) -> np.ndarray:
    """Decode base64 image to numpy array (uint8, RGB)."""
    try:
        img_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        rgb_array = np.array(img)

        if rgb_array.dtype != np.uint8:
            rgb_array = rgb_array.astype(np.uint8)
        return rgb_array
    except Exception as e:
        logger.error(f"Image decode failed: {e}")
        raise ValueError("Invalid image data")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": _model is not None
    })


@app.route('/reset', methods=['POST'])
def reset():
    """Reset session state for a new episode."""
    global _model
    if _model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json() or {}
        session_id = data.get("session_id", "default")
        instruction = data.get("instruction", "")

        # Reset session state
        session_state = get_session_state(session_id)
        session_state.reset(instruction)

        logger.info(f"Reset session {session_id} for instruction: {instruction[:30]}...")
        return jsonify({"status": "reset", "session_id": session_id, "instruction": instruction})
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/act', methods=['POST'])
def act():
    """Get waypoint prediction from model given observation."""
    global _model
    if _model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        session_id = data.get("session_id", "default")
        instruction = data.get("instruction", "")
        left_image_b64 = data.get("left_image")
        front_image_b64 = data.get("front_image")
        right_image_b64 = data.get("right_image")
        pose = data.get("pose", {})
        step = data.get("step", 0)

        if not all([left_image_b64, front_image_b64, right_image_b64]):
            return jsonify({"error": "Missing required camera images"}), 400

        # Get session state
        session_state = get_session_state(session_id)
        if not session_state.instruction:
            session_state.instruction = instruction

        # Decode images
        left_rgb = _decode_image(left_image_b64)
        front_rgb = _decode_image(front_image_b64)
        right_rgb = _decode_image(right_image_b64)
        # Add frame to session history
        session_state.add_frame([left_rgb, right_rgb, front_rgb], pose)

        start_time = time.time()

        # Generate inference prompt
        messages = session_state.generate_infer_prompt()

        # Run inference
        if _inference_lock:
            with _inference_lock:
                wp_pred, arrive_pred, sin_angle, cos_angle = qwen_infer(messages)
        else:
            wp_pred, arrive_pred, sin_angle, cos_angle = qwen_infer(messages)

        inference_time = time.time() - start_time

        # Process results (same logic as original)
        if flow_match:
            cnt = 0
            for cur_arrive in arrive_pred.squeeze():
                if cur_arrive.item() > 0.5:
                    cnt += 1
            arrive_pred_final = 1 if cnt == 5 else 0
        else:
            cnt = 0
            for cur_arrive in arrive_pred.squeeze():
                if cur_arrive.item() >= 0:
                    cnt += 1
            arrive_pred_final = 1 if cnt == 5 else 0

        wp_pred = wp_pred.cpu().type(torch.float32).numpy().squeeze()
        recover_angle_tensor = torch.atan2(sin_angle, cos_angle).detach().cpu().type(torch.float32).numpy().squeeze()

        if _flow_match:
            # Action mode: return action sequence
            action_data = wp_pred.tolist() if hasattr(wp_pred, 'tolist') else wp_pred
            print(f"[OmniNavServer] Action mode response: action={action_data}, arrive_pred={arrive_pred_final}")
            response_data = {
                "action": action_data,
                "arrive_pred": float(arrive_pred_final),
                "recover_angle": recover_angle_tensor.tolist() if hasattr(recover_angle_tensor, 'tolist') else recover_angle_tensor,
                "inference_time": inference_time,
                "mode": "action"
            }
            # Waypoint mode: return ALL 5 waypoints for trajectory following
        # Model predicts 5 waypoints and 5 angles (NUM_ACTION_TRUNK = 5)
        # Return all waypoints to enable move_along_path controller

        # Return all 5 waypoints
        if wp_pred.ndim == 2 and wp_pred.shape[0] == 5:  # Shape: (5, 2)
            waypoints = wp_pred.tolist()
        else:
            # Fallback for unexpected shape
            waypoints = [wp_pred.tolist()] if wp_pred.ndim == 1 else wp_pred.tolist()

        # Return all 5 angles
        if isinstance(recover_angle_tensor, np.ndarray) and recover_angle_tensor.shape[0] == 5:
            recover_angles = recover_angle_tensor.tolist()
        elif isinstance(recover_angle_tensor, np.ndarray) and recover_angle_tensor.size == 1:
            recover_angles = [float(recover_angle_tensor.item())]
        else:
            recover_angles = [float(recover_angle_tensor)]

        response_data = {
            "waypoints": waypoints,
            "arrive_pred": float(arrive_pred_final),
            "recover_angles": recover_angles,
            "inference_time": inference_time,
            "mode": "waypoint"
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="OmniNav HTTP Server")

    # Path args
    parser.add_argument("--model_path", type=str, required=True, help="Path to OmniNav checkpoint")
    parser.add_argument("--omninav_path", type=str, required=True, help="Path to OmniNav source code root")

    # Server args
    parser.add_argument("--port", type=int, required=True,
                        help="Server port (required; pick any free TCP port)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--action_mode", action="store_true", help="Use action mode instead of waypoint mode")

    args = parser.parse_args()

    try:
        load_omninav_model(args, action_mode=args.action_mode)
        logger.info(f"Starting OmniNav Server on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port, threaded=False)  # Use single-threaded for reproducibility
    except Exception as e:
        logger.critical(f"Server startup failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
