#!/usr/bin/env python3
"""
NaVILA HTTP Server for OmniNavBench

This server loads the NaVILA model and provides HTTP endpoints for navigation inference.
NaVILA outputs natural language commands like "turn left 30 degrees", "move forward 2 meters", "stop".
"""

import argparse
import base64
import io
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image

# Global variables
_model = None
_tokenizer = None
_image_processor = None
_device = None
_history_frames: List[Image.Image] = []
_current_instruction: str = ""
_num_video_frames: int = 8

app = Flask(__name__)


def load_navila_model(
    model_path: str,
    navila_path: str,
    model_base: Optional[str] = None,
    device: str = "cuda",
) -> tuple:
    """
    Load NaVILA model from checkpoint.
    
    Args:
        model_path: Path to the NaVILA checkpoint
        navila_path: Path to the NaVILA repository
        model_base: Base model path (for LoRA models)
        device: Device to load model on
    
    Returns:
        Tuple of (model, tokenizer, image_processor)
    """
    global _device
    _device = device if torch.cuda.is_available() else "cpu"
    
    # Add NaVILA to path
    if navila_path not in sys.path:
        sys.path.insert(0, navila_path)
        print(f"[NaVILAServer] Added NaVILA path to sys.path: {navila_path}")
    
    # Import NaVILA modules
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    
    disable_torch_init()
    
    model_name = get_model_name_from_path(model_path)
    print(f"[NaVILAServer] Loading model: {model_name}")
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        model_name, 
        model_base,
        device=_device,
    )
    
    model.eval()
    print(f"[NaVILAServer] ✓ Model loaded successfully on {_device}")
    
    return model, tokenizer, image_processor


def decode_image(image_b64: str) -> Image.Image:
    """Decode base64 image to PIL Image."""
    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image


def sample_and_pad_images(images: List[Image.Image], num_frames: int) -> List[Image.Image]:
    """
    NaVILA eval preprocessing:
    - If len(images) < num_frames: pad black frames at the beginning
    - Uniformly sample num_frames-1 historical frames + latest frame

    Source: NaVILA/evaluation/vlnce_baselines/navila_trainer.py
    """
    if not images:
        raise ValueError("[NaVILAServer] sample_and_pad_images: empty images")
    if num_frames < 1:
        raise ValueError(f"[NaVILAServer] sample_and_pad_images: invalid num_frames={num_frames}")

    frames = list(images)
    width, height = frames[-1].size

    if len(frames) < num_frames:
        pad_count = num_frames - len(frames)
        black = Image.new("RGB", (width, height), color=(0, 0, 0))
        frames = [black] * pad_count + frames

    if num_frames == 1:
        return [frames[-1]]

    latest_frame = frames[-1]
    sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]
    return sampled_frames


def generate_action(
    images: List[Image.Image],
    instruction: str,
) -> Dict[str, Any]:
    """
    Generate navigation action from images and instruction.
    
    Args:
        images: List of PIL images (history + current)
        instruction: Navigation instruction
    
    Returns:
        Dict with action and raw response
    """
    global _model, _tokenizer, _image_processor, _num_video_frames
    
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
    
    num_video_frames = int(getattr(_model.config, "num_video_frames", _num_video_frames))
    past_and_current_rgbs = sample_and_pad_images(images, num_frames=num_video_frames)
    
    # Build prompt
    interleaved_images = "<image>\n" * (len(past_and_current_rgbs) - 1)
    qs = (
        f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
        f'of historical observations {interleaved_images}, and current observation <image>\n. Your assigned task is: "{instruction}" '
        f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
        f"degree, moving forward a certain distance, or stop if the task is completed."
    )
    
    conv_mode = "llama_3"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Process images
    images_tensor = process_images(past_and_current_rgbs, _image_processor, _model.config).to(_model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, _tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(_model.device)
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, _tokenizer, input_ids)
    
    # Generate
    with torch.inference_mode():
        output_ids = _model.generate(
            input_ids,
            images=images_tensor.half(),
            do_sample=False,
            temperature=0.0,
            max_new_tokens=32,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=_tokenizer.eos_token_id,
        )
    
    # Decode output
    outputs = _tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    
    return {"raw_response": outputs}


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": _model is not None,
        "device": str(_device),
        "num_video_frames": int(getattr(_model.config, "num_video_frames", _num_video_frames)) if _model is not None else _num_video_frames,
    })


@app.route("/reset", methods=["POST"])
def reset():
    """Reset episode state."""
    global _history_frames, _current_instruction
    
    data = request.get_json() or {}
    _current_instruction = data.get("instruction", "")
    _history_frames = []
    
    print(f"[NaVILAServer] Reset episode. Instruction: {_current_instruction[:50]}...")
    
    return jsonify({"status": "ok"})


@app.route("/act", methods=["POST"])
def act():
    """
    Generate action from observation.
    
    Expected JSON:
    {
        "rgb_base64": "...",  # Base64 encoded RGB image
        "instruction": "...",  # Optional, uses stored instruction if not provided
    }
    
    Returns:
    {
        "raw_response": "...",  # Raw model output
    }
    """
    global _history_frames, _current_instruction
    
    try:
        data = request.get_json()
        
        # Get instruction
        instruction = data.get("instruction", _current_instruction)
        if not instruction:
            return jsonify({"error": "No instruction provided"}), 400
        
        # Decode current image
        rgb_b64 = data.get("rgb_base64")
        if not rgb_b64:
            return jsonify({"error": "No rgb_base64 provided"}), 400
        
        current_image = decode_image(rgb_b64)
        
        # Add to history
        _history_frames.append(current_image)

        result = generate_action(_history_frames, instruction)
        print(f"[NaVILAServer] raw_response: {result.get('raw_response', '')[:200]}")
        return jsonify(result)
        
    except Exception as e:
        print(f"[NaVILAServer] Error in /act: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def main():
    global _model, _tokenizer, _image_processor, _num_video_frames
    
    parser = argparse.ArgumentParser(description="NaVILA HTTP Server")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to NaVILA checkpoint")
    parser.add_argument("--navila_path", type=str, required=True,
                        help="Path to NaVILA repository")
    parser.add_argument("--model_base", type=str, default=None,
                        help="Base model path (for LoRA)")
    parser.add_argument("--port", type=int, default=8002,
                        help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Number of video frames to use")
    args = parser.parse_args()
    
    _num_video_frames = args.num_frames
    
    print(f"[NaVILAServer] Loading NaVILA model...")
    try:
        _model, _tokenizer, _image_processor = load_navila_model(
            model_path=args.model_path,
            navila_path=args.navila_path,
            model_base=args.model_base,
        )
        print(f"[NaVILAServer] ✓ Model loaded successfully")
    except Exception as e:
        print(f"[NaVILAServer] ❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"[NaVILAServer] Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
