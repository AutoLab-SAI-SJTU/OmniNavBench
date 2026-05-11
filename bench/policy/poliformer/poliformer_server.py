#!/usr/bin/env python3
"""
PoliFormer HTTP Server for OmniNavBench

This server loads a PoliFormer inference agent and exposes simple HTTP
endpoints similar to other model servers in this repo:
  - GET /health
  - POST /reset
  - POST /act

Usage example:
  python -m bench.policy.poliformer_server \
    --poliformer-path /path/to/PoliFormer \
    --ckpt-path /path/to/checkpoint.ckpt \
    --port 8003
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

import torch

# String processing utility will be imported in load_poliformer_agent
# to avoid premature loading of PoliFormer dependencies

app = Flask(__name__)

# Global model/agent
_agent = None
_device = "cpu"
_ckpt_path = None
_poliformer_path = None
_convert_string_to_byte = None  # Global reference to string conversion function


def _decode_image_b64(image_b64: str) -> np.ndarray:
    """Decode base64 PNG/JPEG to uint8 numpy array (H,W,3)."""
    img_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def load_poliformer_agent(
    poliformer_path: str,
    ckpt_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_config: str = "InferenceDINOv2ViTSLLAMATxTxObjectNavDist",
) -> Any:
    """
    Load PoliFormer inference agent.

    This function uses classes from the PoliFormer repo to construct an
    InferenceAgent and load weights from a checkpoint.
    """
    global _poliformer_path, _ckpt_path, _device, _convert_string_to_byte
    _poliformer_path = poliformer_path
    _ckpt_path = ckpt_path
    _device = device

    # Resolve and validate poliformer_path. Try multiple fallbacks if the
    # provided path does not exist to make the server more user-friendly.
    candidate_paths = []
    provided_path = Path(poliformer_path)
    if provided_path.exists():
        candidate_paths.append(provided_path.resolve())
    # Environment variable fallback
    env_path = os.getenv("POLIFORMER_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            candidate_paths.append(p.resolve())

    # Try repo-relative fallback: assume PoliFormer is sibling to this repo root
    repo_root = Path(__file__).resolve().parents[3]
    repo_relative = repo_root / "PoliFormer"
    if repo_relative.exists():
        candidate_paths.append(repo_relative.resolve())

    # Try current working directory / "PoliFormer"
    cwd_relative = Path.cwd() / "PoliFormer"
    if cwd_relative.exists():
        candidate_paths.append(cwd_relative.resolve())

    # Final selection
    if not candidate_paths:
        tried = [str(provided_path), f"ENV POLIFORMER_PATH={env_path}" if env_path else "ENV POLIFORMER_PATH not set", str(repo_relative), str(cwd_relative)]
        raise FileNotFoundError(
            "Cannot find PoliFormer repository. Tried:\n  " + "\n  ".join(tried) + "\n\n"
            "Please pass --poliformer-path pointing to the PoliFormer repo root, or set POLIFORMER_PATH env var."
        )

    poliformer_path = candidate_paths[0]
    print(f"[PoliFormerServer] Using PoliFormer path: {poliformer_path}")
    if str(poliformer_path) not in sys.path:
        sys.path.insert(0, str(poliformer_path))

    # Some modules rely on relative file access (e.g., utils/*.json). Change
    # the current working directory to the PoliFormer repo root while we
    # import and initialize the agent so relative paths resolve correctly.
    original_cwd = os.getcwd()
    os.chdir(str(poliformer_path))
    try:
        # Prevent the code from trying to load large Objaverse house lists by
        # pointing OBJAVERSE_HOUSES_DIR to a harmless empty directory when not set.
        # This allows using the agent for inference without the dataset.
        tmp_houses_dir = poliformer_path / "tmp_objaverse_houses"
        tmp_houses_dir.mkdir(exist_ok=True)
        # Ensure minimal placeholder files exist so LazyJsonHouses.from_dir won't fail.
        import gzip

        for subset_name in ("train", "val", "test"):
            gz_path = tmp_houses_dir / f"{subset_name}.jsonl.gz"
            if not gz_path.exists():
                with gzip.open(gz_path, "wb") as f:
                    # write nothing -> empty jsonl.gz (zero houses)
                    f.write(b"")

        os.environ.setdefault("OBJAVERSE_HOUSES_DIR", str(tmp_houses_dir))

        # Import minimal components required for building the agent
        from architecture.models.allenact_transformer_models.inference_agent import (
            InferenceAgentVIDA,
        )
        # older repo layout places preprocessor under architecture.allenact_preprocessors
        try:
            from architecture.preprocessors.dino_preprocessors import DinoViTPreprocessor
        except Exception:
            from architecture.allenact_preprocessors.dino_preprocessors import DinoViTPreprocessor

        # Import string processing utility
        from utils.string_utils import convert_string_to_byte
        _convert_string_to_byte = convert_string_to_byte

        # Import the experiment config and params used by default evaluation
        from training.online.dinov2_vits_tsfm_rgb_augment_objectnav import (
            DinoV2ViTSTSFMObjectNav,
            DinoV2ViTSTSFMObjectNavParams,
        )

        print("[PoliFormerServer] Building agent parameters...")
        # Build params and instantiate agent using the helper in InferenceAgentVIDA.
        # Set num_train_processes=0 to avoid training-time data loading.
        params = DinoV2ViTSTSFMObjectNavParams()
        params.num_train_processes = 0

        # Auto-detect checkpoint variant from ckpt_path (text_nav / box_nav / text_box_nav)
        ckpt_lower = str(ckpt_path).lower() if ckpt_path is not None else ""
        if "text_box_nav" in ckpt_lower or "text-box-nav" in ckpt_lower or "text_box" in ckpt_lower:
            params.use_text_goal = True
            params.use_bbox = True
            print("[PoliFormerServer] Detected variant: text_box_nav -> use_text_goal=True, use_bbox=True")
        elif "box_nav" in ckpt_lower or "box-nav" in ckpt_lower or "box_nav" in ckpt_lower:
            params.use_text_goal = False
            params.use_bbox = True
            print("[PoliFormerServer] Detected variant: box_nav -> use_text_goal=False, use_bbox=True")
        else:
            # Default to text_nav for inference
            params.use_text_goal = True
            params.use_bbox = False
            print("[PoliFormerServer] Defaulting to text_nav variant -> use_text_goal=True, use_bbox=False")

        # Determine mean/std for image normalization
        img_mean = DinoViTPreprocessor.DINO_RGB_MEANS
        img_std = DinoViTPreprocessor.DINO_RGB_STDS
        print("[PoliFormerServer] Image normalization: mean={}, std={}".format(img_mean, img_std))

        print("[PoliFormerServer] Starting agent construction (this may take a while)...")
        print("[PoliFormerServer] - Loading DINOv2 model (dinov2_vits14)...")
        import time
        start_time = time.time()

        # Use class helper to build agent (this will load ckpt inside)
        agent = InferenceAgentVIDA.build_agent(
            exp_config_type=DinoV2ViTSTSFMObjectNav,
            params=params,
            device=device,
            img_encoder_rgb_mean=img_mean,
            img_encoder_rgb_std=img_std,
            greedy_sampling=False,
            test_augmentation=False,
            ckpt_path=ckpt_path,
        )

        load_time = time.time() - start_time
        print(f"[PoliFormerServer] Agent construction took {load_time:.2f} seconds")
        print("[PoliFormerServer] Agent construction completed successfully!")

        return agent
    finally:
        # Restore original cwd
        os.chdir(original_cwd)


@app.route("/health", methods=["GET"])
def health():
    print(f"[PoliFormerServer] Health check requested")
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": _agent is not None,
            "device": str(_device),
            "ckpt_path": _ckpt_path,
            "poliformer_path": str(_poliformer_path),
        }
    )


@app.route("/reset", methods=["POST"])
def reset():
    """Reset agent episodic state."""
    global _agent
    if _agent is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.get_json() or {}
        instruction = data.get("instruction", "")
        print(f"[PoliFormerServer] Resetting agent state for new episode")
        # In this codebase, agent.reset() does not accept instruction, but
        # we keep the param for API compatibility.
        if hasattr(_agent, "reset"):
            _agent.reset()
            print(f"[PoliFormerServer] Agent reset complete")
        return jsonify({"status": "reset", "instruction": instruction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/act", methods=["POST"])
def act():
    """
    Expect JSON:
      {
        "instruction": "text goal",
        "image": "<base64 png/jpg string>"
      }
    Returns:
      {
        "action": "forward"|"left"|"right"|"stop"|...,
        "prob": float,
        "raw": optional
      }
    """
    global _agent
    if _agent is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        instruction = data.get("instruction", "")
        image_b64 = data.get("image") or data.get("rgb_base64") or data.get("rgb")
        if not image_b64:
            return jsonify({"error": "No image provided"}), 400

        # Decode image
        frame_arr = _decode_image_b64(image_b64)

        # (no per-frame debug logs here)

        # Build observations dict expected by InferenceAgentVIDA
        # The agent expects observations in the format that matches training
        global _convert_string_to_byte
        if _convert_string_to_byte is None:
            raise RuntimeError("String conversion function not loaded. Make sure load_poliformer_agent was called first.")

        observations = {
            "rgb_raw": frame_arr,  # Raw RGB image for DINO preprocessing
            "natural_language_spec": _convert_string_to_byte(instruction, 1000),  # Encoded text instruction
            "time_step": _agent.steps_taken_in_task,  # Current time step
            "traj_index": _agent.num_evaluated_traj,  # Trajectory index
        }

        # Debug: log input to model
        print(f"[PoliFormerServer] Processing request...")
        print(f"[PoliFormerServer] Model input: instruction='{instruction[:50]}{'...' if len(instruction) > 50 else ''}', image_shape={frame_arr.shape}")
        print(f"[PoliFormerServer] Observations keys: {list(observations.keys())}")
        print(f"[PoliFormerServer] Time step: {_agent.steps_taken_in_task}, Traj index: {_agent.num_evaluated_traj}")

        # Debug: show observation details
        print(f"[PoliFormerServer] Observation details:")
        print(f"[PoliFormerServer]   rgb_raw shape: {observations['rgb_raw'].shape}, dtype: {observations['rgb_raw'].dtype}")
        print(f"[PoliFormerServer]   natural_language_spec shape: {observations['natural_language_spec'].shape}, dtype: {observations['natural_language_spec'].dtype}")
        print(f"[PoliFormerServer]   time_step: {observations['time_step']}")
        print(f"[PoliFormerServer]   traj_index: {observations['traj_index']}")

        # Call act method directly with observations (this will apply preprocessing)
        print(f"[PoliFormerServer] Running inference (DINO preprocessing + Transformer)...")
        import time
        inference_start = time.time()

        # Get detailed action information before calling act
        action_list = _agent.get_action_list()
        print(f"[PoliFormerServer] Available actions ({len(action_list)}): {action_list}")

        # Show agent configuration
        greedy_sampling = getattr(_agent, 'greedy_sampling', 'Unknown')
        print(f"[PoliFormerServer] Agent config: greedy_sampling={greedy_sampling}")

        action_str, prob = _agent.act(observations, instruction)
        inference_time = time.time() - inference_start
        print(f"[PoliFormerServer] Inference completed in {inference_time:.3f} seconds")

        # Normalize types for JSON response
        try:
            # Ensure action_str is a plain string
            action_str = str(action_str)
        except Exception:
            action_str = ""

        # Safely convert prob to float (handle torch.Tensor, numpy, list/tuple)
        prob_value = 0.0
        try:
            # torch.Tensor
            import torch as _torch

            if isinstance(prob, _torch.Tensor):
                if prob.numel() == 1:
                    prob_value = float(prob.detach().cpu().item())
                else:
                    prob_value = float(prob.detach().cpu().flatten()[0].item())
            elif isinstance(prob, (list, tuple)):
                prob_value = float(prob[0])
            else:
                # numpy scalar or python number
                prob_value = float(prob)
        except Exception:
            try:
                prob_value = float(prob)
            except Exception:
                prob_value = 0.0

        # Debug output for action prediction
        print(f"[PoliFormerServer] Decision: action='{action_str}', prob={prob_value:.4f}")
        print(f"[PoliFormerServer] Action distribution details: raw_prob={prob}, type={type(prob)}")

        # Try to get full action distribution if available
        try:
            if hasattr(_agent, 'actor_critic') and hasattr(_agent.actor_critic, 'action_space'):
                action_list = _agent.get_action_list()
                print(f"[PoliFormerServer] Full action distribution:")
                for i, action_name in enumerate(action_list):
                    prob_val = "N/A"
                    try:
                        if hasattr(_agent, 'last_action_logits') and _agent.last_action_logits is not None:
                            import torch
                            if isinstance(_agent.last_action_logits, torch.Tensor):
                                logits = _agent.last_action_logits.detach().cpu()
                                if logits.numel() > i:
                                    prob_val = f"{torch.softmax(logits, dim=-1)[i].item():.4f}"
                    except Exception as e:
                        prob_val = f"Error: {e}"
                    print(f"[PoliFormerServer]   {action_name}: {prob_val}")
        except Exception as e:
            print(f"[PoliFormerServer] Could not get full distribution: {e}")

        return jsonify({"action": action_str, "prob": prob_value})
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def main():
    global _agent
    parser = argparse.ArgumentParser(description="PoliFormer HTTP Server for OmniNavBench")
    parser.add_argument("--poliformer-path", type=str, required=True, help="Path to PoliFormer repo root")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to PoliFormer checkpoint (.ckpt)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--port", type=int, required=True,
                        help="Server port (required; pick any free TCP port)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"[PoliFormerServer] Loading PoliFormer agent from {args.ckpt_path} (repo: {args.poliformer_path})")
    print(f"[PoliFormerServer] Using device: {args.device}")
    print(f"[PoliFormerServer] This may take several minutes for first-time DINO model download...")

    try:
        _agent = load_poliformer_agent(
            poliformer_path=args.poliformer_path, ckpt_path=args.ckpt_path, device=args.device
        )
        print("[PoliFormerServer] ✓ Agent loaded successfully")
    except Exception as e:
        print(f"[PoliFormerServer] ❌ Failed to load agent: {e}")
        import traceback

        traceback.print_exc()
        raise

    print(f"[PoliFormerServer] Starting HTTP server on {args.host}:{args.port}")
    print("[PoliFormerServer] Ready to accept requests!")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()


