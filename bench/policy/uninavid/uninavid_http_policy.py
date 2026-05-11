"""Uni-NaVid HTTP client policy for OmniNavBench.

This policy communicates with a Uni-NaVid HTTP server running in a separate
process/environment.

Usage:
    from bench.policy.uninavid import UniNaVidHTTPPolicy

    # Create policy (server must be running)
    policy = UniNaVidHTTPPolicy(server_url="http://localhost:8000")

    # Reset for new episode
    policy.reset(instruction="Go to the kitchen")

    # Get action
    action = policy.act(observation)
"""

from __future__ import annotations

import base64
import io
from typing import Optional, Dict, Any

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image

from bench.policy.base import BasePolicy, Observation, Action


class UniNaVidHTTPPolicy(BasePolicy):
    """Uni-NaVid policy that communicates via HTTP with a remote server.

    This allows running Uni-NaVid in a separate conda environment and
    communicating through HTTP requests.

    The server should be running uninavid_server.py which loads the original
    UniNaVid_Agent from offline_eval_uninavid.py.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        timeout: float = 60.0,
        max_retries: int = 3,
        forward_speed: float = 1.0,  # m/s
        turn_angular_velocity: float = 1.0,  # rad/s
    ):
        """Initialize Uni-NaVid HTTP policy.

        Args:
            server_url: Base URL of the Uni-NaVid HTTP server
            timeout: Request timeout in seconds (longer for model inference)
            max_retries: Maximum number of retries for failed requests
            forward_speed: Linear velocity for forward action (m/s)
            turn_angular_velocity: Angular velocity for turn actions (rad/s)
        """
        super().__init__()
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.forward_speed = forward_speed
        self.turn_angular_velocity = turn_angular_velocity

        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Test connection
        self._check_health()
        print(f"[UniNaVidHTTPPolicy] Connected to server at {self.server_url}")

    def _check_health(self):
        """Check if the server is healthy."""
        try:
            response = self.session.get(
                f"{self.server_url}/health",
                timeout=10.0
            )
            response.raise_for_status()
            result = response.json()
            if not result.get("model_loaded", False):
                raise RuntimeError("Server reports model not loaded")
            print(f"[UniNaVidHTTPPolicy] Server health check passed: {result}")
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Uni-NaVid server at {self.server_url}. "
                f"Make sure the server is running. Error: {e}"
            )

    def reset(self, instruction: str = ""):
        """Reset policy state for new episode.

        Args:
            instruction: Navigation instruction for the episode
        """
        super().reset(instruction)

        # Send reset request to server
        print(f"[UniNaVidHTTPPolicy] Sending reset to {self.server_url}...", flush=True)
        try:
            response = self.session.post(
                f"{self.server_url}/reset",
                json={"instruction": instruction},
                timeout=5.0
            )
            response.raise_for_status()
            result = response.json()
            print(f"[UniNaVidHTTPPolicy] Reset: {result}", flush=True)
        except Exception as e:
            print(f"[UniNaVidHTTPPolicy] ❌ CRITICAL: Reset request failed: {e}", flush=True)
            raise e

    def act(self, observation: Observation) -> Action:
        """Generate action by sending observation to HTTP server.

        Args:
            observation: Current observation from environment

        Returns:
            Action with velocity commands
        """
        # Update history
        self.update_history(observation)

        # Check RGB availability
        if observation.rgb is None:
            print("[UniNaVidHTTPPolicy] Warning: observation.rgb is None, returning stop")
            return Action(stop=True, action_type='stop')

        # Prepare image for transmission
        rgb_array = self._prepare_image(observation.rgb)

        # Encode image as base64
        image_base64 = self._encode_image(rgb_array)

        # Prepare request payload
        payload = {
            "instruction": observation.instruction,
            "image": image_base64,
            "image_shape": list(rgb_array.shape),
        }

        # Send request to server
        try:
            response = self.session.post(
                f"{self.server_url}/act",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as e:
            print(f"[UniNaVidHTTPPolicy] Request failed: {e}, returning stop")
            return Action(stop=True, action_type='stop')

        # Parse server response
        action_str = self._parse_response(result)
        print(f"[UniNaVidHTTPPolicy] Action: {action_str} (source: {result.get('source', 'unknown')})")

        # Convert discrete action to Action object
        return self._discrete_to_continuous(action_str)

    def predict_text(self, instruction: str, rgb: np.ndarray) -> str:
        """Generate text answer for EQA task using HTTP server.

        Args:
            instruction: EQA question/instruction
            rgb: RGB image as numpy array

        Returns:
            Generated text answer
        """
        try:
            rgb_array = self._prepare_image(rgb)

            # Encode image as base64
            image_base64 = self._encode_image(rgb_array)

            # Prepare request payload
            payload = {
                "instruction": instruction,
                "image": image_base64,
                "image_shape": list(rgb_array.shape),
            }

            # Make request to /predict_text endpoint
            response = self.session.post(
                f"{self.server_url}/predict_text",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()

            # Extract answer from response
            if "answer" in result:
                return str(result["answer"])
            else:
                print(f"[UniNaVidHTTPPolicy] Unexpected predict_text response format: {result}")
                return ""

        except requests.exceptions.RequestException as e:
            print(f"[UniNaVidHTTPPolicy] predict_text request failed: {e}")
            return ""
        except Exception as e:
            print(f"[UniNaVidHTTPPolicy] predict_text failed: {e}")
            return ""

    def _prepare_image(self, rgb: np.ndarray) -> np.ndarray:
        """Prepare image for transmission.

        Ensures image is in the correct format (uint8, RGB, 3 channels).
        """
        if not isinstance(rgb, np.ndarray):
            rgb = np.array(rgb)

        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)

        if len(rgb.shape) == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        elif len(rgb.shape) == 3 and rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        elif len(rgb.shape) == 3 and rgb.shape[2] != 3:
            raise ValueError(f"Unexpected image shape: {rgb.shape}")

        return rgb

    def _encode_image(self, rgb_array: np.ndarray) -> str:
        """Encode image as base64 string for JSON transmission."""
        img = Image.fromarray(rgb_array)

        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()

        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_base64

    def _parse_response(self, result: Dict[str, Any]) -> str:
        """Parse server response to extract action string.

        Expected format: {"action": "forward" | "left" | "right" | "stop"}
        """
        if "action" in result:
            return str(result["action"]).lower()
        elif "actions" in result:
            actions = result["actions"]
            if isinstance(actions, list) and len(actions) > 0:
                return str(actions[0]).lower()

        print(f"[UniNaVidHTTPPolicy] Warning: Unexpected response format: {result}")
        return "stop"

    def _discrete_to_continuous(self, action_str: str) -> Action:
        """Convert discrete action string to Action object.

        Action mapping (aligned with VLN-CE):
        - forward: move forward 0.25m
        - left: turn left 15 degrees
        - right: turn right 15 degrees
        - stop: stop navigation

        Args:
            action_str: Action string from model

        Returns:
            Action with action_type set for STEP_ACTION mode
        """
        action_str = action_str.lower().strip()

        if action_str == 'stop':
            return Action(
                action_type='stop',
                stop=True,
                linear_velocity=0.0,
                angular_velocity=0.0,
            )
        elif action_str == 'forward':
            return Action(
                action_type='forward',
                linear_velocity=self.forward_speed,
                angular_velocity=0.0,
                lateral_velocity=0.0,
            )
        elif action_str == 'left':
            return Action(
                action_type='left',
                linear_velocity=0.0,
                angular_velocity=self.turn_angular_velocity,  # positive = left
                lateral_velocity=0.0,
            )
        elif action_str == 'right':
            return Action(
                action_type='right',
                linear_velocity=0.0,
                angular_velocity=-self.turn_angular_velocity,  # negative = right
                lateral_velocity=0.0,
            )
        else:
            print(f"[UniNaVidHTTPPolicy] Warning: Unknown action '{action_str}', using 'stop'")
            return Action(
                action_type='stop',
                stop=True,
                linear_velocity=0.0,
                angular_velocity=0.0,
            )

    def close(self):
        """Close the HTTP session."""
        if hasattr(self, 'session'):
            self.session.close()
