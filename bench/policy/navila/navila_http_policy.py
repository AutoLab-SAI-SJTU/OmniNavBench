#!/usr/bin/env python3
"""
NaVILA HTTP Client Policy for OmniNavBench

This policy communicates with the NaVILA HTTP server to get navigation actions.
NaVILA outputs natural language commands which are parsed into discrete step actions.
"""

import base64
import io
import re
from collections import deque
from typing import Any, Deque, Dict

import numpy as np
import requests
from PIL import Image

from bench.policy.base import BasePolicy, Observation, Action


class NaVILAHTTPPolicy(BasePolicy):
    """
    HTTP client policy for NaVILA navigation model.
    
    Communicates with a NaVILA server to get navigation actions.
    Parses NaVILA's outputs into discrete step actions to match STEP_ACTION semantics.

    Reference implementation: NaVILA/evaluation/vlnce_baselines/navila_trainer.py
    """
    
    def __init__(
        self,
        server_url: str = "http://localhost:8002",
        timeout: float = 60.0,
    ):
        """
        Initialize NaVILA HTTP Policy.
        
        Args:
            server_url: URL of the NaVILA HTTP server
            timeout: Request timeout in seconds
        """
        super().__init__()

        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

        self._current_instruction: str = ""
        self._step_count: int = 0
        self._queue_actions: Deque[str] = deque()

        self._check_server_health()
    
    def _check_server_health(self) -> None:
        """Check if the server is healthy (no silent fallback)."""
        response = requests.get(f"{self.server_url}/health", timeout=10)
        if response.status_code != 200:
            raise RuntimeError(f"[NaVILAPolicy] Server unhealthy: status_code={response.status_code}, body={response.text[:200]}")
        data = response.json()
        print(f"[NaVILAPolicy] Server healthy: {data}")
    
    def reset(self, instruction: str = "", **kwargs) -> None:
        """
        Reset the policy for a new episode.
        
        Args:
            instruction: Navigation instruction for this episode
        """
        self._current_instruction = instruction
        self._step_count = 0
        self._queue_actions.clear()
        
        response = requests.post(
            f"{self.server_url}/reset",
            json={"instruction": instruction},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"[NaVILAPolicy] Reset failed: status_code={response.status_code}, body={response.text[:200]}")
        print(f"[NaVILAPolicy] Reset successful. Instruction: {instruction[:50]}...")
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode numpy image to base64 string."""
        # Convert to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        
        # Encode to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return image_b64

    def _enqueue_from_raw_response(self, raw: str) -> None:
        """
        Parse NaVILA text output into discrete steps, matching the original eval behavior.
        - Forward distances are interpreted in cm, quantized to {25, 50, 75}
        - Turn angles are interpreted in degree, quantized to {15, 30, 45}
        """
        text = raw.strip()

        patterns = {
            "stop": re.compile(r"\bstop\b", re.IGNORECASE),
            "forward": re.compile(r"\bis move forward\b", re.IGNORECASE),
            "left": re.compile(r"\bis turn left\b", re.IGNORECASE),
            "right": re.compile(r"\bis turn right\b", re.IGNORECASE),
        }
        action_type: str | None = None
        for name, pattern in patterns.items():
            if pattern.search(text):
                action_type = name
                break
        if action_type is None:
            raise ValueError(f"[NaVILAPolicy] Unrecognized model output (no action match): {text!r}")

        if action_type == "stop":
            self._queue_actions.append("stop")
            return

        if action_type == "forward":
            match = re.search(r"move forward (\d+) cm", text, flags=re.IGNORECASE)
            if match is None:
                raise ValueError(f"[NaVILAPolicy] Forward action missing 'X cm' distance: {text!r}")
            distance_cm = int(match.group(1))
            if (distance_cm % 25) != 0:
                distance_cm = min([25, 50, 75], key=lambda x: abs(x - distance_cm))
            steps = max(1, int(distance_cm // 25))
            self._queue_actions.extend(["forward"] * steps)
            return

        if action_type in ("left", "right"):
            match = re.search(rf"turn {action_type} (\d+) degree", text, flags=re.IGNORECASE)
            if match is None:
                raise ValueError(f"[NaVILAPolicy] Turn action missing 'X degree' angle: {text!r}")
            degree = int(match.group(1))
            if (degree % 15) != 0:
                degree = min([15, 30, 45], key=lambda x: abs(x - degree))
            steps = max(1, int(degree // 15))
            self._queue_actions.extend([action_type] * steps)
            return

        raise RuntimeError(f"[NaVILAPolicy] Unsupported action_type={action_type!r} for raw={text!r}")
    
    def act(self, observation: Observation) -> Action:
        """
        Get action from NaVILA server.
        
        Args:
            observation: Observation object containing rgb, instruction, etc.
        
        Returns:
            Action object with action_type for STEP_ACTION
        """
        self._step_count += 1
        
        if observation.rgb is None:
            raise ValueError("[NaVILAPolicy] observation.rgb is required for NaVILA")

        if self._queue_actions:
            next_action = self._queue_actions.popleft()
            return Action(action_type=next_action, stop=(next_action == "stop"), extra={"queued": True})

        inst = observation.instruction if observation.instruction else self._current_instruction
        if not inst:
            raise ValueError("[NaVILAPolicy] instruction is required (empty observation.instruction and no prior reset instruction)")
        
        rgb_b64 = self._encode_image(observation.rgb)
        
        response = requests.post(
            f"{self.server_url}/act",
            json={"rgb_base64": rgb_b64, "instruction": inst},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"[NaVILAPolicy] /act failed: status_code={response.status_code}, body={response.text[:200]}")

        payload = response.json()
        raw_response = payload.get("raw_response")
        if not isinstance(raw_response, str) or not raw_response.strip():
            raise ValueError(f"[NaVILAPolicy] Missing/empty raw_response from server: {payload}")

        self._enqueue_from_raw_response(raw_response)
        if not self._queue_actions:
            raise RuntimeError(f"[NaVILAPolicy] Parser produced empty action queue for raw_response={raw_response!r}")

        next_action = self._queue_actions.popleft()
        return Action(
            action_type=next_action,
            stop=(next_action == "stop"),
            extra={"raw_response": raw_response, "queued_remaining": len(self._queue_actions)},
        )
    
