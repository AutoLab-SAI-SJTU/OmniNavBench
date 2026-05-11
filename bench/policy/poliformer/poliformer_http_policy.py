"""PoliFormer HTTP client policy for OmniNavBench.

This policy communicates with a PoliFormer HTTP server running in a separate process.
"""
from __future__ import annotations

import base64
import json
from typing import Optional, Dict, Any
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from bench.policy.base import BasePolicy, Observation, Action


class PoliFormerHTTPPolicy(BasePolicy):
    """PoliFormer policy that communicates via HTTP with a remote server."""

    def __init__(
        self,
        server_url: str = "http://localhost:8003",
        timeout: float = 30.0,
        max_retries: int = 3,
        forward_speed: float = 1.0,
        turn_angular_velocity: float = 2.0,
    ):
        super().__init__()
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.forward_speed = forward_speed
        self.turn_angular_velocity = turn_angular_velocity

        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self._check_health()
        print(f"[PoliFormerHTTPPolicy] Connected to server at {self.server_url}")

    def _check_health(self):
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5.0)
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Cannot connect to PoliFormer server at {self.server_url}: {e}")

    def reset(self, instruction: str = ""):
        super().reset(instruction)
        try:
            self.session.post(f"{self.server_url}/reset", json={"instruction": instruction}, timeout=self.timeout)
        except Exception as e:
            print(f"[PoliFormerHTTPPolicy] ⚠️  Reset request failed: {e}")

    def act(self, observation: Observation) -> Action:
        self.update_history(observation)

        if observation.rgb is None:
            return Action(stop=True)

        rgb_array = self._prepare_image(observation.rgb)
        image_base64 = self._encode_image(rgb_array)

        payload = {
            "instruction": observation.instruction,
            "image": image_base64,
            "image_shape": list(rgb_array.shape),
        }

        try:
            response = self.session.post(f"{self.server_url}/act", json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as e:
            print(f"[PoliFormerHTTPPolicy] ⚠️  Request failed: {e}")
            return Action(stop=True)
        # no verbose logging here

        print(f"[PoliFormerHTTPPolicy] Server response: {result}")
        action_str = self._parse_response(result)
        print(f"[PoliFormerHTTPPolicy] Converting action '{action_str}' to Action object")
        return self._discrete_to_continuous(action_str)

    def _prepare_image(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

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

        # Resize to PoliFormer training dimensions: 384x224
        # INTEL_CAMERA_WIDTH = 396, but training uses (396 - 396%32) = 384
        target_width = 384
        target_height = 224

        if rgb.shape[0] != target_height or rgb.shape[1] != target_width:
            pil_img = Image.fromarray(rgb)
            pil_img = pil_img.resize((target_width, target_height), Image.BILINEAR)
            rgb = np.array(pil_img)

        return rgb

    def _encode_image(self, rgb_array: np.ndarray) -> str:
        from PIL import Image
        import io

        img = Image.fromarray(rgb_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')

    def _parse_response(self, result: Dict[str, Any]) -> str:
        if "action" in result:
            action_str = str(result["action"]).lower()
            print(f"[PoliFormerHTTPPolicy] Parsed action from server: '{result['action']}' -> '{action_str}'")
            return action_str
        elif "actions" in result:
            actions = result["actions"]
            if isinstance(actions, list) and len(actions) > 0:
                action_str = str(actions[0]).lower()
                print(f"[PoliFormerHTTPPolicy] Parsed first action from list: '{actions[0]}' -> '{action_str}'")
                return action_str
        print(f"[PoliFormerHTTPPolicy] ⚠️  Unexpected response format: {result}, using 'forward'")
        return "forward"

    def _discrete_to_continuous(self, action_str: str) -> Action:
        action_str = action_str.lower()

        # Map PoliFormer action abbreviations to OmniNavBench action types
        # PoliFormer actions: ['m', 'r', 'l', 'b', 'end', 'sub_done', 'ls', 'rs', 'p', 'zm', 'zp', 'yp', 'ym', 'wp', 'wm', 'yms', 'zms', 'zps', 'yps', 'd']
        if action_str in ["end", "stop"]:
            print(f"[PoliFormerHTTPPolicy] Action mapping: '{action_str}' -> stop")
            return Action(action_type='stop', stop=True, linear_velocity=0.0, angular_velocity=0.0)
        elif action_str in ["m", "forward", "moveahead"]:  # 'm' = move ahead
            print(f"[PoliFormerHTTPPolicy] Action mapping: '{action_str}' -> forward")
            return Action(action_type='forward', linear_velocity=self.forward_speed)
        elif action_str in ["l", "left", "rotateleft"]:  # 'l' = rotate left
            print(f"[PoliFormerHTTPPolicy] Action mapping: '{action_str}' -> left")
            return Action(action_type='left', angular_velocity=self.turn_angular_velocity)
        elif action_str in ["r", "right", "rotateright"]:  # 'r' = rotate right
            print(f"[PoliFormerHTTPPolicy] Action mapping: '{action_str}' -> right")
            return Action(action_type='right', angular_velocity=-self.turn_angular_velocity)
        elif action_str == "b":  # 'b' = move back - map to forward for now since OmniNavBench may not support backward
            print(f"[PoliFormerHTTPPolicy] ⚠️  Action 'b' (move back) not supported, using forward instead")
            return Action(action_type='forward', linear_velocity=self.forward_speed)
        else:
            print(f"[PoliFormerHTTPPolicy] ⚠️  Unknown action '{action_str}', using forward as default")
            return Action(action_type='forward', linear_velocity=self.forward_speed)

    def close(self):
        if hasattr(self, 'session'):
            self.session.close()




