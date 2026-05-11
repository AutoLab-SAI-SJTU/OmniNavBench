#!/usr/bin/env python3
"""
Test script for UniNaVid Waypoint Server and Policy.

This script tests the waypoint prediction pipeline using sample images.

Usage:
    # First start the server:
    cd /path/to/Uni-NaVid_waypoints
    python -m bench.policy.uninavid_waypoint.uninavid_waypoint_server \
        --uninavid_path /path/to/Uni-NaVid_waypoints \
        --model_path /path/to/Uni-NaVid_waypoints/model_zoo/uninavid-7b-omninav-waypoint \
        --model_base /path/to/Uni-NaVid_waypoints/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2 \
        --port 8001 --debug

    # Then run this test:
    cd $OMNINAV_REPO_ROOT
    python bench/policy/uninavid_waypoint/test_waypoint_policy.py
"""

import os
import sys
import json
import numpy as np
import requests
import cv2
from pathlib import Path

# Add OmniNavBench to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def test_server_health(server_url: str = "http://localhost:8001"):
    """Test server health endpoint."""
    print("\n" + "="*60)
    print("Testing Server Health")
    print("="*60)

    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        response.raise_for_status()
        result = response.json()
        print(f"Server status: {result}")

        if result.get("model_loaded"):
            print("✓ Model loaded successfully")
            return True
        else:
            print("✗ Model not loaded")
            return False
    except Exception as e:
        print(f"✗ Server connection failed: {e}")
        return False


def test_server_reset(server_url: str = "http://localhost:8001"):
    """Test server reset endpoint."""
    print("\n" + "="*60)
    print("Testing Server Reset")
    print("="*60)

    instruction = "Follow the man ahead of you until he stops."

    try:
        response = requests.post(
            f"{server_url}/reset",
            json={"instruction": instruction, "task_type": "vln"},
            timeout=5
        )
        response.raise_for_status()
        result = response.json()
        print(f"Reset result: {result}")
        print("✓ Reset successful")
        return True
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        return False


def test_server_act(server_url: str = "http://localhost:8001", image_path: str = None):
    """Test server act endpoint with a sample image."""
    print("\n" + "="*60)
    print("Testing Server Act")
    print("="*60)

    # Create or load test image
    if image_path and os.path.exists(image_path):
        print(f"Loading image from: {image_path}")
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print("Creating synthetic test image (480x640)")
        # Create a simple synthetic image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some features
        cv2.rectangle(img, (200, 100), (440, 400), (100, 100, 100), -1)  # Gray rectangle
        cv2.circle(img, (320, 250), 50, (200, 150, 100), -1)  # Brown circle
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    instruction = "Follow the man ahead of you until he stops."

    # Encode image
    import base64
    import io
    from PIL import Image

    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    try:
        response = requests.post(
            f"{server_url}/act",
            json={
                "instruction": instruction,
                "image": image_base64,
                "image_shape": list(img.shape)
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()

        print(f"\nServer Response:")
        print(f"  Step: {result.get('step')}")
        print(f"  Inference time: {result.get('inference_time', 0):.3f}s")
        print(f"  Waypoints ({len(result.get('waypoints', []))}):")

        waypoints = result.get('waypoints', [])
        arrive_probs = result.get('arrive_probs', [])

        for i, wp in enumerate(waypoints):
            prob = arrive_probs[i] if i < len(arrive_probs) else 0
            print(f"    WP{i+1}: x={wp[0]:.3f}m, y={wp[1]:.3f}m, yaw={wp[2]:.3f}rad, arrive={prob:.3f}")

        print("✓ Act successful")
        return result
    except Exception as e:
        print(f"✗ Act failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_policy_integration(server_url: str = "http://localhost:8001"):
    """Test full policy integration."""
    print("\n" + "="*60)
    print("Testing Policy Integration")
    print("="*60)

    try:
        from bench.policy.uninavid_waypoint import UniNaVidWaypointHTTPPolicy
        from bench.policy.base import Observation

        # Create policy
        policy = UniNaVidWaypointHTTPPolicy(
            server_url=server_url,
            debug=True,
            debug_dir="debug_test_waypoint",
            debug_interval=1
        )

        # Reset
        instruction = "Follow the man ahead of you until he stops."
        policy.reset(instruction)
        print("✓ Policy reset successful")

        # Create synthetic observation
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(rgb, (200, 100), (440, 400), (100, 100, 100), -1)

        obs = Observation(
            rgb=rgb,
            position=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),  # w, x, y, z
            instruction=instruction,
            step=1
        )

        # Get action
        action = policy.act(obs)

        print(f"\nPolicy Action:")
        print(f"  action_type: {action.action_type}")
        print(f"  stop: {action.stop}")

        if action.extra:
            print(f"  controller: {action.extra.get('controller')}")
            path_points = action.extra.get('path_points', [])
            print(f"  path_points ({len(path_points)}):")
            for i, pt in enumerate(path_points):
                print(f"    Point{i+1}: ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})")

        print("✓ Policy integration successful")

        policy.close()
        return True

    except Exception as e:
        print(f"✗ Policy integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test UniNaVid Waypoint Server and Policy")
    parser.add_argument("--server_url", default="http://localhost:8001", help="Server URL")
    parser.add_argument("--image", default=None, help="Path to test image (optional)")
    parser.add_argument("--skip_policy", action="store_true", help="Skip policy integration test")
    args = parser.parse_args()

    print("="*60)
    print("UniNaVid Waypoint Server/Policy Test")
    print("="*60)
    print(f"Server URL: {args.server_url}")

    # Test 1: Health check
    if not test_server_health(args.server_url):
        print("\n❌ Server not available. Please start the server first.")
        print("\nUsage:")
        print("  cd /path/to/Uni-NaVid_waypoints")
        print("  python -m bench.policy.uninavid_waypoint.uninavid_waypoint_server \\")
        print("      --uninavid_path /path/to/Uni-NaVid_waypoints \\")
        print("      --model_path /path/to/Uni-NaVid_waypoints/model_zoo/uninavid-7b-omninav-waypoint \\")
        print("      --model_base /path/to/Uni-NaVid_waypoints/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2 \\")
        print("      --port 8001 --debug")
        return

    # Test 2: Reset
    test_server_reset(args.server_url)

    # Test 3: Act
    test_server_act(args.server_url, args.image)

    # Test 4: Policy integration
    if not args.skip_policy:
        test_policy_integration(args.server_url)

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
