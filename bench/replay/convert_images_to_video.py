#!/usr/bin/env python3
"""Convert existing image sequences to MP4 videos.

Usage:
    python convert_images_to_video.py --input /path/to/replay_output --fps 30

This will recursively find all rgb/ and depth/ directories and convert them to rgb.mp4 and depth.mp4.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


def find_image_dirs(root: Path) -> List[Path]:
    """Find all directories containing rgb/ subdirectory."""
    result = []
    for rgb_dir in root.rglob("rgb"):
        if rgb_dir.is_dir():
            # Check if it contains frame images
            frames = list(rgb_dir.glob("frame_*.jpg")) + list(rgb_dir.glob("frame_*.png"))
            if frames:
                result.append(rgb_dir.parent)
    return sorted(set(result))


def get_frame_paths(image_dir: Path, prefix: str = "frame_") -> List[Path]:
    """Get sorted list of frame image paths."""
    patterns = [f"{prefix}*.jpg", f"{prefix}*.png", f"{prefix}*.jpeg"]
    frames = []
    for pattern in patterns:
        frames.extend(image_dir.glob(pattern))
    return sorted(frames, key=lambda p: p.stem)


def convert_rgb_to_video(
    rgb_dir: Path,
    output_path: Path,
    fps: int = 30,
) -> int:
    """Convert RGB image sequence to video.

    Returns:
        Number of frames written.
    """
    frames = get_frame_paths(rgb_dir)
    if not frames:
        print(f"  [SKIP] No RGB frames found in {rgb_dir}")
        return 0

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        print(f"  [ERROR] Failed to read first frame: {frames[0]}")
        return 0

    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    frame_count = 0
    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"  [WARN] Failed to read: {frame_path}")
            continue
        writer.write(img)
        frame_count += 1

    writer.release()
    return frame_count


def convert_depth_to_video(
    depth_dir: Path,
    output_path: Path,
    fps: int = 30,
) -> int:
    """Convert depth image sequence to video.

    Depth images are expected to be 16-bit PNG (millimeters).
    They will be normalized to 8-bit grayscale for video.

    Returns:
        Number of frames written.
    """
    frames = get_frame_paths(depth_dir)
    if not frames:
        print(f"  [SKIP] No depth frames found in {depth_dir}")
        return 0

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frames[0]), cv2.IMREAD_UNCHANGED)
    if first_frame is None:
        print(f"  [ERROR] Failed to read first depth frame: {frames[0]}")
        return 0

    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    frame_count = 0
    for frame_path in frames:
        depth = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"  [WARN] Failed to read: {frame_path}")
            continue

        # Handle different depth formats
        if depth.dtype == np.uint16:
            # 16-bit depth in mm, normalize to 0-255
            depth_normalized = (depth.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
        elif depth.dtype == np.uint8:
            depth_normalized = depth
        else:
            # Float depth, assume meters, convert to mm then normalize
            depth_mm = np.clip(depth * 1000.0, 0, 65535)
            depth_normalized = (depth_mm / 65535.0 * 255.0).astype(np.uint8)

        # Convert to BGR for video
        if len(depth_normalized.shape) == 2:
            depth_bgr = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
        else:
            depth_bgr = depth_normalized

        writer.write(depth_bgr)
        frame_count += 1

    writer.release()
    return frame_count


def convert_directory(
    base_dir: Path,
    fps: int = 30,
    skip_existing: bool = True,
    delete_images: bool = False,
) -> dict:
    """Convert a single directory's images to videos.

    Returns:
        dict with conversion stats.
    """
    rgb_dir = base_dir / "rgb"
    depth_dir = base_dir / "depth"
    rgb_video = base_dir / "rgb.mp4"
    depth_video = base_dir / "depth.mp4"

    stats = {"rgb_frames": 0, "depth_frames": 0, "skipped": False}

    # Convert RGB
    if rgb_dir.exists():
        if skip_existing and rgb_video.exists():
            print(f"  [SKIP] RGB video already exists: {rgb_video}")
            stats["skipped"] = True
        else:
            print(f"  Converting RGB: {rgb_dir} -> {rgb_video}")
            stats["rgb_frames"] = convert_rgb_to_video(rgb_dir, rgb_video, fps)
            if stats["rgb_frames"] > 0:
                print(f"  [OK] RGB: {stats['rgb_frames']} frames")
                if delete_images:
                    import shutil
                    shutil.rmtree(rgb_dir)
                    print(f"  [DEL] Removed {rgb_dir}")

    # Convert Depth
    if depth_dir.exists():
        if skip_existing and depth_video.exists():
            print(f"  [SKIP] Depth video already exists: {depth_video}")
        else:
            print(f"  Converting Depth: {depth_dir} -> {depth_video}")
            stats["depth_frames"] = convert_depth_to_video(depth_dir, depth_video, fps)
            if stats["depth_frames"] > 0:
                print(f"  [OK] Depth: {stats['depth_frames']} frames")
                if delete_images:
                    import shutil
                    shutil.rmtree(depth_dir)
                    print(f"  [DEL] Removed {depth_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert image sequences to MP4 videos."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Root directory containing replay output (will search recursively).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate (default: 30).",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-convert even if video already exists.",
    )
    parser.add_argument(
        "--delete-images",
        action="store_true",
        help="Delete image directories after successful conversion.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input path not found: {args.input}")

    print(f"Searching for image directories in: {args.input}")
    dirs = find_image_dirs(args.input)
    print(f"Found {len(dirs)} directories with images")

    total_rgb = 0
    total_depth = 0
    converted = 0
    skipped = 0

    for i, base_dir in enumerate(dirs):
        print(f"\n[{i+1}/{len(dirs)}] {base_dir}")
        stats = convert_directory(
            base_dir,
            fps=args.fps,
            skip_existing=not args.no_skip,
            delete_images=args.delete_images,
        )
        total_rgb += stats["rgb_frames"]
        total_depth += stats["depth_frames"]
        if stats["skipped"]:
            skipped += 1
        elif stats["rgb_frames"] > 0:
            converted += 1

    print(f"\n{'='*60}")
    print(f"Conversion complete:")
    print(f"  Directories processed: {len(dirs)}")
    print(f"  Converted: {converted}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Total RGB frames: {total_rgb}")
    print(f"  Total Depth frames: {total_depth}")


if __name__ == "__main__":
    main()
