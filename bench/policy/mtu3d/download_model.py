#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from OmniNav.local_paths import repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the MTU3D checkpoint from Hugging Face.")
    parser.add_argument("--model-id", default="bigai/MTU3D", help="Hugging Face model identifier.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_path("outputs", "checkpoints", "MTU3D"),
        help="Target directory for the downloaded checkpoint.",
    )
    return parser.parse_args()


def main() -> int:
    from huggingface_hub import snapshot_download

    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"Model downloaded to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
