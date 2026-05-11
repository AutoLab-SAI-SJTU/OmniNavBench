import argparse
import os
import sys
from pathlib import Path

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from OmniNav.local_paths import repo_path

os.environ['MODELSCOPE_DOMAIN'] = 'www.modelscope.ai'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the OmniNav checkpoint.")
    parser.add_argument("--model-id", default="chongchongjj/OmniNav", help="ModelScope model identifier.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_path("OmniNav", "checkpoint"),
        help="Cache directory for downloaded checkpoints.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    save_dir = args.output_dir.expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        from modelscope.hub.snapshot_download import snapshot_download

        print(f"Downloading {args.model_id} ...")
        path = snapshot_download(args.model_id, cache_dir=str(save_dir))
        print(f"Download succeeded. Model saved at: {path}")
        return 0
    except Exception as e:
        print(f"Download failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
