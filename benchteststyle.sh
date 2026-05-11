#!/usr/bin/env bash
set -euo pipefail

# benchteststyle.sh — sweep across the four instruction styles for one robot.
#
# Each iteration runs the full envset for (mode, robot, style) via the
# --omninavbench shortcut, so the dataset root comes from
# $OMNINAV_BENCH_DATASET_ROOT (set in local_paths.env).
#
# Usage:
#   ./benchteststyle.sh [--policy NAME] [--robot h1|aliengo|carter]
#                       [--mode train|test] [--server-url URL]
#                       [--output-root DIR]

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./load_local_paths.sh
source "$REPO_ROOT/load_local_paths.sh"

POLICY="${POLICY:-forward}"
ROBOT="${ROBOT:-aliengo}"
MODE="${MODE:-test}"
SERVER_URL=""
OUTPUT_ROOT=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --policy)      POLICY="$2"; shift 2 ;;
    --robot)       ROBOT="$2"; shift 2 ;;
    --mode)        MODE="$2"; shift 2 ;;
    --server-url)  SERVER_URL="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$REPO_ROOT/results/style_${POLICY}_${MODE}_${ROBOT}"
fi

declare -A ROBOT_CONFIGS=(
  ["h1"]="configs/aliengoh1_test.yaml"
  ["aliengo"]="configs/aliengoh1_test.yaml"
  ["carter"]="configs/carter_v1_test.yaml"
)

if [[ -z "${ROBOT_CONFIGS[$ROBOT]:-}" ]]; then
  echo "Unknown --robot '$ROBOT' (expected h1 / aliengo / carter)." >&2
  exit 1
fi
CONFIG="$REPO_ROOT/${ROBOT_CONFIGS[$ROBOT]}"

build_server_url_args() {
  [[ -z "$SERVER_URL" ]] && return 0
  case "$POLICY" in
    uninavid)    echo "--uninavid-server-url $SERVER_URL" ;;
    mtu3d)       echo "--mtu3d-server-url $SERVER_URL" ;;
    poliformer)  echo "--poliformer-server-url $SERVER_URL" ;;
    omninav)     echo "--omninav-server-url $SERVER_URL" ;;
    forward)
      echo "Note: --server-url is ignored for --policy forward." >&2
      ;;
    *)
      echo "Note: --server-url has no flag mapping for --policy $POLICY." >&2
      ;;
  esac
}

mkdir -p "$OUTPUT_ROOT"

for STYLE in original concise verbose first_person; do
  out="$OUTPUT_ROOT/$STYLE"
  log="$REPO_ROOT/log_test_${POLICY}_${STYLE}_${ROBOT}.txt"
  mkdir -p "$out"
  echo "[STYLE=$STYLE] -> $out (log: $log)"

  # shellcheck disable=SC2046
  python "$REPO_ROOT/runBench.py" \
    --omninavbench --mode "$MODE" --robot "$ROBOT" --style "$STYLE" \
    --config "$CONFIG" \
    --output "$out" \
    --policy "$POLICY" \
    $(build_server_url_args) \
    --headless \
    > "$log" 2>&1
done
