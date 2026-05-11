#!/usr/bin/env bash
set -euo pipefail

# benchtestbatch.sh — multi-GPU batch test runner for OmniNavBench.
#
# Per scene, runs `python runBench.py --envset <scene-dir> ...` in parallel
# across the available GPUs, sourcing the dataset root from
# $OMNINAV_BENCH_DATASET_ROOT (set in local_paths.env) and following the
# HuggingFace layout: annotations/<mode>/<style>/<robot-data-dir>/<scene>/.
#
# Usage:
#   ./benchtestbatch.sh [--policy NAME] [--robot LIST] [--mode train|test]
#                       [--style original|concise|verbose|first_person]
#                       [--workers-per-gpu N] [--num-gpus N]
#                       [--server-url URL] [--output-root DIR]
#                       [--skip-completed]

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./load_local_paths.sh
source "$REPO_ROOT/load_local_paths.sh"

POLICY="${POLICY:-forward}"
ROBOTS="${ROBOTS:-h1,aliengo,carter}"
MODE="${MODE:-test}"
STYLE="${STYLE:-original}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
SERVER_URL=""
OUTPUT_ROOT=""
SKIP_COMPLETED=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --policy)          POLICY="$2"; shift 2 ;;
    --robot)           ROBOTS="$2"; shift 2 ;;
    --mode)            MODE="$2"; shift 2 ;;
    --style)           STYLE="$2"; shift 2 ;;
    --workers-per-gpu) WORKERS_PER_GPU="$2"; shift 2 ;;
    --num-gpus)        NUM_GPUS="$2"; shift 2 ;;
    --server-url)      SERVER_URL="$2"; shift 2 ;;
    --output-root)     OUTPUT_ROOT="$2"; shift 2 ;;
    --skip-completed)  SKIP_COMPLETED=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${OMNINAV_BENCH_DATASET_ROOT:-}" ]]; then
  echo "OMNINAV_BENCH_DATASET_ROOT is not set; configure local_paths.env first." >&2
  exit 1
fi

if [[ -z "$NUM_GPUS" || "$NUM_GPUS" -eq 0 ]]; then
  echo "Could not detect any GPU; pass --num-gpus N." >&2
  exit 1
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$REPO_ROOT/results/batch_${POLICY}_${MODE}_${STYLE}"
fi

declare -A ROBOT_CONFIGS=(
  ["h1"]="configs/aliengoh1_test.yaml"
  ["aliengo"]="configs/aliengoh1_test.yaml"
  ["carter"]="configs/carter_v1_test.yaml"
)

# Public --robot name -> dataset directory under annotations/<mode>/<style>/.
declare -A ROBOT_DATA_DIRS=(
  ["h1"]="human"
  ["aliengo"]="dog"
  ["carter"]="car"
)

LOG_DIR="$REPO_ROOT/logs/batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Logging to $LOG_DIR"

# Forward only the server-url flag matching the selected policy. Empty when
# no --server-url was given.
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

run_scene_task() {
  local robot=$1 scene=$2 worker_id=$3
  local gpu_id=$((worker_id / WORKERS_PER_GPU))
  local config_file="${ROBOT_CONFIGS[$robot]}"
  local data_dir="${ROBOT_DATA_DIRS[$robot]}"
  local scene_dir="$OMNINAV_BENCH_DATASET_ROOT/annotations/$MODE/$STYLE/$data_dir/$scene"
  local output_dir="$OUTPUT_ROOT/$robot/$scene"
  local log_file="$LOG_DIR/${robot}_${scene}.log"

  if [[ ! -d "$scene_dir" ]]; then
    echo "[Worker $worker_id] Skipping $robot/$scene (no scene dir)"
    return
  fi

  if [[ "$SKIP_COMPLETED" == "true" && -f "$output_dir/summary.json" ]]; then
    echo "[Worker $worker_id] Skipping $robot/$scene (already done)"
    return
  fi

  mkdir -p "$output_dir"
  echo "[Worker $worker_id] START $robot/$scene (GPU $gpu_id)"

  # shellcheck disable=SC2046
  if ! CUDA_VISIBLE_DEVICES=$gpu_id python "$REPO_ROOT/runBench.py" \
        --config "$REPO_ROOT/$config_file" \
        --envset "$scene_dir" \
        --output "$output_dir" \
        --policy "$POLICY" \
        $(build_server_url_args) \
        --headless \
        --no-trajectory \
        > "$log_file" 2>&1; then
    echo "[Worker $worker_id] FAILED $robot/$scene"
    mv "$log_file" "${log_file}.failed"
  else
    echo "[Worker $worker_id] DONE  $robot/$scene"
  fi
}

run_robot_batch() {
  local robot=$1
  if [[ -z "${ROBOT_CONFIGS[$robot]:-}" ]]; then
    echo "Warning: unknown robot '$robot', skipping." >&2
    return
  fi
  local data_dir="${ROBOT_DATA_DIRS[$robot]}"
  local annot_root="$OMNINAV_BENCH_DATASET_ROOT/annotations/$MODE/$STYLE/$data_dir"

  echo "=================================================="
  echo "Robot: $robot   Config: ${ROBOT_CONFIGS[$robot]}"
  echo "Annotations: $annot_root"
  echo "=================================================="

  if [[ ! -d "$annot_root" ]]; then
    echo "Warning: $annot_root not found, skipping $robot." >&2
    return
  fi

  mapfile -t SCENES < <(find "$annot_root" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort)
  local num_scenes=${#SCENES[@]}
  if [[ $num_scenes -eq 0 ]]; then
    echo "No scenes found for $robot."
    return
  fi
  echo "Found $num_scenes scenes."

  local total_workers=$((NUM_GPUS * WORKERS_PER_GPU))
  echo "Launching $total_workers workers across $NUM_GPUS GPUs..."

  local pids=()
  for (( w=0; w<total_workers; w++ )); do
    (
      for (( i=0; i<num_scenes; i++ )); do
        if (( i % total_workers == w )); then
          run_scene_task "$robot" "${SCENES[$i]}" "$w"
        fi
      done
    ) &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  echo "Robot $robot completed."
}

IFS=',' read -ra ROBOT_LIST <<< "$ROBOTS"
for robot in "${ROBOT_LIST[@]}"; do
  run_robot_batch "$robot"
done

echo "Batch testing completed."
