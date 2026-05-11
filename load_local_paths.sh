#!/usr/bin/env bash
set -euo pipefail

OMNINAV_REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OMNINAV_REPO_ROOT

# Make `import bench`, `import OmniNav`, `import OmniNavExt` resolvable from any cwd.
export PYTHONPATH="$OMNINAV_REPO_ROOT:${PYTHONPATH:-}"

LOCAL_PATHS_ENV_FILE="$OMNINAV_REPO_ROOT/local_paths.env"
LOCAL_PATHS_EXAMPLE_FILE="$OMNINAV_REPO_ROOT/local_paths.env.example"

if [[ -f "$LOCAL_PATHS_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$LOCAL_PATHS_ENV_FILE"
  set +a
elif [[ -f "$LOCAL_PATHS_EXAMPLE_FILE" ]]; then
  echo "[load_local_paths] local_paths.env not found." >&2
  echo "[load_local_paths] Run: cp local_paths.env.example local_paths.env" >&2
  echo "[load_local_paths] Then edit local_paths.env to point at your data, and re-source this script." >&2
fi

# Backward-compat: anyone with a pre-rename shell or local_paths.env
# may still have UNINAV_* set. Mirror them onto OMNINAV_* if the new
# names are not already defined, so old environments keep working.
for _legacy_var in UNINAV_SCENE_ROOT UNINAV_BENCH_DATASET_ROOT UNINAV_ISAACLAB_SOURCE UNINAV_LOG_ACTIONS UNINAV_BENCH_TEST_MODE; do
  _new_var="OMNINAV_${_legacy_var#UNINAV_}"
  if [[ -n "${!_legacy_var:-}" && -z "${!_new_var:-}" ]]; then
    export "$_new_var"="${!_legacy_var}"
  fi
done
unset _legacy_var _new_var
