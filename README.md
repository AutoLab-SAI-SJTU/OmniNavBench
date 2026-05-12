<div align="center">

# OmniNavBench

**Beyond Isolation: A Unified Benchmark for General-Purpose Navigation**

[![RSS 2026](https://img.shields.io/badge/RSS-2026-blue)](https://roboticsconference.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.09441-red)](https://arxiv.org/abs/2605.09441)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Live-orange)](http://omninavbench.cloud-ip.cc/)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/AutoLab-SJTU/OmniNavBench)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

## 🔥 News

- **[2026.05]** 🎉 OmniNavBench is accepted to **RSS 2026**.
- **[2026.05]** Code release.
- **[2026.05]** Leaderboard live at <http://omninavbench.cloud-ip.cc/>.

## 📝 TODO

- [x] Code release
- [x] Leaderboard submission portal
- [ ] Paper release
- [ ] Dataset release
- [ ] Data-generation pipeline release
- [ ] Replay pipeline release
- [ ] Docker version release

## 🔎 Overview

Most embodied-navigation benchmarks isolate a single skill (PointNav, VLN, ObjectNav, SocialNav, Human Following, or EQA) on a single robot morphology, against shortest-path reference data. **OmniNavBench** breaks all three constraints at once: composite instructions that interleave six sub-task families, three robot embodiments, and reference trajectories collected from human teleoperation rather than A\* shortest-path planners.

**Three paradigm shifts:**

- 🧩 **Compositional complexity** — every instruction weaves together **at least two of six sub-task primitives** (PointNav, VLN, ObjectNav, SocialNav, Human Following, EQA), forcing agents to switch strategies mid-episode while satisfying overarching SocialNav / EQA constraints.
- 🤖 **Morphological universality & sensor flexibility** — the same instruction set runs on **H1 humanoid, Aliengo quadruped, and Carter wheeled** robots through a modular sensor interface (RGB-D, LiDAR, panoramic), across 170 environments blending 85 GRScenes synthetic assets and 85 real-world Matterport3D scans.
- 🧑‍✈️ **Naturalistic human demonstrations** — **1,779 expert trajectories collected via human teleoperation**, 16.7 m average length, 29.5 km cumulative, 24 hours of egocentric RGB-D and 2.6 M frames. The data captures exploratory glance, anticipatory avoidance, and other behaviours shortest-path planners cannot reproduce.

**At a glance:**

| | |
|---|---|
| Sub-task families | PointNav · VLN · ObjectNav · SocialNav · Human Following · EQA |
| Robot embodiments | H1 humanoid · Aliengo quadruped · Carter wheeled |
| Environments | 170 (85 GRScenes synthetic + 85 Matterport3D real) |
| Composite instructions | 1,779 base · 7,116 with 4 linguistic styles |
| Reference video | 1,700+ teleoperated demonstrations · 2.6 M frames |
| Trajectory-only runtime | scoring is offline; local eval and leaderboard submission go through the same `bench/evaluator/offline_test.py` code path |
| Bring your own policy | one HTTP endpoint to implement; reference adapters bundled as templates |

## 📋 Requirements

| Component | Version | Notes |
|---|---|---|
| OS | Linux | Vulkan-required; Windows not supported |
| Python | 3.11 | conda recommended |
| Isaac Sim | 5.0.0 | install via the [Isaac Lab pip-installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) |
| Isaac Lab | 2.3.0 | same guide; the `omni.isaac.matterport` extension under `IsaacLab/source/` is required |
| GPU | NVIDIA, CUDA 12.8 | ≥ 24 GB VRAM recommended for policy servers (Uni-NaVid: ~8 GB, OmniNav: ~12 GB) |
| RAM | ≥ 32 GB | Isaac Sim baseline |

Python packages: see [`pyproject.toml`](pyproject.toml). They install via `pip install -e .` after the Isaac Lab guide (which handles `isaacsim` / `isaaclab` themselves).

## 🛠️ Installation

### 1. Install Isaac Sim and Isaac Lab

Follow NVIDIA's official guide: [Isaac Lab — pip installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html). It walks you through creating a Python 3.11 conda env, pip-installing Isaac Sim, and installing Isaac Lab in one place.

> Tested with **Isaac Sim 5.0.0** + **Isaac Lab 2.3.0** + **Python 3.11**. Newer versions may also work but have not been verified for this benchmark.

### 2. Clone this repo and install the Python deps

```bash
git clone <this-repo> ~/OmniNavBench
cd ~/OmniNavBench
pip install -e .
```

`pip install -e .` resolves the runtime dependencies declared in `pyproject.toml` (without touching the Isaac Sim / Isaac Lab install from Step 1). The shell helper `load_local_paths.sh` (see next section) additionally puts the repo root on `PYTHONPATH` so `import bench`, `import OmniNav`, `import OmniNavExt` resolve from any working directory.

## 🔧 Configuring for Your Machine

**Data lives anywhere on disk.** This repo and the dataset are independent — the dataset can sit on a separate drive (e.g. `/media/<user>/<some-disk>/OmniNavBench`), inside your home directory, or anywhere else. You just tell the runner where to look via two environment variables.

After cloning, copy the template and edit it once:

```bash
cp local_paths.env.example local_paths.env
$EDITOR local_paths.env
```

The two paths you must set:

```bash
# local_paths.env
OMNINAV_BENCH_DATASET_ROOT="/absolute/path/to/OmniNavBench"   # OmniNavBench dataset root
OMNINAV_SCENE_ROOT="/absolute/path/to/Assets"                 # GRScenes + Matterport3D scene assets
#OMNINAV_ISAACLAB_SOURCE="/absolute/path/to/IsaacLab/source"  # optional; auto-detected if Isaac Lab is on a standard path
```

Source it once per shell:

```bash
source load_local_paths.sh
```

This sets `OMNINAV_REPO_ROOT`, prepends the repo to `PYTHONPATH`, and exports the variables from `local_paths.env`. After this, `runBench.py` picks up the data and scene paths automatically — no CLI flags required.

> **Precedence:** explicit CLI flags (`--omninavbench-root`, `--scene-root`) override the env vars. The env vars override nothing else — if neither is set when you use `--omninavbench`, `runBench.py` exits with a clear error pointing you back to this section.

## 📦 Data

### OmniNavBench (primary)

Download the dataset from [AutoLab-SJTU/OmniNavBench](https://huggingface.co/datasets/AutoLab-SJTU/OmniNavBench) (HuggingFace) and unpack it anywhere on disk. Point at the unpack location via `OMNINAV_BENCH_DATASET_ROOT` in `local_paths.env` (or pass `--omninavbench-root /path` on the CLI). The expected layout under that root:

```
OmniNavBench/
├── annotations/                              # scenario JSONs consumed by runBench.py
│   ├── train/                                # with GT — local offline scoring is supported
│   │   └── {original,concise,verbose,first_person}/
│   │       └── {human,dog,car}/              # robot dirs: human=H1, dog=Aliengo, car=Carter
│   │           └── <scene_id>/
│   │               └── final_episode_N.json
│   └── test/                                 # sanitized (no GT) — submit results to the leaderboard
│       └── <style>/<robot>/<scene>/...
│
└── videos/                                   # GT replay videos (optional, train split only)
    └── train/...
```

The `human` / `dog` / `car` directories are **robot embodiments**, not object types. Mapping:

| `--robot` flag | Dataset directory | Robot model |
|---|---|---|
| `h1` | `human/` | Unitree H1 humanoid |
| `aliengo` | `dog/` | Unitree Aliengo quadruped |
| `carter` | `car/` | NVIDIA Carter wheeled |

### Scene assets

OmniNavBench uses a hybrid suite of **170 environments**: 85 high-fidelity synthetic assets from [GRScenes](https://github.com/OpenRobotLab/GRUtopia) and 85 photorealistic real-world scans from [Matterport3D](https://github.com/niessner/Matterport). Both download separately and live under the `OMNINAV_SCENE_ROOT` directory you set in [Configuring for Your Machine](#-configuring-for-your-machine). Matterport3D usage is governed by the [Matterport3D Terms of Use](https://github.com/niessner/Matterport).

## 🚀 Quick Start

A no-server smoke run end-to-end — verifies the simulator + I/O wiring without any policy server. The `forward` policy just drives the robot straight forward.

```bash
source load_local_paths.sh   # once per shell — exports the data/scene roots
python runBench.py \
    --omninavbench --mode test --robot h1 --style original \
    --config configs/aliengoh1_test.yaml \
    --output results/smoke/ \
    --policy forward \
    --headless
```

The dataset path comes from `OMNINAV_BENCH_DATASET_ROOT` (set in `local_paths.env`); pass `--omninavbench-root /path` only if you want to override it for this run.

## 🔬 Running Evaluations

### Full policy evaluation (test mode → submit to leaderboard)

Your policy runs as an HTTP server in its own process. `runBench.py` queries it once per simulation step. Start the server first, then point `runBench.py` at its URL.

```bash
# 1) Start your policy server (example below uses a bundled reference adapter)
python -m bench.policy.<your_adapter>.<your_server> --port <port> [your-args]

# 2) Run the benchmark (dataset path picked up from local_paths.env)
python runBench.py \
    --omninavbench --mode test --robot h1 --style original \
    --config configs/aliengoh1_test.yaml \
    --output results/my_run/ \
    --policy <your_policy> \
    --<your_policy>-server-url http://localhost:<port> \
    --headless
```

The `--output` directory after a run contains per-episode trajectories and a minimal `summary.json` (steps / time only). **No scoring fields are written** by the runtime — scoring is exclusively offline.

### Bringing your own policy

The benchmark talks to any policy via a small HTTP protocol. To benchmark a new policy: (1) write a server that exposes the same step/action endpoint that `bench/policy/<reference_adapter>/` uses, (2) add a `--policy <name>` choice in `runBench.py` that wires its URL flag, (3) run as above. Use any of the bundled reference adapters as a copy-paste template.

### Reference policy adapters (already wired)

These are external policies we tested as part of building this benchmark. They are **examples**, not the benchmark itself — bring your own policy to actually evaluate something new.

| `--policy` | Reference for | Notes |
|---|---|---|
| `forward` | smoke-test sanity check | built-in, no server, drives the robot straight forward |
| `uninavid` | Uni-NaVid (3rd-party) | external repo + checkpoint required |
| `mtu3d` | MTU3D (3rd-party) | external repo + checkpoint required |
| `poliformer` | PoliFormer (3rd-party) | external repo + checkpoint required |
| `omninav` | OmniNav (3rd-party) | external repo + checkpoint required |

Server ports are user-chosen — start the server with `--port <port>` and pass the matching URL to `runBench.py` via `--<policy>-server-url`. Per-policy launch commands and required checkpoints are in `HowtoTestModel.md`.

### Robot ↔ config compatibility

| `--robot` | Recommended `--config` |
|---|---|
| `h1` | `configs/aliengoh1_test.yaml` |
| `aliengo` | `configs/aliengoh1_test.yaml` |
| `carter` | `configs/carter_v1_test.yaml` |

### Batch helpers (multiple scenes or styles)

For larger sweeps, two thin wrappers around `runBench.py` ship in the repo root.

`benchtestbatch.sh` runs one `runBench.py` per scene in parallel across the GPUs `nvidia-smi -L` reports. Reads `OMNINAV_BENCH_DATASET_ROOT` from `local_paths.env` and walks `annotations/<mode>/<style>/<robot>/<scene>/` itself.

```bash
# forward smoke-test across every scene/robot in the test split
./benchtestbatch.sh --mode test --style original

# evaluate your own server-backed policy
./benchtestbatch.sh --mode test --style concise \
    --policy omninav --server-url http://localhost:<port>
```

Selected flags: `--policy NAME` (default `forward`), `--robot h1,aliengo,carter` (comma-separated, defaults to all three), `--mode train|test` (default `test`), `--style original|concise|verbose|first_person` (default `original`), `--workers-per-gpu N` (default 4), `--num-gpus N` (default = autodetected), `--server-url URL`, `--skip-completed`.

`benchteststyle.sh` sweeps one robot across all four instruction styles via the `--omninavbench` shortcut:

```bash
./benchteststyle.sh --robot aliengo --policy forward
./benchteststyle.sh --robot h1 --policy omninav --server-url http://localhost:<port>
```

Both scripts pass exactly one `--<policy>-server-url` flag (the one matching `--policy`); for `--policy forward`, `--server-url` is ignored.

### Local scoring (train split only — has GT)

Run the benchmark in train mode, then score offline:

```bash
# 1) Run on train (GT present in private envset)
python runBench.py \
    --omninavbench --mode train --robot aliengo --style concise \
    --config configs/aliengoh1_test.yaml \
    --output results/my_train_run/ \
    --policy <your_policy> --<your_policy>-server-url http://localhost:<port> \
    --headless

# 2) Score against the (with-GT) train annotations
python -m bench.evaluator.offline_test \
    --private "$OMNINAV_BENCH_DATASET_ROOT/annotations/train/concise/dog" \
    --results results/my_train_run/ \
    --output results/my_train_run/scoring.json
```

The scorer outputs `sr`, `csr`, `softsr`, `spl`, `ne`, `osr`, `social_violation_ratio`, `eqa_accuracy`, plus per-episode breakdowns. **For the test split, do not run the offline scorer locally** — submit your `--output` directory to the leaderboard at <http://omninavbench.cloud-ip.cc/>; the same `offline_test.py` runs server-side against the private GT.

## 📤 Per-Episode Output Schema

Each `<scenario_id>.json` in `--output` contains only embodiment-independent runtime metadata:

```json
{
  "scenario_id": "matterport_11",
  "source_envset": "/path/to/episode.json",
  "instruction": "Follow the man ahead of you ...",
  "robot_type": "h1",
  "initial_pose": {"position": [...], "orientation_deg": 0.0},
  "termination_reason": "stop_action | timeout | max_steps",
  "steps": 123,
  "time_s": 45.6,
  "path_length": 12.3,
  "stop_step": 98,
  "trajectory": [
    {"step": 0, "time_s": 0.0, "position": [x,y,z], "orientation": [w,x,y,z]},
    ...
  ]
}
```

`success` / `distance_to_goal` and any aggregate score fields are **deliberately not written** so that local development on the train split and remote evaluation on the test split compute metrics through the exact same `bench/evaluator/offline_test.py` code path.

## 📂 Repository Layout

```
OmniNavBench/
├── runBench.py                   # main benchmark runner (this is what you run)
├── load_local_paths.sh           # env-var loader (source it before running)
├── local_paths.env.example       # template — copy to local_paths.env and edit
│
├── configs/                      # robot/physics configs (aliengoh1_test.yaml, carter_v1_test.yaml)
├── HowtoTestModel.md             # per-policy launch commands
│
├── bench/
│   ├── evaluator/                # benchmark runner + offline scorer
│   ├── metrics/                  # SR/SPL/CSR/SoftSR etc.
│   ├── policy/                   # one HTTP-server module per supported policy
│   ├── datasets/adapters/        # dataset → envset adapters
│   └── replay/                   # video rendering pipeline
│
├── OmniNav/                       # Isaac Sim integration core
└── OmniNavExt/                    # Isaac Sim extensions, robot configs, scene loaders
```

## ❤️ Acknowledgements

### Foundations

- [InternUtopia](https://github.com/InternRobotics/InternUtopia) — the Isaac Sim integration scaffolding under `OmniNav/` and `OmniNavExt/` (config schema, simulator runner, sensor / robot abstractions, extension lifecycle) is built on it. We extended it heavily for OmniNavBench, adding the NavMesh baking pipeline, scenario / scene loader, and virtual-human spawning + control stack.
- [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) — simulation platform.
- [Matterport3D](https://niessner.github.io/Matterport/) — 85 of the 170 real-world scene scans.
- [GRScenes](https://github.com/OpenRobotLab/GRUtopia) — 85 of the 170 synthetic scene assets.

### Reference policy implementations

- [Uni-NaVid](https://github.com/jzhzhang/NaVid-VLN-CE)
- [MTU3D](https://github.com/bigai-research/MTU3D)
- [PoliFormer](https://github.com/allenai/PoliFormer)
- [OmniNav](https://github.com/amap-cvlab/OmniNav)

Thanks to all authors for releasing high-quality code.

## 📄 License

OmniNavBench code is released under the MIT License — see [LICENSE](LICENSE). Note that the bundled scene data is governed by the [Matterport3D Terms of Use](https://github.com/niessner/Matterport) and is not redistributed by this repository.
