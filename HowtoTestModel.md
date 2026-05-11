# OmniNavBench Policy Test Notes

`runBench.py` only ships the models that actually live in `bench/policy/`, plus the local `forward` baseline.

## Supported Policies

| `--policy` | Policy class | Action form |
| --- | --- | --- |
| `forward` | `ForwardPolicy` | `step_action` |
| `uninavid` | `UniNaVidHTTPPolicy` | `step_action` |
| `mtu3d` | `MTU3DHTTPPolicy` | `waypoint` |
| `poliformer` | `PoliFormerHTTPPolicy` | `step_action` |
| `omninav` | `OmniNavHTTPPolicy` | `waypoint` |

## OmniNavBench Dataset (`--omninavbench`)

`runBench.py` exposes a `--omninavbench` group of arguments that points at the OmniNavBench data directory automatically, so you do not need to pass `--envset` by hand.

Arguments:

| Argument | Values | Description |
| --- | --- | --- |
| `--mode` | `train` / `test` | **Required.** `test` reads from `annotations/test/` (GT stripped — runs locally only and must be submitted to the server for scoring); `train` reads from `annotations/train/` (with GT — can be self-scored locally with `bench/evaluator/offline_test.py`). |
| `--robot` | `h1` / `aliengo` / `carter` | Robot embodiment. Maps to data directories `human/` / `dog/` / `car/`. Make sure `--config` matches the robot. |
| `--style` | `original` / `concise` / `verbose` / `first_person` | Instruction style; defaults to `original`. |
| `--omninavbench-root` | path | OmniNavBench data root. Defaults to `$OMNINAV_BENCH_DATASET_ROOT` (set in `local_paths.env`). |

`runBench.py` **no longer writes success / SPL fields** to disk — scoring always goes through the offline evaluator:

```bash
python -m bench.evaluator.offline_test --private <envset_with_GT> --results <results_dir> --output <scoring_result.json>
```

### Test mode (submit to server for scoring)

```bash
python runBench.py \
    --omninavbench --mode test --robot h1 --style original \
    --config configs/aliengoh1_test.yaml \
    --output results/omninav_test_h1/ \
    --policy omninav --omninav-server-url http://localhost:<port> \
    --headless
```

- Video recording is **off by default** (so inference videos do not accidentally end up in the submission package). Add `--record-video` if you want the recording locally.
- The output directory only contains trajectories / step counts / durations — no scores. Scores come from the server running `offline_test.py` after submission.

### Train mode (local self-scoring)

```bash
# 1) Run the bench and produce trajectory files
python runBench.py \
    --omninavbench --mode train --robot aliengo --style concise \
    --config configs/aliengoh1_test.yaml \
    --output results/omninav_train_aliengo/ \
    --policy omninav --omninav-server-url http://localhost:<port> \
    --headless

# 2) Run the offline evaluator against the GT envset
python -m bench.evaluator.offline_test \
    --private "$OMNINAV_BENCH_DATASET_ROOT/annotations/train/concise/dog" \
    --results results/omninav_train_aliengo/ \
    --output results/omninav_train_aliengo/scoring.json
```

Recommended `--robot` ↔ `--config` pairings:

| `--robot` | Recommended `--config` |
| --- | --- |
| `h1` | `configs/aliengoh1_test.yaml` |
| `aliengo` | `configs/aliengoh1_test.yaml` |
| `carter` | `configs/carter_v1_test.yaml` |

## Forward Baseline

```bash
python runBench.py \
    --omninavbench --mode test --robot h1 --style original \
    --config configs/aliengoh1_test.yaml \
    --output results/forward_test_h1/ \
    --policy forward \
    --headless
```

## Uni-NaVid

```bash
python -m bench.policy.uninavid.uninavid_server \
    --model_path /path/to/Uni-NaVid/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2 \
    --uninavid_path /path/to/Uni-NaVid \
    --port <port>

python runBench.py \
    --omninavbench --mode test --robot h1 --style original \
    --config configs/aliengoh1_test.yaml \
    --output results/uninavid_test/ \
    --policy uninavid \
    --uninavid-server-url http://localhost:<port> \
    --headless
```

## MTU3D

```bash
python bench/policy/mtu3d/mtu3d_server.py \
    --mtu3d_path /path/to/MTU3D \
    --stage1_dir /path/to/stage1 \
    --stage2_dir /path/to/stage2 \
    --port <port>

python runBench.py \
    --omninavbench --mode test --robot carter --style original \
    --config configs/carter_v1_test.yaml \
    --output results/mtu3d_test/ \
    --policy mtu3d \
    --mtu3d-server-url http://localhost:<port> \
    --headless
```

## PoliFormer

```bash
python -m bench.policy.poliformer.poliformer_server \
    --poliformer-path /path/to/PoliFormer \
    --ckpt-path /path/to/model.ckpt \
    --port <port>

python runBench.py \
    --omninavbench --mode test --robot h1 --style original \
    --config configs/aliengoh1_test.yaml \
    --output results/poliformer_test/ \
    --policy poliformer \
    --poliformer-server-url http://localhost:<port> \
    --headless
```

## OmniNav

```bash
python bench/policy/omninav/omninav_server.py \
    --model_path /path/to/OmniNav/checkpoint \
    --omninav_path /path/to/OmniNav \
    --port <port>

python runBench.py \
    --omninavbench --mode test --robot carter --style original \
    --config configs/carter_v1_test.yaml \
    --output results/omninav_test/ \
    --policy omninav \
    --omninav-server-url http://localhost:<port> \
    --headless
```
