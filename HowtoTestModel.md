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

## EQA: explicit user opt-in

EQA is **off by default for every policy, including Uni-NaVid**. A model name
or an existing method does not enable it automatically. In the simulator YAML
passed to `--config`, set the top-level boolean:

```yaml
enable_eqa: true
```

Alternatively add `--enable-eqa` to the existing `runBench.py` command.
`--no-enable-eqa` explicitly disables it. Precedence is CLI > YAML > false;
subprocess/grouped runs preserve the resolved choice. Python integrations use
`BenchConfig(..., enable_eqa=True)` or `EpisodeRunner(..., enable_eqa=True)`.
These Python parameters are explicit; they do not read the YAML themselves.

Before enabling, implement this optional method in the policy adapter:

```python
def predict_text(self, question: str, rgb: np.ndarray) -> str | None:
    return self.model.answer(question, rgb)  # Use your model's own interface.
```

Enabling EQA without a callable interface fails before simulation startup.
The call receives the task question and final RGB image, not the reference
answer. Return None or an empty string only for a normal empty model response.
Let transport, inference, and malformed-response errors raise. The provided
Uni-NaVid HTTP adapter follows this contract and no longer swallows HTTP errors.
An interface check validates wiring, not a model's reasoning capability.

Each episode result JSON records `eqa_enabled` and `eqa_status`, plus
`eqa_answer` when present and a safe `eqa_reason` for disabled/error cases:

| Status | Meaning |
| --- | --- |
| `not_applicable` | No EQA task in this episode; no call. |
| `disabled` | User did not enable EQA; no claim about model capability. |
| `answered` | A nonblank answer was produced; correctness is judged offline. |
| `no_answer` | Call completed normally but produced no answer. |
| `not_run` | Required input (question/final RGB) was unavailable. |
| `error` | Call failed or produced an invalid response type. |

A disabled, unanswered, or wrong answer to an official EQA task scores zero on
the evaluation platform; disabling the switch **does not remove that task from
the denominator**. Other task scores are still evaluated. Execution faults are
reported separately. The platform must support `eqa_status=disabled`.

Resume checks do not reuse disabled/legacy outputs after enabling EQA, or reuse
an explicit EQA execution failure. Switching the setting reruns incompatible
outputs. `--no-skip` forces a rerun regardless. Back up outputs when comparing
runs. Existing EQA answers, scoring formulas, and private GT data are not changed
by the switch. In-repository offline scoring revisions may differ from the
hosted platform; this change is to runtime invocation and result reporting.
