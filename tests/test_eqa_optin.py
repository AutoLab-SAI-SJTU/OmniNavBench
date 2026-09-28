"""CPU-only tests of explicit EQA opt-in and runtime/result wiring."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from bench.evaluator.eqa_runtime import (
    EQASetupError, answer_text, execute_eqa, reusable_eqa_result,
    resolve_eqa_enabled, validate_eqa_interface,
)
from bench.evaluator.episode_runner import EpisodeConfig, EpisodeResult, EpisodeRunner
from bench.evaluator.bench_runner import BenchConfig, BenchRunner, ScenarioRef
from bench.policy.base import Action, BasePolicy, Observation

ROOT = Path(__file__).resolve().parents[1]
RGB = np.zeros((2, 2, 3), dtype=np.uint8)
QA = {"qa": {"question": "How many chairs?", "answer": "3"}}

class AnyNamedPolicy(BasePolicy):
    def __init__(self, value="three", error=None):
        super().__init__()
        self.value, self.error, self.calls = value, error, []
    def act(self, observation):
        return Action(stop=True)
    def predict_text(self, question, rgb):
        self.calls.append((question, rgb))
        if self.error is not None:
            raise self.error
        return self.value

class NavigationOnly(BasePolicy):
    def act(self, observation):
        return Action(stop=True)

@pytest.mark.parametrize("body,override,expected", [
    ("{}", None, False), ("enable_eqa: false", None, False),
    ("enable_eqa: true", None, True), ("enable_eqa: false", True, True),
    ("enable_eqa: true", False, False), ("", None, False),
])
def test_yaml_precedence(tmp_path, body, override, expected):
    path = tmp_path / "config.yaml"
    path.write_text(body)
    assert resolve_eqa_enabled(path, override) is expected

@pytest.mark.parametrize("value", ['"false"', '"true"', '1', 'null', '[]'])
def test_bad_boolean_is_rejected(tmp_path, value):
    path = tmp_path / "config.yaml"
    path.write_text("enable_eqa: " + value)
    with pytest.raises(EQASetupError):
        resolve_eqa_enabled(path)

@pytest.mark.parametrize("enabled", [True, False])
def test_no_task_is_not_applicable(enabled):
    policy = AnyNamedPolicy()
    assert execute_eqa(policy, enabled, {}, RGB).status == "not_applicable"
    assert not policy.calls

def test_disabled_does_not_guess_capability():
    policy = AnyNamedPolicy()
    out = execute_eqa(policy, False, QA, RGB)
    assert (out.status, out.reason, out.answer) == ("disabled", "disabled_by_user", None)
    assert not policy.calls

def test_uninavid_name_does_not_auto_enable():
    policy = type("UniNaVidCustom", (AnyNamedPolicy,), {})()
    assert execute_eqa(policy, False, QA, RGB).status == "disabled"
    assert not policy.calls

def test_arbitrary_name_can_answer():
    policy = AnyNamedPolicy()
    out = execute_eqa(policy, True, QA, RGB)
    assert out.status == "answered" and out.answer == "three"
    assert len(policy.calls) == 1 and policy.calls[0][0] == QA["qa"]["question"]
    assert policy.calls[0][1] is RGB

@pytest.mark.parametrize("value", [None, "", "   ", "\n"])
def test_normal_no_answer(value):
    assert execute_eqa(AnyNamedPolicy(value), True, QA, RGB).status == "no_answer"

@pytest.mark.parametrize("value,expected", [(0, "0"), (0.0, "0.0"), ("0", "0"), ("three", "three")])
def test_answer_values(value, expected):
    assert answer_text(value) == expected

@pytest.mark.parametrize("value", [True, [], {}, float("nan"), float("inf")])
def test_invalid_response_is_error(value):
    assert execute_eqa(AnyNamedPolicy(value), True, QA, RGB).status == "error"

@pytest.mark.parametrize("error", [RuntimeError("private-token"), requests.Timeout("private-url"), NotImplementedError()])
def test_execution_error_not_a_model_nonanswer(error):
    out = execute_eqa(AnyNamedPolicy(error=error), True, QA, RGB)
    assert out.status == "error" and out.answer is None
    assert "private" not in str(out)

@pytest.mark.parametrize("extra,rgb,reason", [
    ({"subtasks": [{"type": "EQA"}]}, RGB, "missing_question"),
    ({"qa": {"question": ""}}, RGB, "missing_question"),
    (QA, None, "missing_final_rgb"), (QA, np.array([]), "missing_final_rgb"),
])
def test_not_run_has_reason(extra, rgb, reason):
    policy = AnyNamedPolicy()
    out = execute_eqa(policy, True, extra, rgb)
    assert (out.status, out.reason) == ("not_run", reason)
    assert not policy.calls

def test_fail_fast_for_missing_interface(tmp_path):
    with pytest.raises(EQASetupError, match="predict_text"):
        EpisodeRunner(NavigationOnly(), [], enable_eqa=True)
    config = BenchConfig(tmp_path / "config.yaml", tmp_path / "env", tmp_path, enable_eqa=True)
    with pytest.raises(EQASetupError, match="predict_text"):
        BenchRunner(config, NavigationOnly())
    validate_eqa_interface(NavigationOnly(), False)

def test_noncallable_and_wrong_flag_fail():
    with pytest.raises(EQASetupError):
        validate_eqa_interface(SimpleNamespace(predict_text=1), True)
    with pytest.raises(EQASetupError):
        validate_eqa_interface(AnyNamedPolicy(), "false")

def test_default_is_off():
    config = BenchConfig(Path("c"), Path("e"), Path("o"))
    assert config.enable_eqa is False
    assert EpisodeRunner(AnyNamedPolicy(), []).enable_eqa is False

def test_episode_state_does_not_leak():
    runner = EpisodeRunner(AnyNamedPolicy(), [], enable_eqa=True)
    runner._config = EpisodeConfig("s1", "nav", (0, 0, 0), extra=QA)
    runner._call_eqa_if_available(Observation(rgb=RGB), "nav")
    assert runner._eqa_result == "three" and runner._eqa_accuracy is True
    runner._config = EpisodeConfig("s2", "nav", (0, 0, 0))
    runner._call_eqa_if_available(Observation(rgb=RGB), "nav")
    assert runner._eqa_result is None and runner._eqa_accuracy is None
    assert runner._eqa_status == "not_applicable"

@pytest.mark.parametrize("enabled", [True, False])
def test_subprocess_preserves_choice(tmp_path, enabled):
    config = BenchConfig(tmp_path / "c.yaml", tmp_path / "env", tmp_path, enable_eqa=enabled)
    cmd = BenchRunner(config, AnyNamedPolicy())._build_subprocess_cmd([tmp_path / "one.json"], ["s"])
    assert ("--enable-eqa" if enabled else "--no-enable-eqa") in cmd

@pytest.mark.parametrize("flag,expected", [(None, None), ("--enable-eqa", True), ("--no-enable-eqa", False)])
def test_cli_flag(monkeypatch, flag, expected):
    import runBench
    argv = ["runBench.py", "--config", "c.yaml", "--envset", "env.json"]
    if flag:
        argv.append(flag)
    monkeypatch.setattr(sys, "argv", argv)
    assert runBench.parse_args().enable_eqa is expected

@pytest.mark.parametrize("enabled,status,old,expected", [
    (True, "answered", True, True), (True, "no_answer", True, True),
    (True, "not_applicable", True, True), (True, "disabled", False, False),
    (False, "answered", True, False), (False, "disabled", False, True),
    (True, "error", True, False), (True, "not_run", True, False),
])
def test_resume(tmp_path, enabled, status, old, expected):
    path = tmp_path / "e.json"
    path.write_text(json.dumps({"eqa_enabled": old, "eqa_status": status}))
    assert reusable_eqa_result(path, enabled) is expected

def test_legacy_resume(tmp_path):
    path = tmp_path / "e.json"
    path.write_text('{"scenario_id":"s"}')
    assert reusable_eqa_result(path, False)
    assert not reusable_eqa_result(path, True)

def test_actual_save_preserves_status_and_zero(tmp_path):
    config = BenchConfig(tmp_path / "c.yaml", tmp_path / "env", tmp_path)
    runner = BenchRunner(config, AnyNamedPolicy())
    result = EpisodeResult("s", False, "stop", 1, 0.1, 0, 0,
                           extra={"eqa_status": "disabled", "eqa_enabled": False,
                                  "eqa_reason": "disabled_by_user", "eqa_answer": "0"})
    runner._save_episode_result(result)
    data = json.loads((tmp_path / "s.json").read_text())
    assert data["eqa_enabled"] is False and data["eqa_status"] == "disabled"
    assert data["eqa_reason"] == "disabled_by_user" and data["eqa_answer"] == "0"
    assert "metrics" not in data

def test_constructor_and_result_wiring():
    tree = ast.parse((ROOT / 'bench/evaluator/bench_runner.py').read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'EpisodeRunner']
    assert len(calls) == 2
    assert all(any(k.arg == 'enable_eqa' for k in call.keywords) for call in calls)
    source = (ROOT / 'bench/evaluator/episode_runner.py').read_text()
    assert "startswith('uninavid')" not in source
    assert 'extra["eqa_status"] = self._eqa_status' in source
    assert 'extra["eqa_enabled"] = self.enable_eqa' in source

@pytest.mark.parametrize("result,expected", [({"answer": None}, None), ({"answer": ""}, None), ({"answer": 0}, "0")])
def test_http_response(result, expected):
    from bench.policy.uninavid.uninavid_http_policy import UniNaVidHTTPPolicy
    policy = UniNaVidHTTPPolicy.__new__(UniNaVidHTTPPolicy)
    policy.server_url, policy.timeout = "http://unused", 1
    policy._prepare_image = lambda rgb: rgb
    policy._encode_image = lambda rgb: "image"
    policy.session = SimpleNamespace(post=lambda *a, **k: SimpleNamespace(raise_for_status=lambda: None, json=lambda: result))
    assert policy.predict_text("q", RGB) == expected

@pytest.mark.parametrize("result", [{}, [], {"answer": {"unexpected": True}}])
def test_bad_http_response_not_silenced(result):
    from bench.policy.uninavid.uninavid_http_policy import UniNaVidHTTPPolicy
    policy = UniNaVidHTTPPolicy.__new__(UniNaVidHTTPPolicy)
    policy.server_url, policy.timeout = "http://unused", 1
    policy._prepare_image = lambda rgb: rgb
    policy._encode_image = lambda rgb: "image"
    policy.session = SimpleNamespace(post=lambda *a, **k: SimpleNamespace(raise_for_status=lambda: None, json=lambda: result))
    with pytest.raises(ValueError):
        policy.predict_text("q", RGB)

@pytest.mark.parametrize("enabled,extra,value,error,status", [
    (False, QA, "three", None, "disabled"),
    (True, QA, "three", None, "answered"),
    (True, QA, None, None, "no_answer"),
    (True, QA, None, RuntimeError("fault"), "error"),
    (True, {}, "three", None, "not_applicable"),
])
def test_episode_loop_to_saved_json(monkeypatch, tmp_path, enabled, extra, value, error, status):
    import bench.evaluator.episode_runner as module
    from bench.configs.execution import ExecutionMode
    import types
    agent_module = types.ModuleType("OmniNavExt.envset.agent_manager")
    agent_module.AgentManager = SimpleNamespace(has_instance=lambda: False)
    monkeypatch.setitem(sys.modules, "OmniNavExt.envset.agent_manager", agent_module)
    measure = SimpleNamespace(reset_measures=lambda *a: None, update_measures=lambda *a: None,
                              get_measurements=lambda: {})
    monkeypatch.setattr(module, "add_measurement", lambda _: measure)
    executor = SimpleNamespace(finished=True, active_action=None)
    def start(action, *_):
        executor.active_action = action
    executor.start = start
    executor.execute = lambda *_: None
    monkeypatch.setattr(module, "create_executor", lambda **kw: executor)
    policy = AnyNamedPolicy(value, error)
    runner = EpisodeRunner(policy, [], enable_eqa=enabled)
    monkeypatch.setattr(runner, "_resolve_mode", lambda: ExecutionMode.STEP_ACTION)
    monkeypatch.setattr(runner, "_build_measure_setup", lambda _: None)
    monkeypatch.setattr(runner, "_build_observation", lambda **kw: Observation(rgb=RGB, step=kw["step"], time_s=kw["step"] * 0.1))
    sim = SimpleNamespace(config=SimpleNamespace(simulator=SimpleNamespace(physics_dt=0.1)))
    result = runner.run(sim, EpisodeConfig("mock-episode", "nav", (0, 0, 0), extra=extra))
    assert result.steps == 1 and result.extra["eqa_status"] == status
    assert result.extra["eqa_enabled"] is enabled
    assert len(policy.calls) == (1 if enabled and extra else 0)
    saver = BenchRunner(BenchConfig(Path("c"), Path("e"), tmp_path, enable_eqa=enabled), policy)
    saver._save_episode_result(result)
    saved = json.loads((tmp_path / "mock-episode.json").read_text())
    assert saved["eqa_status"] == status and saved["eqa_enabled"] is enabled
    assert ("eqa_answer" in saved) == (status == "answered")
    assert "answer" not in saved and "qa" not in saved
