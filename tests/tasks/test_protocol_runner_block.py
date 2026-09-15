"""protocol.yaml's runner block decides how each bench runs — and did not, until 2026-08-13.

The block declared `runner`, `num_concurrent`, `timeout` and `gpu_memory_utilization` per bench
while `run_lmeval_eval.py` decided all four from its own literals. Nothing read the config. A file
whose header states that "protocol decisions may read only this config" was, for the runner half,
decorative — and it declared values for a bench (bfcl at concurrency 32) that had been
catastrophic under an earlier defect, so anyone wiring it up later would have been wiring in a
number they had no reason to re-check.

This module pins the wiring in both directions:

* the evaluator's resolved values equal what protocol.yaml declares (the config really decides), and
* those values equal the literals the script used BEFORE the wiring — the snapshot below. That
  second assertion is what makes the change provably behaviour-preserving; it is not a claim that
  these values are eternally correct. Changing one is allowed, deliberately, by editing both
  protocol.yaml and this snapshot in the same commit — which is exactly the review moment the old
  arrangement had no way to force.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

_EVAL_DIR = Path(__file__).resolve().parents[2] / "configs/tasks/posttrainbench/eval"
_SCRIPT = _EVAL_DIR / "scripts/run_lmeval_eval.py"

#: The literals `run_lmeval_eval.py` carried until 2026-08-13, verbatim:
#:   DIRECT_VLLM_TASKS = {"gpqamain", "humaneval"}
#:   SERVER_TASKS = {"gsm8k", "bfcl", "aime2025", "healthbench", "arenahardwriting"}
#:   JUDGE_TASKS = {"healthbench", "arenahardwriting"}
#:   DIRECT_GPU_MEMORY = {"humaneval": 0.3, "gpqamain": 0.8}
#:   DEFAULT_NUM_CONCURRENT = 32
#:   default_timeout(): aime2025 18_000_000 | arena/bfcl/gpqamain/health 1800 | humaneval 900 | 300
_FROZEN = {
    "gsm8k": {"runner": "server", "judge": False, "timeout": 300, "num_concurrent": 32},
    "bfcl": {"runner": "server", "judge": False, "timeout": 1800, "num_concurrent": 32},
    "aime2025": {"runner": "server", "judge": False, "timeout": 18_000_000, "num_concurrent": 32},
    "healthbench": {"runner": "server", "judge": True, "timeout": 1800, "num_concurrent": 32},
    "arenahardwriting": {
        "runner": "server", "judge": True, "timeout": 1800, "num_concurrent": 32,
    },
    "humaneval": {
        "runner": "direct", "judge": False, "timeout": 900, "gpu_memory_utilization": 0.3,
    },
    "gpqamain": {
        "runner": "direct", "judge": False, "timeout": 1800, "gpu_memory_utilization": 0.8,
    },
}


@pytest.fixture(scope="module")
def evaluator():
    """Import the frozen evaluator script by path (it is task config, not an installed package)."""
    spec = importlib.util.spec_from_file_location("ptb_run_lmeval_eval", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - lm_eval/torch may be absent on the agent host
        pytest.skip(f"evaluator needs deps unavailable here: {type(exc).__name__}: {exc}")
    return module


@pytest.fixture(scope="module")
def declared() -> dict:
    return yaml.safe_load((_EVAL_DIR / "protocol.yaml").read_text(encoding="utf-8"))["benches"]


def test_the_runner_block_covers_exactly_the_benches_the_script_knows(declared, evaluator):
    assert set(declared) == set(_FROZEN), (
        "protocol.yaml's benches and this test's frozen snapshot disagree — if a bench was added "
        "or removed, update the snapshot in the same commit"
    )
    assert set(evaluator.TASK_ALIASES) >= set(declared)


@pytest.mark.parametrize("bench", sorted(_FROZEN))
def test_the_evaluator_resolves_every_value_from_the_config(bench, declared, evaluator):
    """The config decides. Reading it from the script's own directory (not from argv) is what makes
    the value un-spoofable: a caller that could pass `--num-concurrent` for bfcl could move the
    protocol, which is the same class of hole as guessing the chat template from `--model-name`."""
    spec = evaluator.bench_protocol(bench)
    assert spec == declared[bench]
    assert evaluator.is_direct(bench) is (declared[bench]["runner"] == "direct")
    assert evaluator.needs_judge(bench) is bool(declared[bench]["judge"])
    assert evaluator.default_timeout(bench) == declared[bench]["timeout"]
    if declared[bench]["runner"] == "direct":
        assert evaluator.direct_gpu_memory(bench) == declared[bench]["gpu_memory_utilization"]
    else:
        assert evaluator.default_num_concurrent(bench) == declared[bench]["num_concurrent"]


@pytest.mark.parametrize("bench", sorted(_FROZEN))
def test_wiring_the_config_changed_no_value(bench, declared):
    """Behaviour preservation, key by key: 21 keys, all equal at wiring time."""
    assert declared[bench] == _FROZEN[bench], (
        f"{bench}: protocol.yaml now differs from the pre-wiring literals. That is a scoring "
        "change, not a refactor — re-baseline the bench or revert."
    )


def test_an_undeclared_bench_is_refused_not_guessed(evaluator):
    """The old literals silently answered for any bench; a protocol gap must stop the run."""
    with pytest.raises(SystemExit) as err:
        evaluator.bench_protocol("not_a_bench")
    assert "not declared" in str(err.value)


@pytest.mark.parametrize("family, parser", [("qwen3_5", "qwen3_coder"), ("qwen3", "hermes")])
def test_tool_parser_uses_checkpoint_architecture(evaluator, tmp_path, family, parser):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": family}))
    assert evaluator.tool_call_parser_name(str(tmp_path), "misleading-llama-name") == parser


def test_qwen38_server_uses_all_allocated_gpus(evaluator, tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"model_type": "qwen3_5"}')
    commands = []
    monkeypatch.setattr(evaluator, "find_free_port", lambda: 12345)
    monkeypatch.setattr(evaluator, "wait_for_health", lambda *_, **__: None)
    monkeypatch.setattr(
        evaluator.subprocess, "Popen",
        lambda command, **_: commands.append(command) or SimpleNamespace(),
    )
    evaluator.start_vllm_server(
        "bfcl", str(tmp_path), tmp_path / "template.jinja", "checkpoint", tmp_path,
        tensor_parallel_size=8,
    )
    command = commands[0]
    for flag, value in [
        ("--tensor-parallel-size", "8"), ("--tool-call-parser", "qwen3_coder"),
        ("--reasoning-parser", "qwen3"), ("--max-model-len", "32768"),
    ]:
        assert command[command.index(flag) + 1] == value


def test_legacy_single_gpu_server_arguments_are_unchanged(evaluator, tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"model_type": "qwen3"}')
    commands = []
    monkeypatch.setattr(evaluator, "find_free_port", lambda: 12345)
    monkeypatch.setattr(evaluator, "wait_for_health", lambda *_, **__: None)
    monkeypatch.setattr(
        evaluator.subprocess, "Popen",
        lambda command, **_: commands.append(command) or SimpleNamespace(),
    )
    evaluator.start_vllm_server("bfcl", str(tmp_path), tmp_path / "template", "qwen", tmp_path)
    assert "--tensor-parallel-size" not in commands[0]
    assert "--reasoning-parser" not in commands[0]
    assert "qwen3_coder" not in commands[0]


def test_qwen38_retry_cleans_workers_and_uses_declared_startup_settings(
    evaluator, tmp_path, monkeypatch,
):
    (tmp_path / "config.json").write_text('{"model_type": "qwen3_5"}')
    events = []

    def popen(command, **kwargs):
        events.append("start")
        assert kwargs["start_new_session"] is True
        assert kwargs["env"]["PATH"].startswith(f"{Path(evaluator.sys.executable).parent}:")
        assert kwargs["env"]["VLLM_CACHE_ROOT"] == f"{tmp_path}/cache/vllm"
        assert "--enforce-eager" in command
        assert command[command.index("--gdn-prefill-backend") + 1] == "triton"
        return SimpleNamespace()

    def health(port, process, timeout):
        assert timeout == 1800
        if events.count("start") == 1:
            raise TimeoutError("cold startup")

    monkeypatch.setattr(evaluator.subprocess, "Popen", popen)
    monkeypatch.setattr(evaluator, "find_free_port", lambda: 12345)
    monkeypatch.setattr(evaluator, "wait_for_health", health)
    monkeypatch.setattr(evaluator, "stop_vllm_server", lambda p: events.append("cleanup"))
    monkeypatch.setattr(evaluator.time, "sleep", lambda _: None)
    evaluator.start_vllm_server("bfcl", str(tmp_path), tmp_path / "t", "qwen", tmp_path)
    assert events == ["start", "cleanup", "start"]


def test_cleanup_signals_private_group_even_after_parent_exit(evaluator, monkeypatch):
    signals = []
    monkeypatch.setattr(evaluator.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    process = SimpleNamespace(pid=123456, wait=lambda **_: 0, poll=lambda: 1)
    evaluator.stop_vllm_server(process)
    assert signals == [
        (123456, evaluator.signal.SIGTERM), (123456, evaluator.signal.SIGKILL),
    ]
