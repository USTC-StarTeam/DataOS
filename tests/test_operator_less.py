from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from dataos.core.artifact import DatasetArtifact, make_artifact_id
from dataos.core.sandbox import Sandbox
from dataos.operators.less import LessSelectOperator


def _build_artifact(tmp_path, name: str, dataset_obj) -> DatasetArtifact:
    artifact_id = make_artifact_id("dataset")
    path = tmp_path / "src" / artifact_id
    path.mkdir(parents=True)
    artifact = DatasetArtifact(
        artifact_id=artifact_id,
        name=name,
        path=path,
        parent_ids=[],
        producer="test",
    )
    artifact.save(dataset_obj)
    return artifact


def _pool(n: int = 20) -> Dataset:
    return Dataset.from_dict(
        {
            "instruction": [f"Q-{i}" for i in range(n)],
            "response": [f"A-{i}" for i in range(n)],
        }
    )


def _target(n: int = 4) -> Dataset:
    return Dataset.from_dict(
        {
            "instruction": [f"target-{i}" for i in range(n)],
            "response": [f"target-resp-{i}" for i in range(n)],
        }
    )


def _make_pair(tmp_path):
    pool = _build_artifact(tmp_path, "pool", _pool())
    target = _build_artifact(tmp_path, "target", _target())
    return pool, target


def test_less_stub_selects_top_k_with_lineage_and_manifest(tmp_path):
    pool, target = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = LessSelectOperator().run(
        [target, pool],
        sandbox,
        target_artifact_id=target.artifact_id,
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        top_k=5,
        mode="stub",
        seed=7,
    )

    loaded = artifact.load()
    assert isinstance(loaded, Dataset)
    assert loaded.num_rows == 5
    assert artifact.parent_ids == [pool.artifact_id, target.artifact_id]
    assert artifact.meta["less_mode"] == "stub"
    assert artifact.meta["less_pool_count"] == 20
    assert artifact.meta["less_selected_count"] == 5
    assert artifact.meta["less_target_artifact_id"] == target.artifact_id
    stage_names = [stage["name"] for stage in artifact.meta["less_stages"]]
    assert stage_names == ["warmup", "train_grads", "val_grads", "select"]
    assert all(stage["cache_status"] == "skipped" for stage in artifact.meta["less_stages"])

    manifest = json.loads((artifact.path / "selection_manifest.json").read_text(encoding="utf-8"))
    assert manifest["selected_count"] == 5
    assert len(manifest["selected_indices"]) == 5
    assert manifest["pool_artifact_id"] == pool.artifact_id
    assert manifest["target_artifact_id"] == target.artifact_id

    assert (sandbox.logs_dir / "less.log").exists()
    for stage in ("warmup", "train_grads", "val_grads", "select"):
        assert (sandbox.logs_dir / f"less_{stage}.log").exists()


def test_less_stub_top_fraction(tmp_path):
    pool = _build_artifact(tmp_path, "pool", _pool(100))
    target = _build_artifact(tmp_path, "target", _target())
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = LessSelectOperator().run(
        [target, pool],
        sandbox,
        target_artifact_id=target.artifact_id,
        base_model="m",
        top_fraction=0.1,
        mode="stub",
    )

    assert artifact.load().num_rows == 10
    assert artifact.meta["less_top_fraction"] == 0.1
    assert artifact.meta["less_top_k"] is None


def test_less_preserves_other_splits_when_pool_is_dict(tmp_path):
    pool = _build_artifact(
        tmp_path,
        "pool",
        DatasetDict(
            {
                "train": _pool(20),
                "test": Dataset.from_dict({"instruction": ["t1"], "response": ["r1"]}),
            }
        ),
    )
    target = _build_artifact(tmp_path, "target", _target())
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = LessSelectOperator().run(
        [target, pool],
        sandbox,
        target_artifact_id=target.artifact_id,
        base_model="m",
        top_k=5,
        mode="stub",
    )

    loaded = artifact.load()
    assert isinstance(loaded, DatasetDict)
    assert loaded["train"].num_rows == 5
    assert loaded["test"].num_rows == 1


def test_less_stub_is_deterministic_with_seed(tmp_path):
    pool = _build_artifact(tmp_path, "pool", _pool(50))
    target = _build_artifact(tmp_path, "target", _target())

    sandbox_a = Sandbox(tmp_path / "sandbox_a")
    sandbox_a.prepare()
    artifact_a = LessSelectOperator().run(
        [target, pool],
        sandbox_a,
        target_artifact_id=target.artifact_id,
        base_model="m",
        top_k=10,
        mode="stub",
        seed=42,
    )
    sandbox_b = Sandbox(tmp_path / "sandbox_b")
    sandbox_b.prepare()
    artifact_b = LessSelectOperator().run(
        [target, pool],
        sandbox_b,
        target_artifact_id=target.artifact_id,
        base_model="m",
        top_k=10,
        mode="stub",
        seed=42,
    )

    indices_a = json.loads((artifact_a.path / "selection_manifest.json").read_text(encoding="utf-8"))[
        "selected_indices"
    ]
    indices_b = json.loads((artifact_b.path / "selection_manifest.json").read_text(encoding="utf-8"))[
        "selected_indices"
    ]
    assert indices_a == indices_b


def test_less_pool_position_in_inputs_does_not_matter(tmp_path):
    pool, target = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    # Order: pool first, target last (inverse of the typical pattern).
    artifact = LessSelectOperator().run(
        [pool, target],
        sandbox,
        target_artifact_id=target.artifact_id,
        base_model="m",
        top_k=5,
        mode="stub",
    )

    assert artifact.parent_ids == [pool.artifact_id, target.artifact_id]


def test_less_requires_target_artifact_id(tmp_path):
    pool, _ = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="target_artifact_id"):
        LessSelectOperator().run([pool], sandbox, base_model="m", top_k=5, mode="stub")


def test_less_target_artifact_id_not_in_inputs(tmp_path):
    pool, _ = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="not found in inputs"):
        LessSelectOperator().run(
            [pool],
            sandbox,
            target_artifact_id="dataset-does-not-exist",
            base_model="m",
            top_k=5,
            mode="stub",
        )


def test_less_pool_must_differ_from_target(tmp_path):
    only = _build_artifact(tmp_path, "only", _pool())
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="distinct from the target"):
        LessSelectOperator().run(
            [only],
            sandbox,
            target_artifact_id=only.artifact_id,
            base_model="m",
            top_k=5,
            mode="stub",
        )


def test_less_pool_missing_required_columns(tmp_path):
    pool = _build_artifact(tmp_path, "pool", Dataset.from_dict({"text": ["a", "b", "c"]}))
    target = _build_artifact(tmp_path, "target", _target())
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="missing required columns"):
        LessSelectOperator().run(
            [target, pool],
            sandbox,
            target_artifact_id=target.artifact_id,
            base_model="m",
            top_k=5,
            mode="stub",
        )


def test_less_target_missing_required_columns(tmp_path):
    pool = _build_artifact(tmp_path, "pool", _pool())
    target = _build_artifact(tmp_path, "target", Dataset.from_dict({"foo": ["x"]}))
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="target dataset is missing required columns"):
        LessSelectOperator().run(
            [target, pool],
            sandbox,
            target_artifact_id=target.artifact_id,
            base_model="m",
            top_k=5,
            mode="stub",
        )


def test_less_top_k_or_fraction_required(tmp_path):
    pool, target = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="top_k.*top_fraction"):
        LessSelectOperator().run(
            [target, pool],
            sandbox,
            target_artifact_id=target.artifact_id,
            base_model="m",
            mode="stub",
        )


def test_less_top_k_and_fraction_mutually_exclusive(tmp_path):
    pool, target = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="mutually exclusive"):
        LessSelectOperator().run(
            [target, pool],
            sandbox,
            target_artifact_id=target.artifact_id,
            base_model="m",
            top_k=5,
            top_fraction=0.5,
            mode="stub",
        )


def test_less_top_k_exceeds_pool_size_raises(tmp_path):
    pool, target = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="exceeds pool size"):
        LessSelectOperator().run(
            [target, pool],
            sandbox,
            target_artifact_id=target.artifact_id,
            base_model="m",
            top_k=999,
            mode="stub",
        )


def test_less_real_mode_requires_repo_path(tmp_path):
    pool, target = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="less_repo_path"):
        LessSelectOperator().run(
            [target, pool],
            sandbox,
            target_artifact_id=target.artifact_id,
            base_model="m",
            top_k=5,
            mode="less",
        )


def test_less_unknown_param_rejected(tmp_path):
    pool, target = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="unsupported params"):
        LessSelectOperator().run(
            [target, pool],
            sandbox,
            target_artifact_id=target.artifact_id,
            base_model="m",
            top_k=5,
            mode="stub",
            mystery_kwarg=True,
        )


def test_less_invalid_mode_rejected(tmp_path):
    pool, target = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="'mode' must be one of"):
        LessSelectOperator().run(
            [target, pool],
            sandbox,
            target_artifact_id=target.artifact_id,
            base_model="m",
            top_k=5,
            mode="bogus",
        )


def test_less_real_mode_missing_repo_dir_rejected(tmp_path):
    pool, target = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="less_repo_path does not exist"):
        LessSelectOperator().run(
            [target, pool],
            sandbox,
            target_artifact_id=target.artifact_id,
            base_model="m",
            top_k=5,
            mode="less",
            less_repo_path=str(tmp_path / "no-such-repo"),
        )


# ── Real-mode subprocess tests (commit 2) ───────────────────────────────────


def _arg(args: list[str], flag: str) -> str:
    for index, value in enumerate(args):
        if value == flag and index + 1 < len(args):
            return args[index + 1]
    raise AssertionError(f"flag {flag!r} not found in {args}")


def _make_fake_run(num_checkpoints: int = 2):
    """Build a subprocess.run replacement that fakes each LESS stage's outputs."""

    calls: list[list[str]] = []

    def fake_run(args, cwd, stdout, stderr, check):
        del cwd, stderr, check
        calls.append(list(args))
        cmd = " ".join(args)
        stdout.write(f"FAKE: {cmd}\n")
        stdout.flush()
        if "less.train.train" in cmd:
            output_dir = Path(_arg(args, "--output_dir"))
            output_dir.mkdir(parents=True, exist_ok=True)
            for epoch in range(1, num_checkpoints + 1):
                ckpt = output_dir / f"checkpoint-{epoch}"
                ckpt.mkdir(parents=True, exist_ok=True)
                (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
        elif "less.data_selection.get_info" in cmd:
            output_path = Path(_arg(args, "--output_path"))
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "grads.bin").write_text("fake-grads", encoding="utf-8")
        elif len(args) >= 2 and args[1] == "-c":
            # select script: args = [<python>, "-c", SCRIPT, train_root, val_root, proj_dim, keep, indices_path]
            keep = int(args[6])
            indices_path = Path(args[7])
            indices_path.parent.mkdir(parents=True, exist_ok=True)
            indices_path.write_text(json.dumps(sorted(range(keep))), encoding="utf-8")
        return subprocess.CompletedProcess(args, 0)

    return fake_run, calls


def _patch_runtime(monkeypatch, fake_run):
    monkeypatch.setattr(
        "dataos.operators._less.runner.subprocess.run", fake_run
    )
    monkeypatch.setattr(
        "dataos.operators._less.runner.shutil.which", lambda _: "/usr/bin/fake"
    )


def test_less_real_mode_runs_full_pipeline(tmp_path, monkeypatch):
    pool, target = _make_pair(tmp_path)
    repo = tmp_path / "less_repo"
    repo.mkdir()
    cache = tmp_path / "less_cache"
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    fake_run, calls = _make_fake_run(num_checkpoints=2)
    _patch_runtime(monkeypatch, fake_run)

    artifact = LessSelectOperator().run(
        [target, pool],
        sandbox,
        target_artifact_id=target.artifact_id,
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        top_k=5,
        mode="less",
        less_repo_path=str(repo),
        cache_dir=str(cache),
    )

    assert artifact.load().num_rows == 5
    assert artifact.parent_ids == [pool.artifact_id, target.artifact_id]
    assert artifact.meta["less_mode"] == "less"
    assert artifact.meta["less_cache_root"] == str(cache)

    stages = {stage["name"]: stage for stage in artifact.meta["less_stages"]}
    assert stages["warmup"]["cache_status"] == "miss"
    assert stages["train_grads"]["cache_status"] == "miss"
    assert stages["val_grads"]["cache_status"] == "miss"
    assert stages["select"]["cache_status"] == "miss"

    invocations = [" ".join(call) for call in calls]
    # 1 warmup + 2 train_grads (per ckpt) + 2 val_grads (per ckpt) + 1 select = 6
    assert len(invocations) == 6
    assert sum("less.train.train" in inv for inv in invocations) == 1
    assert sum("less.data_selection.get_info" in inv for inv in invocations) == 4
    assert sum(len(call) >= 2 and call[1] == "-c" for call in calls) == 1

    assert (cache / "warmup").exists()
    assert (cache / "train_grads").exists()
    assert (cache / "val_grads").exists()
    assert (sandbox.logs_dir / "less_warmup.log").exists()
    assert (sandbox.logs_dir / "less_select.log").exists()


def test_less_real_mode_reuses_cache_across_targets(tmp_path, monkeypatch):
    pool = _build_artifact(tmp_path, "pool", _pool())
    target_a = _build_artifact(tmp_path, "target_a", _target())
    target_b = _build_artifact(tmp_path, "target_b", _target())
    repo = tmp_path / "less_repo"
    repo.mkdir()
    cache = tmp_path / "less_cache"

    fake_run, calls = _make_fake_run(num_checkpoints=2)
    _patch_runtime(monkeypatch, fake_run)

    sandbox_a = Sandbox(tmp_path / "sandbox_a")
    sandbox_a.prepare()
    LessSelectOperator().run(
        [target_a, pool],
        sandbox_a,
        target_artifact_id=target_a.artifact_id,
        base_model="m",
        top_k=5,
        mode="less",
        less_repo_path=str(repo),
        cache_dir=str(cache),
    )
    first_call_count = len(calls)

    sandbox_b = Sandbox(tmp_path / "sandbox_b")
    sandbox_b.prepare()
    artifact_b = LessSelectOperator().run(
        [target_b, pool],
        sandbox_b,
        target_artifact_id=target_b.artifact_id,
        base_model="m",
        top_k=5,
        mode="less",
        less_repo_path=str(repo),
        cache_dir=str(cache),
    )
    second_run_calls = calls[first_call_count:]

    stages = {stage["name"]: stage for stage in artifact_b.meta["less_stages"]}
    assert stages["warmup"]["cache_status"] == "hit"
    assert stages["train_grads"]["cache_status"] == "hit"
    assert stages["val_grads"]["cache_status"] == "miss"
    assert stages["select"]["cache_status"] == "miss"

    invocations = [" ".join(call) for call in second_run_calls]
    assert sum("less.train.train" in inv for inv in invocations) == 0
    # Only target gradients (2 ckpts) + select = 3 calls in the 2nd run
    assert len(invocations) == 3
    assert sum("less.data_selection.get_info" in inv for inv in invocations) == 2
    assert all(
        "--gradient_type sgd" in inv for inv in invocations if "get_info" in inv
    )


def test_less_real_mode_force_recompute_bypasses_cache(tmp_path, monkeypatch):
    pool, target = _make_pair(tmp_path)
    repo = tmp_path / "less_repo"
    repo.mkdir()
    cache = tmp_path / "less_cache"

    fake_run, calls = _make_fake_run(num_checkpoints=2)
    _patch_runtime(monkeypatch, fake_run)

    sandbox_a = Sandbox(tmp_path / "sandbox_a")
    sandbox_a.prepare()
    LessSelectOperator().run(
        [target, pool],
        sandbox_a,
        target_artifact_id=target.artifact_id,
        base_model="m",
        top_k=5,
        mode="less",
        less_repo_path=str(repo),
        cache_dir=str(cache),
    )

    first_count = len(calls)
    sandbox_b = Sandbox(tmp_path / "sandbox_b")
    sandbox_b.prepare()
    artifact = LessSelectOperator().run(
        [target, pool],
        sandbox_b,
        target_artifact_id=target.artifact_id,
        base_model="m",
        top_k=5,
        mode="less",
        less_repo_path=str(repo),
        cache_dir=str(cache),
        force_recompute=True,
    )
    second_run_calls = len(calls) - first_count

    assert second_run_calls == first_count
    assert all(
        stage["cache_status"] == "miss"
        for stage in artifact.meta["less_stages"]
    )


def test_less_real_mode_subprocess_failure_surfaces_log(tmp_path, monkeypatch):
    pool, target = _make_pair(tmp_path)
    repo = tmp_path / "less_repo"
    repo.mkdir()
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    def fake_run(args, cwd, stdout, stderr, check):
        del cwd, stderr, check
        stdout.write("FAKE failure\n")
        stdout.flush()
        return subprocess.CompletedProcess(args, returncode=42)

    _patch_runtime(monkeypatch, fake_run)

    with pytest.raises(RuntimeError, match="exit code 42"):
        LessSelectOperator().run(
            [target, pool],
            sandbox,
            target_artifact_id=target.artifact_id,
            base_model="m",
            top_k=5,
            mode="less",
            less_repo_path=str(repo),
            cache_dir=str(tmp_path / "cache"),
        )
    assert (sandbox.logs_dir / "less_warmup.log").exists()


def test_less_real_mode_warmup_must_produce_checkpoints(tmp_path, monkeypatch):
    pool, target = _make_pair(tmp_path)
    repo = tmp_path / "less_repo"
    repo.mkdir()
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    def fake_run(args, cwd, stdout, stderr, check):
        del cwd, stderr, check
        stdout.write("FAKE: warmup that produces nothing\n")
        stdout.flush()
        # Intentionally don't create any checkpoint dirs.
        return subprocess.CompletedProcess(args, 0)

    _patch_runtime(monkeypatch, fake_run)

    with pytest.raises(RuntimeError, match="warmup produced no checkpoint"):
        LessSelectOperator().run(
            [target, pool],
            sandbox,
            target_artifact_id=target.artifact_id,
            base_model="m",
            top_k=5,
            mode="less",
            less_repo_path=str(repo),
            cache_dir=str(tmp_path / "cache"),
        )


def test_less_registered_in_default_registry():
    from dataos.operators import build_default_registry

    registry = build_default_registry()
    operator = registry.get("less")
    assert isinstance(operator, LessSelectOperator)


# ── Phase 1 ① — PYTHON executable configurable ────────────────────────────────


def _stub_config(**overrides):
    from dataos.operators._less.config import LessConfig

    base = dict(
        target_artifact_id="t",
        base_model="m",
        mode="less",
        top_k=5,
        top_fraction=None,
        seed=42,
        warmup_epochs=1,
        projection_dim=8,
        lora_r=8,
        lora_alpha=16,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        less_repo_path="/tmp/less",
        cache_dir=None,
        force_recompute=False,
        python_executable=None,
    )
    base.update(overrides)
    return LessConfig(**base)


def test_python_for_uses_explicit_kwarg(monkeypatch):
    from dataos.operators._less.commands import python_for

    monkeypatch.delenv("DATAOS_LESS_PYTHON", raising=False)
    cfg = _stub_config(python_executable="/opt/custom/bin/python")
    assert python_for(cfg) == "/opt/custom/bin/python"


def test_python_for_falls_back_to_env(monkeypatch):
    from dataos.operators._less.commands import python_for

    monkeypatch.setenv("DATAOS_LESS_PYTHON", "/env/bin/python3")
    cfg = _stub_config()
    assert python_for(cfg) == "/env/bin/python3"


def test_python_for_falls_back_to_sys_executable(monkeypatch):
    import sys

    from dataos.operators._less.commands import python_for

    monkeypatch.delenv("DATAOS_LESS_PYTHON", raising=False)
    cfg = _stub_config()
    assert python_for(cfg) == sys.executable


def test_grad_command_uses_resolved_python(monkeypatch, tmp_path):
    from dataos.operators._less.commands import grad_command

    monkeypatch.delenv("DATAOS_LESS_PYTHON", raising=False)
    cfg = _stub_config(python_executable="/opt/custom/bin/python")
    cmd = grad_command(
        cfg,
        gradient_type="adam",
        train_files=tmp_path / "p.jsonl",
        model_path=tmp_path / "ckpt",
        output_path=tmp_path / "out",
    )
    assert cmd[0] == "/opt/custom/bin/python"
    assert cmd[1:3] == ["-m", "less.data_selection.get_info"]


def test_select_command_uses_resolved_python(monkeypatch, tmp_path):
    from dataos.operators._less.commands import select_command

    monkeypatch.delenv("DATAOS_LESS_PYTHON", raising=False)
    cfg = _stub_config(python_executable="/opt/custom/bin/python")
    cmd = select_command(
        cfg,
        train_grads_root=tmp_path / "tg",
        val_grads_root=tmp_path / "vg",
        keep=5,
        indices_path=tmp_path / "out.json",
    )
    assert cmd[0] == "/opt/custom/bin/python"
    assert cmd[1] == "-c"


def test_less_config_accepts_python_executable_kwarg(tmp_path):
    pool, target = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = LessSelectOperator().run(
        [target, pool],
        sandbox,
        target_artifact_id=target.artifact_id,
        base_model="m",
        top_k=5,
        mode="stub",
        python_executable="/opt/custom/bin/python",
    )
    assert artifact.meta["less_selected_count"] == 5


# ── Phase 1 ② — cache file-lock concurrency ──────────────────────────────────


def _cache_lock_worker(cache_path_str: str, counter_path: str) -> None:
    """Module-level helper run by multiprocessing.Process.

    Acquires the cache lock, increments a shared counter only if cache_meta
    isn't there yet (mimicking the runner's double-check), then writes the
    meta. If the lock works, exactly one worker should bump the counter.
    """
    import json as _json
    import time
    from pathlib import Path as _Path

    from dataos.operators._less.cache import CacheEntry

    entry = CacheEntry(stage="warmup", key="ktest", path=_Path(cache_path_str))
    with entry.lock():
        if entry.hit():
            return
        # Simulate slow subprocess so the second worker is forced to wait.
        time.sleep(0.3)
        # Atomic-ish counter bump
        cp = _Path(counter_path)
        prev = int(cp.read_text()) if cp.exists() else 0
        cp.write_text(str(prev + 1))
        entry.write_meta({"log_path": "/dev/null"})


def test_cache_lock_serialises_concurrent_workers(tmp_path):
    import multiprocessing

    cache_path = tmp_path / "cache" / "warmup" / "ktest"
    counter_path = tmp_path / "counter.txt"

    procs = [
        multiprocessing.Process(
            target=_cache_lock_worker,
            args=(str(cache_path), str(counter_path)),
        )
        for _ in range(3)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=10)
        assert p.exitcode == 0, f"worker exited {p.exitcode}"

    assert counter_path.read_text() == "1"
    assert (cache_path / "cache_meta.json").exists()


def test_cache_lock_can_be_reentered_after_release(tmp_path):
    from dataos.operators._less.cache import CacheEntry

    entry = CacheEntry(stage="warmup", key="k", path=tmp_path / "cache" / "warmup" / "k")
    with entry.lock():
        pass
    # second acquisition in the same process should also succeed (sequential).
    with entry.lock():
        entry.write_meta({"log_path": "/dev/null"})
    assert entry.hit()
