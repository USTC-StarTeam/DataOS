"""The Spec/Stage base: what the three stages now share, and what they may not state.

The three used to be peers in name only — ``TrainSpec`` was a closed schema, while evaluation
and data building turned an ``argparse.Namespace`` straight into a Job. These tests are about the
shape that is now common to all three, not about any one stage's content.
"""

from __future__ import annotations

import pytest
from dataos_launch.spec import Spec, SpecError
from dataos_launch.stage import Stage
from dataos_launch.types import Job, Resources, Status
from pydantic import ValidationError

from dataos_data.stage import DataSpec, DataStage
from dataos_eval.stage import EvalSpec
from dataos_train.spec import TrainSpec

LOCAL_ENV = {"name": "T", "kind": "local", "launcher": "local",
             "tools": {"sft_env": {"path": "/envs/sft"}}}


class _Fake(Stage):
    name = "fake"

    def __init__(self, env, errors=()):
        super().__init__(env)
        self.errors = list(errors)

    def validate(self, spec):
        return list(self.errors)

    def build(self, spec) -> Job:
        return Job(backend="fake", job_name=spec.name, command=["true"], run_dir=spec.run_dir,
                   resources=spec.resources)


class _FakeSpec(Spec):
    stage = "fake"


def test_all_three_specs_share_one_base() -> None:
    for cls in (TrainSpec, EvalSpec, DataSpec):
        assert issubclass(cls, Spec)
    assert {c.stage for c in (TrainSpec, EvalSpec, DataSpec)} == {"train", "eval", "data"}


def test_no_spec_may_state_placement() -> None:
    """The invariant the whole refactor rests on: a Spec says what work to do, never where.

    ``TrainSpec`` carried a ``cluster`` block until 2026-08-14, and because it rode to the
    launcher on the Job it silently outranked the deployment's own configuration.
    """
    common = dict(name="n", run_dir="/r")
    with pytest.raises(ValidationError):
        TrainSpec.model_validate({**common, "base_model": "/m", "cluster": {"qos": "x"}})
    with pytest.raises(ValidationError):
        DataSpec.model_validate({**common, "recipe": "r.py", "qos": "gpu_priv"})
    for cls in (TrainSpec, EvalSpec, DataSpec):
        assert not {"cluster", "qos", "partition", "nodelist"} & set(cls.model_fields)


def test_each_spec_gets_its_own_file(tmp_path) -> None:
    """One run_dir, many jobs. `<stage>_spec.json` gave a checkpoint sweep's six evaluations
    one path between them, so the file recording what was intended held only the last."""
    first = DataSpec(name="d", run_dir=str(tmp_path), recipe="build.py",
                     resources=Resources(gpus=0, cpus=4))
    second = first.model_copy(update={"name": "d2", "recipe": "other.py"})
    p1, p2 = first.write(), second.write()
    assert p1 != p2 and p1.name == "data-d.json"
    assert DataSpec.read(p1) == first and DataSpec.read(p2) == second   # neither overwrote


def test_prepare_reports_every_reason_at_once(tmp_path) -> None:
    with pytest.raises(SpecError) as exc:
        _Fake(LOCAL_ENV, ["first", "second"]).prepare(_FakeSpec(name="f", run_dir=str(tmp_path)))
    assert exc.value.errors == ["first", "second"]


def test_submit_validates_first(tmp_path) -> None:
    """dataos-train's submit used to skip validation, so a spec its backend rejected at build
    time went out anyway when the caller passed a TrainSpec straight to submit."""
    with pytest.raises(SpecError):
        _Fake(LOCAL_ENV, ["nope"]).submit(_FakeSpec(name="f", run_dir=str(tmp_path)))
    assert not (tmp_path / "handle.json").exists()


def test_submit_runs_the_job_through_the_environments_launcher(tmp_path) -> None:
    stage = _Fake(LOCAL_ENV)
    handle = stage.submit(_FakeSpec(name="f", run_dir=str(tmp_path),
                                    resources=Resources(gpus=0, cpus=1, time_h=0.5)))
    assert handle.launcher == "local"  # LOCAL_ENV declares it; nothing named it on a flag
    assert stage.status(handle) in set(Status)


def test_preview_answers_what_would_be_submitted(tmp_path) -> None:
    out = _Fake(LOCAL_ENV).preview(_FakeSpec(name="f", run_dir=str(tmp_path)),
                                   {"qos": "gpu_priv"})
    assert out["stage"] == "fake" and out["launcher"] == "local"
    # placement is not on the Job any more, so a dry run has to print it separately or lie
    assert out["placement"]["qos"] == "gpu_priv"
    assert out["spec"]["name"] == "f" and out["job"]["command"] == ["true"]


def test_load_env_reads_the_deployment_once(tmp_path) -> None:
    """Three copies of this used to open env.yaml by hand, one per instrument."""
    d = tmp_path / "configs" / "environments" / "E"
    d.mkdir(parents=True)
    (d / "env.yaml").write_text("name: E\nkind: local\nlauncher: local\n", encoding="utf-8")
    assert DataStage.from_repo(tmp_path, "E").env["launcher"] == "local"
