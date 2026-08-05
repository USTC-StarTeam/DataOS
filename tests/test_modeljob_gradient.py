"""Gradient (LESS-style) interaction: spec, pure projection math, build, collect.

All torch-free — the model-side glue in ``score_entry._score_gradient`` lazily imports
torch/peft at job time; here we only exercise the numpy pure functions and the
validate/build/collect orchestration.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import dataos_train.modeljob as mj
from dataos_train.modeljob import gradient
from dataos_train.types import Status


def _spec_dict(**kw) -> dict:
    base = {
        "name": "less",
        "model": "/models/qwen",
        "dataset": {"path": "/data/train.jsonl", "format": "alpaca"},
        "interaction": {"kind": "gradient"},
    }
    return base | kw


def _spec(**kw) -> mj.ModelJobSpec:
    return mj.ModelJobSpec.model_validate(_spec_dict(**kw))


# --- spec ---------------------------------------------------------------------


def test_gradient_kind_accepted_with_defaults() -> None:
    spec = _spec()
    inter = spec.interaction
    assert isinstance(inter, mj.GradientInteraction)
    assert inter.kind == "gradient"
    assert inter.projection_dim == 1024
    assert inter.projection_seed == 42
    assert inter.lora_r == 8
    assert inter.max_len == 512
    assert inter.target is None
    assert inter.output_key == "metadata.less.score"
    assert mj.validate(spec) == []


def test_union_discriminates_both_kinds() -> None:
    loss = mj.ModelJobSpec.model_validate(_spec_dict(interaction={"kind": "loss"}))
    assert isinstance(loss.interaction, mj.LossInteraction)
    assert isinstance(_spec().interaction, mj.GradientInteraction)
    with pytest.raises(ValidationError):
        _spec(interaction={"kind": "perplexity"})
    # no kind at all -> discriminator cannot resolve
    with pytest.raises(ValidationError):
        _spec(interaction={"batch_size": 2})


def test_gradient_rejects_bad_projection_dim_and_lora_r() -> None:
    with pytest.raises(ValidationError):
        _spec(interaction={"kind": "gradient", "projection_dim": 0})
    with pytest.raises(ValidationError):
        _spec(interaction={"kind": "gradient", "lora_r": -1})
    # belt-and-braces: a spec bypassing schema validation is still rejected by validate()
    bypassed = mj.GradientInteraction.model_construct(
        kind="gradient", projection_dim=0, lora_r=0, max_len=0, output_key=""
    )
    errors = mj.validate(_spec(interaction=bypassed))
    assert any("projection_dim" in e for e in errors)
    assert any("lora_r" in e for e in errors)
    assert any("max_len" in e for e in errors)
    assert any("output_key" in e for e in errors)


def test_gradient_validate_target(tmp_path: Path) -> None:
    # schema: target must be a DatasetRef
    with pytest.raises(ValidationError):
        _spec(interaction={"kind": "gradient", "target": "/not/a/ref"})
    # validate: a given target must be a readable dataset
    errors = mj.validate(_spec(interaction={"kind": "gradient", "target": {"path": "/no/such"}}))
    assert any("target" in e and "readable" in e for e in errors)
    bench = tmp_path / "bench.jsonl"
    bench.write_text('{"instruction": "q", "output": "a"}\n', encoding="utf-8")
    ok = _spec(interaction={"kind": "gradient", "target": {"path": str(bench)}})
    assert mj.validate(ok) == []


# --- pure functions -------------------------------------------------------------


def test_rademacher_deterministic() -> None:
    a = gradient.rademacher_matrix(64, 32, seed=7)
    b = gradient.rademacher_matrix(64, 32, seed=7)
    c = gradient.rademacher_matrix(64, 32, seed=8)
    assert np.array_equal(a, b)  # same seed -> same matrix
    assert not np.array_equal(a, c)  # different seed -> different matrix
    assert a.shape == (64, 32)
    assert set(np.unique(a)).issubset({-1.0 / math.sqrt(32), 1.0 / math.sqrt(32)})


def test_project_dimension_determinism_and_scale() -> None:
    v = np.random.default_rng(0).normal(size=257)
    p = gradient.project(v, 64, seed=42)
    assert p.shape == (64,)
    # deterministic per seed
    assert np.array_equal(p, gradient.project(v, 64, seed=42))
    assert not np.allclose(p, gradient.project(v, 64, seed=43))
    # JL: norm is roughly preserved up to the 1/sqrt(dim) scaling
    assert np.linalg.norm(p) == pytest.approx(np.linalg.norm(v) / 8, rel=0.5)
    with pytest.raises(ValueError, match="dim"):
        gradient.project(v, 0, seed=42)


def test_sign_normalize() -> None:
    out = gradient.sign_normalize(np.array([2.0, -0.5, 0.0]))
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(-1.0)
    assert out[2] == 0.0  # eps keeps it finite without flipping the sign
    big = gradient.sign_normalize(np.random.default_rng(1).normal(size=1000) * 1e6)
    assert np.all(np.abs(big) <= 1.0)


def test_cosine_orthogonal_parallel() -> None:
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])
    assert gradient.cosine(x, y) == pytest.approx(0.0)  # orthogonal
    assert gradient.cosine(x, 3 * x) == pytest.approx(1.0)  # same direction
    assert gradient.cosine(x, -x) == pytest.approx(-1.0)  # opposite
    assert gradient.cosine(x, np.zeros(2)) == 0.0  # zero vector is safe


def test_flatten_and_l2_and_mean_direction() -> None:
    flat = gradient.flatten_grads([np.ones((2, 2)), np.zeros(3)])
    assert flat.shape == (7,) and flat.sum() == pytest.approx(4.0)
    assert gradient.l2_norm(np.array([3.0, 4.0])) == pytest.approx(5.0)
    direction = gradient.mean_direction([np.array([1.0, 0.0]), np.array([2.0, 0.0])])
    assert gradient.cosine(direction, np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert gradient.mean_direction([]).shape == (0,)


def test_shared_seed_makes_cosines_comparable() -> None:
    """The historical-bug guard: one seed -> one projection matrix -> comparable cosines."""
    rng = np.random.default_rng(0)
    g = rng.normal(size=512)
    other = rng.normal(size=512)
    a1 = gradient.project(g, 128, seed=42)
    a2 = gradient.project(g, 128, seed=42)  # second record, SAME seed
    assert np.array_equal(a1, a2)
    assert gradient.cosine(a1, a2) == pytest.approx(1.0)  # same vector -> similarity 1
    b = gradient.project(other, 128, seed=42)
    assert gradient.cosine(a1, b) < 1.0  # different vector -> strictly below 1
    # a different seed silently changes the matrix — the constraint the spec pins down
    assert not np.allclose(a1, gradient.project(g, 128, seed=43))


# --- build ----------------------------------------------------------------------


def test_build_renders_gradient_config() -> None:
    spec = _spec(
        interaction={
            "kind": "gradient",
            "projection_seed": 7,
            "projection_dim": 256,
            "target": {"path": "/data/bench_train.jsonl", "format": "alpaca"},
        }
    )
    job = mj.build(spec, run_dir="/runs/less")
    script = job.assets["run_score.sh"]
    assert "cd /runs/less" in script
    assert "python -m dataos_train.modeljob.score_entry --config modeljob.json" in script
    cfg = json.loads(job.assets["modeljob.json"])
    inter = cfg["interaction"]
    assert inter["kind"] == "gradient"
    assert inter["projection_seed"] == 7
    assert inter["projection_dim"] == 256
    assert inter["target"]["path"] == "/data/bench_train.jsonl"
    assert inter["output_key"] == "metadata.less.score"


# --- collect --------------------------------------------------------------------


def _write_run(run_dir: Path, scores: list[dict], n_records: int = 3) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    data_path = run_dir / "data.jsonl"
    records = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(n_records)]
    data_path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    (run_dir / "modeljob.json").write_text(
        json.dumps(_spec_dict(dataset={"path": str(data_path), "format": "alpaca"})),
        encoding="utf-8",
    )
    (run_dir / "scores.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in scores), encoding="utf-8"
    )


def _score_lines(values: list) -> list[dict]:
    return [{"index": i, "key": "metadata.less.score", "score": v} for i, v in enumerate(values)]


def test_collect_parses_score_field_and_stats(tmp_path: Path) -> None:
    _write_run(tmp_path, _score_lines([0.1, 0.2, 0.3]))
    result = mj.collect(str(tmp_path))
    assert result.status == Status.COMPLETED
    assert result.rows == 3 and result.key == "metadata.less.score"
    m = result.metrics
    assert m["n_scored"] == 3 and m["n_null"] == 0
    assert m["score_mean"] == pytest.approx(0.2)
    assert m["score_min"] == pytest.approx(0.1)
    assert m["score_max"] == pytest.approx(0.3)
    assert m["score_p50"] == pytest.approx(0.2)


def test_collect_rejects_non_finite_score(tmp_path: Path) -> None:
    _write_run(tmp_path, _score_lines([0.1, float("nan"), 0.3]))
    result = mj.collect(str(tmp_path))
    assert result.status == Status.FAILED
    assert "non-finite score" in (result.error or "")


def test_collect_tolerates_null_score_but_counts_it(tmp_path: Path) -> None:
    _write_run(tmp_path, _score_lines([0.1, None, 0.3]))
    result = mj.collect(str(tmp_path))
    assert result.status == Status.COMPLETED
    assert result.metrics["n_null"] == 1 and result.metrics["n_scored"] == 2
    assert result.metrics["score_mean"] == pytest.approx(0.2)


# --- torch discipline -------------------------------------------------------------


def test_gradient_import_is_torch_free() -> None:
    import dataos_train.modeljob.gradient  # noqa: F401
    import dataos_train.modeljob.score_entry  # noqa: F401

    assert "torch" not in sys.modules  # heavy deps load only inside the scoring job
