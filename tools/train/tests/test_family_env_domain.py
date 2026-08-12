"""The training env is a function of the model family, declared by the environment config.

Architecture support is a dependency fact, not something a spec can assert: SmolLM3 needs
transformers >= 4.53, so an env pinned below that cannot train it. In the 2026-08-11 smol3
batch the sft env was pinned at 4.52.3 and 146 of 677 jobs (22%) were pure workaround —
112 architecture errors, 34 failed version checks, 23 hand-written scripts in which agents
borrowed another env to train in. None of it was visible outside the agent transcripts.

So the environment declares which families each env can train (``tools.*.model_families``),
and the tool resolves the domain from the model's OWN ``model_type`` — it holds no model table.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataos_train.backends.llamafactory import LlamaFactoryBackend
from dataos_train.spec import SFT, DatasetRef, EnvSpec, TrainSpec


def _model_dir(tmp_path: Path, model_type: str) -> str:
    d = tmp_path / f"model-{model_type}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    return str(d)


def _spec(tmp_path: Path, base_model: str, backends: dict[str, str], **method) -> TrainSpec:
    return TrainSpec(
        name="e0",
        base_model=base_model,
        run_dir=str(tmp_path / "run"),
        dataset=DatasetRef(path=str(tmp_path / "data")),
        method=SFT(**method),
        env=EnvSpec(backends=backends),
    )


def test_env_for_resolves_most_specific_first() -> None:
    env = EnvSpec(
        backends={
            "llamafactory": "plain",
            "llamafactory:ppo": "ppo",
            "llamafactory@smollm3": "smol",
            "llamafactory:ppo@smollm3": "ppo-smol",
        }
    )
    assert env.env_for("llamafactory", "ppo", "smollm3") == "ppo-smol"
    assert env.env_for("llamafactory", "ppo", "qwen3") == "ppo"
    assert env.env_for("llamafactory", "sft", "smollm3") == "smol"
    assert env.env_for("llamafactory", "sft", "qwen3") == "plain"
    assert env.env_for("llamafactory") == "plain"
    assert env.env_for("verl") is None


def test_build_runs_the_model_in_its_family_env(tmp_path: Path) -> None:
    backends = {"llamafactory@qwen3": "/envs/old", "llamafactory@smollm3": "/envs/new"}
    for family, expected in (("qwen3", "/envs/old"), ("smollm3", "/envs/new")):
        job = LlamaFactoryBackend().build(_spec(tmp_path, _model_dir(tmp_path, family), backends))
        assert job.conda_env == expected
        # the launch plan and the self-sufficient script must agree, or running run_train.sh
        # by hand silently uses a different interpreter than the submitted job
        assert f"source {expected}/bin/activate" in job.assets["run_train.sh"]


def test_an_undeclared_family_stops_the_build(tmp_path: Path) -> None:
    """No fallback to the unscoped domain: that fallback IS the 146-dead-job failure mode."""
    # gemma3 has a template, so this isolates the env failure from the template one
    spec = _spec(
        tmp_path,
        _model_dir(tmp_path, "gemma3"),
        {"llamafactory": "/envs/any", "llamafactory@qwen3": "/envs/old"},
    )
    with pytest.raises(ValueError, match="model_families"):
        LlamaFactoryBackend().build(spec)


def test_an_environment_without_scoping_is_unchanged(tmp_path: Path) -> None:
    """Declaring no families keeps the old single-env behaviour: the unscoped domain still wins."""
    spec = _spec(tmp_path, _model_dir(tmp_path, "qwen3"), {"llamafactory": "/envs/Llama"})
    assert LlamaFactoryBackend().build(spec).conda_env == "/envs/Llama"


def test_the_template_is_derived_never_accepted(tmp_path: Path) -> None:
    """Two halves of one rule: build needs a readable model, and refuses a contradicting template.

    Naming a template explicitly used to make an unreadable model workable — and that affordance
    is what let the 2026-08-11 gsm8k batch train under another family's turn framing while the
    frozen scorer sent this family's, producing the run's top score that way (audit 13.2).
    """
    with pytest.raises(ValueError, match="cannot derive the model family"):
        LlamaFactoryBackend().build(_spec(tmp_path, "/not/on/this/host", {"llamafactory": "/e"}))

    contradicting = _spec(
        tmp_path, _model_dir(tmp_path, "smollm3"), {"llamafactory": "/e"}, template="qwen3"
    )
    with pytest.raises(ValueError, match="contradicts the model"):
        LlamaFactoryBackend().build(contradicting)


def test_the_family_also_drives_the_template(tmp_path: Path) -> None:
    """One model_type read, both decisions — they cannot disagree about what model this is."""
    job = LlamaFactoryBackend().build(
        _spec(tmp_path, _model_dir(tmp_path, "smollm3"), {"llamafactory@smollm3": "/envs/new"})
    )
    assert "template: smollm3" in job.assets["train_config.yaml"]
    assert job.conda_env == "/envs/new"
