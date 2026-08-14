"""Contract: DataOS's build_train_spec output must pass dataos-train's own validate() —
the spec crosses the tool boundary as data, and the two schemas must never drift apart.

dataos-train lives in this repo (an instrument of ``src/dataos-tools``) and is imported from there. It used to be a
separate repo, and this module used to prepend a sibling checkout (``../DataOS-train/src``) to
``sys.path`` as a fallback. That fallback outlived the move and became a hazard: a stale sibling
directory SHADOWED the in-repo package, so running ``pytest tests/ src/dataos-tools/tests/`` in one
process collected an old ``dataos_train`` and died on imports of modules added since. Same shape
as the defects in audit batch 12 — an outdated external copy silently beating the declared one.
"""

import pytest
from pydantic import ValidationError

from dataos.agent.train_spec import build_train_spec

try:
    from dataos_train import TrainSpec, get_backend
except ImportError:  # pragma: no cover - dataos-train is installed in the dev venv
    pytest.skip("dataos_train not importable", allow_module_level=True)


def _resolved() -> dict:
    return {
        "environment": {
            "tools": {
                "sft_env": {"path": "/envs/Llama", "provides": ["llamafactory", "torch"]},
                "rl_env": {"path": "/envs/verl", "provides": ["verl", "torch"]},
            },
            "slurm": {"partition": "gpu", "qos": "gpu_priv"},
        },
        "run": {"resource": {"gpus": 1, "cpus": 8, "node": "gpu04"}},
    }


def _spec(**train) -> dict:
    return build_train_spec(
        _resolved(), name="e0", plan_train=train, parent_model="/models/base",
        run_root="/rb/e0", time_h=2.0,
    )


def test_sft_lora_spec_passes_dataos_train_validate():
    spec = TrainSpec.model_validate(_spec(finetuning_type="lora", learning_rate=1e-4))
    errors = get_backend(spec.backend).validate(spec)
    assert errors == [], f"dataos-train rejected a DataOS-built spec: {errors}"


def test_sft_full_spec_passes_dataos_train_validate():
    spec = TrainSpec.model_validate(_spec(finetuning_type="full", num_train_epochs=2))
    errors = get_backend(spec.backend).validate(spec)
    assert errors == []


def test_unknown_method_keys_flow_via_extra():
    spec = TrainSpec.model_validate(_spec(finetuning_type="lora", some_future_knob=3))
    assert spec.method.extra.get("some_future_knob") == 3


def test_full_env_menu_is_in_the_spec():
    spec = TrainSpec.model_validate(_spec(finetuning_type="lora"))
    assert spec.env.backends["llamafactory"] == "/envs/Llama"
    assert spec.env.backends["verl"] == "/envs/verl"
    assert spec.env.env_for("llamafactory", "sft") == "/envs/Llama"


def test_the_spec_states_no_placement_at_all():
    """The rule two tests used to guard here did not go away; it moved.

    `nodelist` (run pin over env pool) and `qos` (run choice over env default) were asserted on
    a `cluster` block this function wrote into every TrainSpec — the third hand-written copy of
    a precedence dataos-eval and dataos-data already read from env.yaml, and the copy that went
    stale when `qos` moved into the run config. Both now live in `dataos_launch.placement`,
    covered by tools/tests/launch/test_placement.py; dataos-train resolves them at submit from
    --repo/--environment.

    What is left to protect here is the absence. A spec that could state placement would outrank
    the config again, silently, exactly the way the old one did.
    """
    spec = build_train_spec(_resolved(), name="e0", plan_train={"finetuning_type": "lora"},
                            parent_model="/models/base", run_root="/rb/e0", time_h=2.0)
    assert "cluster" not in spec
    assert not {"partition", "qos", "nodelist", "gres"} & set(spec)
    with pytest.raises(ValidationError):   # and dataos-train refuses one that tries
        TrainSpec.model_validate({**spec, "cluster": {"qos": "gpu_normal"}})


def test_sft_field_mirror_matches_dataos_train():
    """`_SFT_FIELDS` decides whether a plan key becomes a TYPED method field or rides in
    `method.extra`. Both reach LLaMA-Factory, but only typed fields are visible to dataos-train's
    own reasoning — `cli.precheck_disk` sizes a run from save_steps/save_total_limit, so a
    checkpoint-saving plan that lands in `extra` would be green-lit on a disk that cannot hold it.
    """
    from dataos.agent.train_spec import _SFT_FIELDS
    from dataos_train.spec import SFT

    # `template` is deliberately NOT plan-settable: it is derived from the model at build
    # (audit 13.2 — an accepted template let a run train under another family's turn framing
    # while the frozen scorer sent this family's). It lives in _STRIP, not _SFT_FIELDS.
    declared = set(SFT.model_fields) - {"kind", "extra", "template"}
    assert _SFT_FIELDS == declared, (
        "dataos.agent.train_spec._SFT_FIELDS and dataos_train.spec.SFT disagree — "
        f"missing here: {sorted(declared - _SFT_FIELDS)}; "
        f"stale here: {sorted(_SFT_FIELDS - declared)}"
    )


def test_the_spec_is_named_after_the_node_not_after_an_env_key():
    """Found in the first end-to-end run on the new stack, in the artifacts rather than in a test.

    `build_train_spec` took `name` as a keyword argument and then wrote `for name in tools:` a few
    lines below, so by the time the return dict read `"name": name` the parameter was gone and
    the value was the LAST key of `env.tools` — `eval_env` on both deployments. Every training
    job in every batch was therefore called `eval_env`: in its spec, in its Job, in the generated
    script's filename, and in `sbatch -J`. squeue and sacct could not tell one experiment from
    another, and a training job read as an evaluation one.
    """
    spec = build_train_spec(_resolved(), name="fmt-full-sft-ckpt",
                            plan_train={"finetuning_type": "lora"}, parent_model="/m",
                            run_root="/r", time_h=2.0)
    assert spec["name"] == "fmt-full-sft-ckpt"
    assert spec["name"] not in (_resolved()["environment"]["tools"] or {})
