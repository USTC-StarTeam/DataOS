"""Cross-repo contract: DataOS's build_train_spec output must pass dataos-train's
own validate() — the two repos' schemas must never drift apart.

dataos-train is NOT a DataOS dependency (the spec crosses the boundary as data);
locally it is importable via a sibling checkout (../DataOS-train) or an installed
package — the test skips when neither is available.
"""

import sys
from pathlib import Path

import pytest

from dataos.agent.train_spec import build_train_spec

_SIBLING = Path(__file__).resolve().parents[3] / "DataOS-train" / "src"

try:
    if _sibling := (_SIBLING if _SIBLING.is_dir() else None):
        sys.path.insert(0, str(_sibling))
    from dataos_train import TrainSpec, get_backend  # type: ignore
except ImportError:  # pragma: no cover
    pytest.skip("dataos_train not importable (no package, no ../DataOS-train checkout)",
                allow_module_level=True)


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
