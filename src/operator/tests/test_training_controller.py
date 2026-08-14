from __future__ import annotations

import json

import pytest


def test_training_controller_applies_online_policy_actions(tmp_path) -> None:
    pytest.importorskip("torch")

    from dataos_operator.training import DataOSController

    train_path = tmp_path / "train.jsonl"
    records = [
        {
            "dataos_id": "sample-high",
            "dataos_domain": "math",
            "messages": [
                {"role": "user", "content": "1+1?"},
                {"role": "assistant", "content": "2"},
            ],
            "metadata": {"runtime": {"reference_loss": 0.1}},
        },
        {
            "dataos_id": "sample-low",
            "dataos_domain": "code",
            "messages": [
                {"role": "user", "content": "print hi"},
                {"role": "assistant", "content": "print('hi')"},
            ],
            "metadata": {"runtime": {"reference_loss": 0.1}},
        },
    ]
    train_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    config_path = tmp_path / "dataos_policy.json"
    config_path.write_text(
        json.dumps(
            {
                "mode": "online",
                "update_interval": 1,
                "train_dataset_paths": [str(train_path)],
                "operators": [
                    {
                        "operator": "loss_selector",
                        "kwargs": {
                            "sample_size": 1,
                            "focus": "high",
                            "focus_weight": 1000.0,
                            "random_state": 0,
                        },
                    },
                    {"operator": "loss_reweighter", "kwargs": {"strategy": "linupper"}},
                    {
                        "operator": "doremi_mixer",
                        "kwargs": {
                            "domain_key": "dataos_domain",
                            "eta": 0.5,
                            "epsilon": 0.0,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    controller = DataOSController.from_config(config_path=config_path, output_dir=tmp_path)
    batch = controller.make_batch(
        record_ids=["sample-high", "sample-low"],
        domains=["math", "code"],
    )
    per_sample_losses = [2.0, 0.5]
    controller.record_forward(
        batch=batch,
        per_sample_losses=per_sample_losses,
        reference_losses=[0.1, 0.1],
    )

    weights = controller.batch_weights(batch=batch, per_sample_losses=per_sample_losses)
    assert len(weights) == 2
    assert pytest.approx(sum(weights)) == 1.0

    controller.refresh_policy(step=0, force=True)

    assert controller.active_ids == {"sample-high"}
    assert controller.current_domain_weights is not None
    assert controller.current_domain_weights["math"] > controller.current_domain_weights["code"]
