"""VerlBackend: validate, build (3-step script), data transform, collect."""

from __future__ import annotations

import json
from pathlib import Path

from dataos_train.backends.verl import VerlBackend
from dataos_train.backends.verl import data as vdata
from dataos_train.backends.verl.collect import parse_metrics
from dataos_train.backends.verl.config import render_overrides
from dataos_train.spec import GRPO, SFT, DatasetRef, EnvSpec, Reward, Rollout, TrainSpec, VerlPPO
from dataos_train.types import Resources, Status


def _spec(**kw) -> TrainSpec:
    base = dict(
        name="grpo",
        backend="verl",
        base_model="/models/base",
        run_dir="/r",
        dataset=DatasetRef(path="/data/rl"),
        method=GRPO(reward=Reward(ref="gsm8k", path="/reward.py")),
        resources=Resources(nodes=1, gpus=8, cpus=16, time_h=6),
        env=EnvSpec(conda_sh="/c.sh", backends={"verl": "verl"}, deploy_root="/deploy"),
    )
    base.update(kw)
    return TrainSpec(**base)


def test_validate() -> None:
    be = VerlBackend()
    assert be.validate(_spec()) == []
    assert any("grpo" in e for e in be.validate(_spec(method=SFT())))
    assert any(
        "reward.path" in e for e in be.validate(_spec(method=GRPO(reward=Reward(ref="gsm8k"))))
    )
    assert any(
        "group_size" in e
        for e in be.validate(_spec(method=GRPO(group_size=4, rollout=Rollout(n=8))))
    )
    assert any("multi-node" in e for e in be.validate(_spec(resources=Resources(nodes=2, gpus=8))))
    assert any(
        "verl conda env" in e
        for e in be.validate(_spec(env=EnvSpec(conda_sh="/c", backends={}, deploy_root="/d")))
    )


def test_verl_ppo_validate() -> None:
    be = VerlBackend()
    assert "verl_ppo" in be.capabilities().methods
    assert be.validate(_spec(method=VerlPPO(reward=Reward(ref="gsm8k", path="/reward.py")))) == []
    assert any(
        "divisible by ppo_mini_batch" in e
        for e in be.validate(_spec(method=VerlPPO(train_batch=100, ppo_mini_batch=64)))
    )
    assert any(
        "reward.path" in e for e in be.validate(_spec(method=VerlPPO()))
    )
    # no group_size coupling for PPO: rollout.n is free
    ok_ppo = VerlPPO(reward=Reward(ref="gsm8k", path="/reward.py"), rollout=Rollout(n=2))
    assert be.validate(_spec(method=ok_ppo)) == []


def test_verl_ppo_render_overrides() -> None:
    ov = render_overrides(
        _spec(method=VerlPPO()),
        VerlPPO(critic_learning_rate=2e-5, critic_warmup=1, kl_coef=0.002),
        "/m",
        Resources(gpus=2, nodes=1),
        "/reward.py",
        "/d/rl",
    )
    j = "\n".join(ov)
    assert "algorithm.adv_estimator=gae" in j
    assert "algorithm.use_kl_in_reward=true" in j
    assert "algorithm.kl_ctrl.kl_coef=0.002" in j
    assert "actor_rollout_ref.actor.use_kl_loss=false" in j
    assert "critic.model.path=/m" in j  # defaults to base_model
    assert "critic.optim.lr=2e-05" in j
    assert "critic.ppo_micro_batch_size_per_gpu=4" in j
    assert "trainer.critic_warmup=1" in j
    assert "kl_loss_coef" not in j and "lora_rank" not in j
    # explicit critic model wins over base_model
    ov2 = render_overrides(
        _spec(), VerlPPO(critic_model="/critic"), "/m", Resources(gpus=2), "/r.py", "/d"
    )
    assert "critic.model.path=/critic" in "\n".join(ov2)


def test_verl_ppo_build() -> None:
    job = VerlBackend().build(_spec(method=VerlPPO()))
    s = job.assets["run_train.sh"]
    assert "verl.trainer.main_ppo" in s and "--bench gsm8k" in s
    assert "dataos_train.backends.verl.collect --merge" in s
    meta = json.loads(job.assets["dataos_meta.json"])
    assert meta["mode"] == "verl_ppo" and meta["finetuning_type"] == "full"
    assert job.conda_env == "verl"  # falls back to the backend-level env domain


def test_env_domain_scoped_key() -> None:
    scoped = _spec(env=EnvSpec(conda_sh="/c", backends={"verl:grpo": "verl-grpo"}))
    assert VerlBackend().validate(scoped) == []
    assert VerlBackend().build(scoped).conda_env == "verl-grpo"
    # the scoped domain wins over the backend-level one
    both = _spec(
        env=EnvSpec(conda_sh="/c", backends={"verl": "verl", "verl:grpo": "verl-grpo"})
    )
    assert VerlBackend().build(both).conda_env == "verl-grpo"


def test_validate_tensor_parallel_matches_local_model_config(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"num_attention_heads": 14}))
    bad = GRPO(
        group_size=4,
        rollout=Rollout(n=4, tensor_parallel_size=4),
        ppo_micro_batch_per_gpu=1,
    )
    good = bad.model_copy(update={"rollout": Rollout(n=4, tensor_parallel_size=2)})

    assert any(
        "num_attention_heads" in e
        for e in VerlBackend().validate(_spec(base_model=str(model), method=bad))
    )
    assert not any(
        "num_attention_heads" in e
        for e in VerlBackend().validate(_spec(base_model=str(model), method=good))
    )


def test_render_overrides() -> None:
    ov = render_overrides(_spec(), GRPO(), "/m", Resources(gpus=8, nodes=1), "/reward.py", "/d/rl")
    j = "\n".join(ov)
    assert "actor_rollout_ref.model.path=/m" in j
    assert "algorithm.adv_estimator=grpo" in j
    assert (
        "custom_reward_function.path=/reward.py" in j
        and "custom_reward_function.name=compute_score" in j
    )
    assert "actor_rollout_ref.actor.use_kl_loss=true" in j
    assert "foo.bar=7" in render_overrides(
        _spec(overrides=["foo.bar=7"]), GRPO(), "/m", Resources(gpus=8), "/r.py", "/d"
    )


def test_build_script_and_env() -> None:
    job = VerlBackend().build(_spec())
    s = job.assets["run_train.sh"]
    assert job.command == ["bash", "run_train.sh"]
    assert "dataos_train.backends.verl.data" in s and "--bench gsm8k" in s
    assert "verl.trainer.main_ppo" in s
    assert "dataos_train.backends.verl.collect --merge" in s
    assert "custom_reward_function.path=/reward.py" in s
    assert "DATAOS_BRIDGE_ROOT" not in job.env
    assert job.env["HF_HUB_OFFLINE"] == "1" and job.conda_env == "verl"
    assert json.loads(job.assets["dataos_meta.json"])["bench"] == "gsm8k"


def test_to_verl_rows() -> None:
    rows = vdata.to_verl_rows(
        [
            {"prompt": "6*7?", "ground_truth": "x\n#### 42", "data_source": "gsm8k"},
            {"question": "2+2?", "answer": "4"},
        ],
        "gsm8k",
        "{prompt} GO",
    )
    assert rows[0]["prompt"][0]["content"] == "6*7? GO"
    assert rows[0]["reward_model"]["ground_truth"] == "42"  # #### stripped
    assert rows[1]["data_source"] == "gsm8k" and rows[1]["reward_model"]["ground_truth"] == "4"


def test_collect(tmp_path: Path) -> None:
    (tmp_path / "final_model").mkdir()
    (tmp_path / "final_model" / "config.json").write_text("{}")
    (tmp_path / "final_model" / "model.safetensors").write_text("")
    (tmp_path / "data" / "rl").mkdir(parents=True)
    (tmp_path / "data" / "rl" / "_count.json").write_text(json.dumps({"rows": 128}))
    (tmp_path / "dataos_meta.json").write_text(
        json.dumps(
            {
                "base_model": "/models/base",
                "bench": "gsm8k",
                "finetuning_type": "full",
                "mode": "grpo",
            }
        )
    )
    (tmp_path / "verl").mkdir()
    (tmp_path / "verl" / "metrics.jsonl").write_text(
        json.dumps({"global_step": 5, "reward": 0.42}) + "\n"
    )
    res = VerlBackend().collect(str(tmp_path))
    assert res.status == Status.COMPLETED and res.base_model == "/models/base" and res.rows == 128
    assert res.metrics["reward"] == 0.42 and res.report_fragment["model"]["mode"] == "grpo"


def test_parse_metrics_logline(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "j.out").write_text(
        "step:10 critic/rewards/mean: 0.7 actor/kl_loss=0.01\n"
    )
    m = parse_metrics(str(tmp_path))
    assert m["critic/rewards/mean"] == 0.7 and m["actor/kl_loss"] == 0.01
