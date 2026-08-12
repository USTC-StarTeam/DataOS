"""LlamaFactoryBackend: canonical->alpaca convert, config render, build, collect (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import MODEL_DIR

from dataos_train.backends.llamafactory import LlamaFactoryBackend
from dataos_train.backends.llamafactory.config import render_train_config
from dataos_train.data import convert
from dataos_train.schemas import backend_schema, method_schema
from dataos_train.spec import DPO, GRPO, KTO, ORPO, PPO, SFT, DatasetRef, EnvSpec, TrainSpec
from dataos_train.types import Resources, Status


def test_convert_formats() -> None:
    a = convert.to_alpaca([{"instruction": "q", "input": "", "output": "a"}], "auto")
    assert a[0] == {"instruction": "q", "input": "", "output": "a"}
    m = convert.to_alpaca(
        [
            {
                "messages": [
                    {"role": "system", "content": "S"},
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "u2"},
                    {"role": "assistant", "content": "a2"},
                ]
            }
        ],
        "auto",
    )
    assert m[0]["instruction"] == "u2" and m[0]["output"] == "a2"
    assert m[0]["history"] == [["u1", "a1"]] and m[0]["system"] == "S"
    sg = convert.to_alpaca(
        [{"conversations": [{"from": "human", "value": "h"}, {"from": "gpt", "value": "g"}]}],
        "auto",
    )
    assert sg[0]["instruction"] == "h" and sg[0]["output"] == "g"
    qa = convert.to_alpaca([{"question": "6*7?", "answer": "#### 42"}], "auto")
    assert qa[0] == {"instruction": "6*7?", "input": "", "output": "#### 42"}


def test_convert_writes_dataset(tmp_path: Path) -> None:
    recs = convert.to_alpaca([{"prompt": "2+2?", "target": "4"}], "auto")
    n = convert.write_alpaca_dataset(recs, str(tmp_path / "sft"))
    assert n == 1
    info = json.loads((tmp_path / "sft" / "dataset_info.json").read_text())
    # The columns map is load-bearing: LF drops any optional alpaca column it was not told about,
    # so omitting it silently discarded the `system` field that convert writes into every row.
    assert info == {
        "dataos_sft": {
            "file_name": "train.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
                "history": "history",
            },
        }
    }


def test_convert_preference_kto_ppo(tmp_path: Path) -> None:
    pref = convert.to_preference(
        [{"prompt": "q", "chosen": "good", "rejected": "bad", "system": "S"}], "auto"
    )
    assert pref == [
        {
            "instruction": "q",
            "input": "",
            "system": "S",
            "history": [],
            "chosen": "good",
            "rejected": "bad",
        }
    ]
    assert convert.write_preference_dataset(pref, str(tmp_path / "dpo"), "dataos_dpo") == 1
    pref_info = json.loads((tmp_path / "dpo" / "dataset_info.json").read_text())
    assert pref_info["dataos_dpo"]["ranking"] is True
    assert pref_info["dataos_dpo"]["columns"]["chosen"] == "chosen"
    pref_rows = json.loads((tmp_path / "dpo" / "train.json").read_text())
    assert pref_rows[0]["history"] == []

    kto = convert.to_kto([{"prompt": "q", "response": "a", "label": "false"}], "auto")
    assert kto == [
        {
            "instruction": "q",
            "input": "",
            "system": "",
            "history": [],
            "output": "a",
            "kto_tag": False,
        }
    ]
    assert convert.write_kto_dataset(kto, str(tmp_path / "kto"), "dataos_kto") == 1
    kto_info = json.loads((tmp_path / "kto" / "dataset_info.json").read_text())
    assert kto_info["dataos_kto"]["columns"]["kto_tag"] == "kto_tag"

    ppo = convert.to_ppo([{"messages": [{"role": "user", "content": "q"}]}], "auto")
    assert ppo == [{"instruction": "q", "input": "", "system": "", "history": [], "output": ""}]
    assert convert.write_ppo_dataset(ppo, str(tmp_path / "ppo"), "dataos_ppo") == 1


def test_schema_and_capabilities_include_rlhf_methods() -> None:
    caps = LlamaFactoryBackend().capabilities()
    for method in ("sft", "dpo", "ppo", "kto", "orpo"):
        assert method in caps.methods
        assert method in backend_schema("llamafactory")["methods"]
        assert "kind" in method_schema(method)["properties"]


def test_render_config_lora_vs_full() -> None:
    spec = TrainSpec(
        name="e",
        base_model=MODEL_DIR,
        run_dir="/r",
        dataset=DatasetRef(path="/d"),
        method=SFT(finetuning_type="lora", learning_rate=1.5e-4, lora_rank=64, template="qwen3"),
        overrides={"flash_attn": "fa2"},
    )
    cfg = render_train_config(spec, spec.method, "/r/data/sft", "/r/adapter")
    assert cfg["model_name_or_path"] == MODEL_DIR
    assert cfg["lora_rank"] == 64 and cfg["learning_rate"] == 1.5e-4
    assert cfg["flash_attn"] == "fa2"
    full = render_train_config(
        TrainSpec(
            name="f",
            base_model=MODEL_DIR,
            run_dir="/r",
            dataset=DatasetRef(path="/d"),
            method=SFT(finetuning_type="full", template="qwen3"),
        ),
        SFT(finetuning_type="full", template="qwen3"),
        "/r/data/sft",
        "/r/final_model",
    )
    assert "lora_rank" not in full

    orpo = render_train_config(
        TrainSpec(
            name="o",
            base_model=MODEL_DIR,
            run_dir="/r",
            dataset=DatasetRef(path="/d"),
            method=ORPO(extra={"stage": "sft", "dataset": "x"}, template="qwen3"),
        ),
        ORPO(extra={"stage": "sft", "dataset": "x"}, template="qwen3"),
        "/r/data/orpo",
        "/r/adapter",
    )
    assert orpo["stage"] == "dpo" and orpo["pref_loss"] == "orpo"
    assert orpo["dataset"] == "dataos_orpo"


def test_validate() -> None:
    be = LlamaFactoryBackend()
    ok = TrainSpec(
        name="a", base_model=MODEL_DIR, run_dir="/r", dataset=DatasetRef(path="/d"), method=SFT(template="qwen3")
    )
    assert be.validate(ok) == []
    grpo = TrainSpec(
        name="b", base_model=MODEL_DIR, run_dir="/r", dataset=DatasetRef(path="/d"), method=GRPO()
    )
    assert any("supports method.kind" in e for e in be.validate(grpo))
    assert any(
        "dataset is required" in e
        for e in be.validate(TrainSpec(name="c", base_model=MODEL_DIR, run_dir="/r", method=SFT(template="qwen3")))
    )
    leak = TrainSpec(
        name="d",
        base_model=MODEL_DIR,
        run_dir="/r",
        dataset=DatasetRef(path="/d"),
        method=SFT(extra={"model_name_or_path": "/x"}, template="qwen3"),
    )
    assert any("model_name_or_path" in e for e in be.validate(leak))
    ppo = TrainSpec(
        name="e", base_model=MODEL_DIR, run_dir="/r", dataset=DatasetRef(path="/d"), method=PPO(template="qwen3")
    )
    assert any("reward_model" in e for e in be.validate(ppo))
    ppo_full_lora_reward = TrainSpec(
        name="f",
        base_model=MODEL_DIR,
        run_dir="/r",
        dataset=DatasetRef(path="/d"),
        method=PPO(finetuning_type="full", reward_model="/rm", reward_model_type="lora", template="qwen3"),
    )
    assert any("reward_model_type='lora'" in e for e in be.validate(ppo_full_lora_reward))


def test_ppo_env_domain() -> None:
    be = LlamaFactoryBackend()
    assert be.capabilities().env_domains == {"ppo": "llamafactory:ppo"}

    def ppo_spec(env: EnvSpec) -> TrainSpec:
        return TrainSpec(
            name="p",
            base_model=MODEL_DIR,
            run_dir="/r",
            dataset=DatasetRef(path="/d"),
            method=PPO(reward_model="/rm", template="qwen3"),
            env=env,
        )

    # the trap: a generic LF env only — PPO would fail at runtime, so validate rejects it
    generic_only = ppo_spec(EnvSpec(backends={"llamafactory": "Llama"}))
    assert any("llamafactory:ppo" in e for e in be.validate(generic_only))

    # ambient env (no conda mapping at all) stays allowed
    assert be.validate(ppo_spec(EnvSpec())) == []

    # the scoped domain satisfies validate and wins at build time
    scoped = ppo_spec(
        EnvSpec(backends={"llamafactory": "Llama", "llamafactory:ppo": "LlamaPPO"})
    )
    assert be.validate(scoped) == []
    assert be.build(scoped).conda_env == "LlamaPPO"

    # non-PPO methods keep resolving the backend-level env
    sft = TrainSpec(
        name="s",
        base_model=MODEL_DIR,
        run_dir="/r",
        dataset=DatasetRef(path="/d"),
        method=SFT(template="qwen3"),
        env=EnvSpec(backends={"llamafactory": "Llama", "llamafactory:ppo": "LlamaPPO"}),
    )
    assert be.build(sft).conda_env == "Llama"


def test_build_lora(tmp_path: Path) -> None:
    spec = TrainSpec(
        name="exp",
        base_model=MODEL_DIR,
        run_dir=str(tmp_path),
        dataset=DatasetRef(path="/data/canon", format="messages"),
        method=SFT(finetuning_type="lora", template="qwen3"),
        resources=Resources(nodes=1, gpus=2, cpus=8, time_h=2),
        env=EnvSpec(conda_sh="/c.sh", backends={"llamafactory": "Llama"}, deploy_root="/deploy"),
        cluster={"partition": "gpu", "gres": "gpu:2", "nodelist": "gpu04"},
    )
    job = LlamaFactoryBackend().build(spec)
    assert job.command == ["bash", "run_train.sh"]
    s = job.assets["run_train.sh"]
    assert "dataos_train.data.convert" in s and "llamafactory-cli train" in s
    assert "--to alpaca --name dataos_sft" in s
    assert "llamafactory-cli export" in s and "final_model" in s
    assert job.env["FORCE_TORCHRUN"] == "1"
    assert job.conda_env == "Llama" and job.pythonpath == ["/deploy/src", "/deploy"]
    assert job.slurm.gres == "gpu:2" and job.slurm.nodelist == "gpu04"


def test_build_preference_rlhf_methods(tmp_path: Path) -> None:
    cases = [
        (DPO(template="qwen3"), "dpo", "stage: dpo", "--to preference --name dataos_dpo", "pref_loss: sigmoid"),
        (ORPO(template="qwen3"), "orpo", "stage: dpo", "--to preference --name dataos_orpo", "pref_loss: orpo"),
        (KTO(template="qwen3"), "kto", "stage: kto", "--to kto --name dataos_kto", "kto_chosen_weight: 1"),
        (
            PPO(reward_model="/rm", template="qwen3"),
            "ppo",
            "stage: ppo",
            "--to ppo --name dataos_ppo",
            "reward_model: /rm",
        ),
    ]
    for method, name, stage, convert_arg, method_key in cases:
        run_dir = tmp_path / name
        spec = TrainSpec(
            name=name,
            base_model=MODEL_DIR,
            run_dir=str(run_dir),
            dataset=DatasetRef(path="/d"),
            method=method,
        )
        job = LlamaFactoryBackend().build(spec)
        cfg = job.assets["train_config.yaml"]
        script = job.assets["run_train.sh"]
        meta = json.loads(job.assets["dataos_meta.json"])
        assert f"model_name_or_path: {MODEL_DIR}" in cfg
        assert stage in cfg
        assert f"dataset: dataos_{name}" in cfg
        assert method_key in cfg
        assert convert_arg in script
        assert "llamafactory-cli export" in script
        assert meta["mode"] == name and meta["method"] == name
        assert meta["base_model"] == MODEL_DIR


def test_build_full_no_merge(tmp_path: Path) -> None:
    spec = TrainSpec(
        name="exp",
        base_model=MODEL_DIR,
        run_dir=str(tmp_path),
        dataset=DatasetRef(path="/d"),
        method=SFT(finetuning_type="full", template="qwen3"),
    )
    job = LlamaFactoryBackend().build(spec)
    assert "llamafactory-cli export" not in job.assets["run_train.sh"]
    # LF 0.9.x only loads YAML configs correctly (its JSON branch is buggy)
    assert "train_config.yaml" in job.assets
    assert (
        "output_dir: " in job.assets["train_config.yaml"]
        and "final_model" in job.assets["train_config.yaml"]
    )


def test_lora_export_only_for_lora_and_qlora(tmp_path: Path) -> None:
    qlora = TrainSpec(
        name="q",
        base_model=MODEL_DIR,
        run_dir=str(tmp_path / "q"),
        dataset=DatasetRef(path="/d"),
        method=DPO(quantization_bit=4, template="qwen3"),
    )
    qjob = LlamaFactoryBackend().build(qlora)
    assert "quantization_bit: 4" in qjob.assets["train_config.yaml"]
    assert "llamafactory-cli export" in qjob.assets["run_train.sh"]

    full = TrainSpec(
        name="f",
        base_model=MODEL_DIR,
        run_dir=str(tmp_path / "f"),
        dataset=DatasetRef(path="/d"),
        method=ORPO(finetuning_type="full", template="qwen3"),
    )
    fjob = LlamaFactoryBackend().build(full)
    assert "llamafactory-cli export" not in fjob.assets["run_train.sh"]

    freeze = TrainSpec(
        name="z",
        base_model=MODEL_DIR,
        run_dir=str(tmp_path / "z"),
        dataset=DatasetRef(path="/d"),
        method=KTO(finetuning_type="freeze", template="qwen3"),
    )
    zjob = LlamaFactoryBackend().build(freeze)
    assert "llamafactory-cli export" not in zjob.assets["run_train.sh"]


def test_collect_success(tmp_path: Path) -> None:
    (tmp_path / "final_model").mkdir()
    (tmp_path / "final_model" / "config.json").write_text("{}")
    (tmp_path / "final_model" / "model.safetensors").write_text("")
    (tmp_path / "adapter").mkdir()
    (tmp_path / "adapter" / "all_results.json").write_text(
        json.dumps({"train_loss": 0.5, "note": "x"})
    )
    (tmp_path / "data" / "sft").mkdir(parents=True)
    (tmp_path / "data" / "sft" / "train.json").write_text(json.dumps([{"instruction": "a"}] * 5))
    (tmp_path / "dataos_meta.json").write_text(
        json.dumps(
            {
                "base_model": "/models/base",
                "finetuning_type": "lora",
                "template": "qwen3",
                "mode": "sft",
            }
        )
    )
    res = LlamaFactoryBackend().collect(str(tmp_path))
    assert res.status == Status.COMPLETED and res.model_path.endswith("final_model")
    assert res.base_model == "/models/base" and res.rows == 5
    assert res.metrics["train_loss"] == 0.5 and "note" not in res.metrics
    assert res.report_fragment["model"]["finetuning_type"] == "lora"


def test_collect_success_new_method_path(tmp_path: Path) -> None:
    (tmp_path / "final_model").mkdir()
    (tmp_path / "final_model" / "config.json").write_text("{}")
    (tmp_path / "final_model" / "model.safetensors").write_text("")
    (tmp_path / "data" / "orpo").mkdir(parents=True)
    (tmp_path / "data" / "orpo" / "train.json").write_text(json.dumps([{"instruction": "a"}] * 2))
    (tmp_path / "dataos_meta.json").write_text(
        json.dumps(
            {
                "base_model": "/models/base",
                "finetuning_type": "full",
                "template": "qwen3",
                "mode": "orpo",
                "method": "orpo",
            }
        )
    )
    res = LlamaFactoryBackend().collect(str(tmp_path))
    assert res.status == Status.COMPLETED
    assert res.rows == 2
    assert res.artifacts["data"].endswith("data/orpo")
    assert res.report_fragment["model"]["method"] == "orpo"


def test_collect_failure(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "j.out").write_text("boom\n")
    res = LlamaFactoryBackend().collect(str(tmp_path))
    assert res.status == Status.FAILED and "HF model" in res.error
    assert res.log_tail == "boom"


def test_checkpoint_saving_is_off_by_default_and_weights_only_when_on(tmp_path: Path) -> None:
    """Sampling the training trajectory is opt-in, and when it is on the checkpoints must be
    weights-only: the optimizer state is ~4x the weights and we never resume a run — a checkpoint
    exists to be evaluated and to be continued from as a fresh run."""
    def _cfg(**method_kwargs):
        spec = TrainSpec(
            name="e",
            base_model=MODEL_DIR,
            run_dir=str(tmp_path),
            dataset=DatasetRef(path="/d"),
            method=SFT(template="qwen3", finetuning_type="full", **method_kwargs),
        )
        return render_train_config(spec, spec.method, "/data", str(tmp_path), "ds")

    off = _cfg()
    assert off["save_strategy"] == "no"
    assert "save_steps" not in off and "save_only_model" not in off

    on = _cfg(save_steps=150, save_total_limit=4)
    assert on["save_strategy"] == "steps"
    assert on["save_steps"] == 150
    assert on["save_total_limit"] == 4
    assert on["save_only_model"] is True


def test_system_survives_the_round_trip_to_llamafactory(tmp_path: Path) -> None:
    """A row's `system` must reach training, not just the disk.

    convert materialises `system` into every alpaca row, but LLaMA-Factory only indexes optional
    columns declared in dataset_info.json — so for one batch the field was written and then
    ignored, and bfcl lost 33 points when a recipe moved its tool schema from `instruction` into
    `system` (audit 13.4). This pins both halves: the value is in the row AND the column is
    declared, which is what makes LF read it.
    """
    recs = convert.to_alpaca(
        [{"messages": [
            {"role": "system", "content": "TOOLS: [get_weather]"},
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "<tool_call>"},
        ]}],
        "auto",
    )
    assert recs[0]["system"] == "TOOLS: [get_weather]"
    convert.write_alpaca_dataset(recs, str(tmp_path / "sft"))
    info = json.loads((tmp_path / "sft" / "dataset_info.json").read_text())["dataos_sft"]
    assert info["columns"]["system"] == "system"
    rows = json.loads((tmp_path / "sft" / "train.json").read_text())
    assert rows[0]["system"] == "TOOLS: [get_weather]"


def test_every_declared_column_exists_in_every_row(tmp_path: Path) -> None:
    """Simulates how LF actually reads the dataset, which the test above did not.

    LF's alpaca converter subscripts an optional column unconditionally once it is mapped, so a
    declared-but-absent column is not ignored — it raises. The test above fed one row that HAD a
    system message and only checked that the column was declared, so it passed while the declaring
    half was shipped without the guaranteeing half: the Gemma3 bfcl run then died with
    ``KeyError: 'history'`` 12 seconds into four consecutive jobs, on its last round of budget.

    The failure needs MIXED turn counts to appear at all. ``to_alpaca`` emits ``history`` for a
    multi-turn record and omits it for a single-turn one, so a dataset of either kind alone can
    look fine while their union is the landmine.
    """
    recs = convert.to_alpaca(
        [
            {"messages": [  # multi-turn -> carries `history`
                {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"},
            ]},
            {"messages": [  # single-turn -> `history` absent before materialization
                {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
            ]},
        ],
        "auto",
    )
    convert.write_alpaca_dataset(recs, str(tmp_path / "sft"))
    info = json.loads((tmp_path / "sft" / "dataset_info.json").read_text())["dataos_sft"]
    rows = json.loads((tmp_path / "sft" / "train.json").read_text())
    missing = [
        (i, column) for i, row in enumerate(rows)
        for column in info["columns"].values() if column not in row
    ]
    assert not missing, f"LF would raise KeyError on {missing}"
    # and the multi-turn row keeps its turns: dropping the column would silently truncate the
    # conversation to its last exchange, which is the same class of loss as the `system` incident
    assert rows[0]["history"] == [["u1", "a1"]]
    assert rows[1]["history"] == []
