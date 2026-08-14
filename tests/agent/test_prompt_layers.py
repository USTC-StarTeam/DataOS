"""The prompt layering, as executable invariants.

Layout: ``src/dataos/prompt/<role>/{context,turn}/*.yaml``.

  context/  composed once per run into the CLI's own context file (AGENTS.md / CLAUDE.md /
            QWEN.md) — see dataos_agent.context_filename and llm.config.build_agent
  turn/     rendered per turn into that turn's prompt

Two invariants keep the split honest, so a later prompt edit cannot quietly violate it:

  1. membership — the directory decides WHICH blocks a role has, ``_BLOCKS`` only their
     ORDER, and no block belongs to two roles
  2. variable domain — a context template may reference only run-level facts
     (``prompt_facts``), never a per-turn value
"""

from pathlib import Path

import jinja2
import pytest
from jinja2 import meta as jinja_meta

from dataos import prompt as _prompt
from dataos.agent.backend import _BLOCKS, prompt_facts
from dataos.agent.config import AutoRunConfig

_PROMPT_DIR = Path(_prompt.__file__).resolve().parent
_ROLES = sorted(_BLOCKS)
_FACT_KEYS = set(prompt_facts(AutoRunConfig(project_id="t", workspace="/tmp/x")))


def _context_blocks(role: str) -> list[str]:
    return sorted(p.stem for p in (_PROMPT_DIR / role / "context").glob("*.yaml"))


@pytest.mark.parametrize("role", _ROLES)
def test_context_dir_and_blocks_agree(role):
    """Adding a file under <role>/context/ without wiring it (or the reverse) must fail."""
    wired = [b.rsplit("/", 1)[-1] for b in _BLOCKS[role]]
    assert len(wired) == len(set(wired)), f"{role}: a block is listed twice in _BLOCKS"
    assert sorted(wired) == _context_blocks(role)


@pytest.mark.parametrize("role", _ROLES)
def test_no_block_is_shared_between_roles(role):
    """A block serving two roles has to be written for both, so it sharpens neither.
    Every role owns its own copy, free to diverge."""
    for block in _BLOCKS[role]:
        assert block.startswith(f"{role}/context/"), f"{role} pulls {block} from another role"


def test_turn_templates_are_not_part_of_the_context_file():
    for role, blocks in _BLOCKS.items():
        assert not any("/turn/" in b for b in blocks), f"{role} composes a turn template"


@pytest.mark.parametrize("role", _ROLES)
def test_every_role_has_turn_templates(role):
    assert list((_PROMPT_DIR / role / "turn").glob("*.yaml")), f"{role} has no turn template"


@pytest.mark.parametrize("role", _ROLES)
def test_context_templates_use_only_run_level_facts(role):
    """A context file is written ONCE per run. A per-turn variable in it would freeze at the
    run's first value for every later turn — so the legal variables are exactly the facts."""
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    for name in _context_blocks(role):
        source = _prompt._template_text(f"{role}/context/{name}")
        used = jinja_meta.find_undeclared_variables(env.parse(source))
        assert used <= _FACT_KEYS, (
            f"{role}/context/{name} references {sorted(used - _FACT_KEYS)}, which is not a "
            f"run-level fact ({sorted(_FACT_KEYS)}) — move it to {role}/turn/"
        )


def test_the_guard_prescription_follows_the_qos_choice(monkeypatch):
    """The one constraint that is a CONSEQUENCE of a choice, not a fact about the machine.

    It used to live as prose in env.yaml, which is why there were two environment files: 79 lines
    of identical description duplicated so that one line of it could differ. Now the choice is
    `qos` and the prompt carries both branches, so one env suffices.
    """
    from dataos.agent.backend import system_prompt
    from dataos.agent.run_config import ResourceConfig, _cluster

    def _prompt_for(qos: str) -> str:
        # the QOS travels on the RUN, which is the whole point; env.yaml only says which one
        # is the guarded one
        env = {"access": {}, "slurm": {"guard_qos": "gpu_normal",
                                       "qos_available": ["gpu_priv", "gpu_normal"],
                                       "mem_per_gpu_mb": 125000, "cpus_per_gpu": 11}}
        res = ResourceConfig.from_dict({"gpus": 1, "cpus": 8, "qos": qos},
                                       qos_available=env["slurm"]["qos_available"])
        cfg = AutoRunConfig(project_id="t", workspace="/tmp/x", target_bench="gsm8k")
        cfg.runtime_config = {"cluster": _cluster(env, res)}
        return system_prompt("experimenter", cfg)

    guarded = _prompt_for("gpu_normal")
    assert "GPU GUARD" in guarded and "30 GiB" in guarded

    unguarded = _prompt_for("gpu_priv")
    assert "GPU GUARD" not in unguarded
    assert "NOT watched by the GPU Guard" in unguarded


def test_the_machines_limits_are_numbers_from_the_env_not_prose():
    """A hard limit stated in the prompt has to come from the machine's own declaration.

    env.yaml carried "每张卡 125000 MB + 11 逻辑 CPU" as text in a `discipline` list — untested,
    per-environment, and re-sent on every experiment turn. The number now travels as a fact.
    """
    from dataos.agent.backend import system_prompt
    from dataos.agent.run_config import ResourceConfig, _cluster

    env = {"access": {}, "slurm": {"mem_per_gpu_mb": 64000, "qos_available": ["gpu_priv"],
                                   "cpus_per_gpu": 7, "gres_required": "gpu:2"}}
    res = ResourceConfig.from_dict({"gpus": 1, "cpus": 8, "qos": "gpu_priv"},
                                   qos_available=["gpu_priv"])
    cfg = AutoRunConfig(project_id="t", workspace="/tmp/x", target_bench="gsm8k")
    cfg.runtime_config = {"cluster": _cluster(env, res)}
    text = system_prompt("experimenter", cfg)
    assert "64000 MB" in text and "7 logical CPUs" in text and "gpu:2" in text


def test_the_qos_is_the_runs_and_a_typo_is_refused_at_load():
    """The change that let two near-identical environment files collapse into one.

    `qos` was an env.yaml field with no run-level override, so switching between gpu_priv and
    gpu_normal meant switching ENVIRONMENTS — and 18 run configs pointed at a second env whose
    only real difference was that one line. It is now required on the run and checked against
    what the environment says exists, because the failure mode of a wrong QOS is a job that PENDs
    or gets scancelled with nothing that names the cause.
    """
    import pytest

    from dataos.agent.run_config import ResourceConfig, _cluster

    env = {"access": {}, "slurm": {"qos_available": ["gpu_priv", "gpu_normal"]}}
    chosen = ResourceConfig.from_dict(
        {"gpus": 1, "cpus": 8, "qos": "gpu_normal"}, qos_available=["gpu_priv", "gpu_normal"]
    )
    assert _cluster(env, chosen)["qos"] == "gpu_normal"

    with pytest.raises(ValueError, match="qos_available"):
        ResourceConfig.from_dict({"gpus": 1, "cpus": 8, "qos": "gpu_prov"},
                                 qos_available=["gpu_priv", "gpu_normal"])

    with pytest.raises(ValueError, match="qos"):   # declared, so choosing is mandatory
        ResourceConfig.from_dict({"gpus": 1, "cpus": 8},
                                 qos_available=["gpu_priv", "gpu_normal"])

    # ...and the same rule read the other way. A workstation declares no QOS, so there is
    # nothing to choose: requiring one made the LOCAL environment unloadable, and accepting one
    # would put a Slurm word in the prompt of a machine that has no Slurm.
    assert ResourceConfig.from_dict({"gpus": 1, "cpus": 8}).qos is None
    with pytest.raises(ValueError, match="no QOS here"):
        ResourceConfig.from_dict({"gpus": 1, "cpus": 8, "qos": "gpu_priv"})
