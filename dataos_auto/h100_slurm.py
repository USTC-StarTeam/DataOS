"""H100 Slurm execution backend — train + generate on H100 via sbatch, driven over ssh.

The autonomous loop runs locally (mac); for ``train_executor == "slurm_h100"`` each experiment
becomes a per-(base, bench) Slurm job submitted to the hcgu H100 cluster (``gpu_priv``, 1 GPU):
``prepare_pool_data`` (fixed pool) -> ``train_sft`` (LoRA, merged to final_model) ->
``run_posttrainbench_protocol_eval`` (vLLM generation + native scoring). We poll ``sacct``,
read the eval metrics back, and write a DataOS-compatible ``report.json`` into the local
experiment dir so ``study.summary`` / best-over-history work unchanged.

Judge benches (healthbench, arenahardwriting) generate in the GPU job; their API judge runs on
the H100 login node (see :mod:`dataos_auto.judge`) — never inside Slurm.
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import time
from pathlib import Path

from dataos._ids import _make_id, _utc_now_iso
from dataos._io import atomic_write_json
from dataos._study import _slug

log = logging.getLogger(__name__)

# native metric key per bench in the protocol-eval metrics.json
SCORE_KEYS = {
    "gsm8k": "accuracy,none",
    "bfcl": "bfcl_accuracy,none",
    "gpqamain": "accuracy,none",
    "humaneval": "pass@1,create_test",
    "healthbench": "healthbench_accuracy,none",
    "aime2025": "exact_match,k32",
    "arenahardwriting": "arena_winrate,none",
}
JUDGE_BENCHES = {"healthbench", "arenahardwriting"}

_DEFAULTS = {
    "user": "hcgu",
    "host": "114.214.215.159",
    "port": "22",
    "key": str(Path.home() / ".ssh" / "ghc_miyao"),
    "runs_base": "/data/hcgu/users/tjh/dataos_agentver_runs",
    "agentver": "/data/hcgu/users/tjh/dataos_agentver",
    "template": "/data/hcgu/users/tjh/dataos_agentver/run-dataos/harnesses/posttrainbench/ptb_job.sbatch.tmpl",
    "bailian_keys": "/data/hcgu/users/tjh/.bailian_keys",
    "time_limit": "08:00:00",
    # login-node Bailian judge (healthbench / arenahardwriting) — see dataos_auto.judge
    "judge_model": "qwen3.5-plus",
    "judge_workers": "16",
    "judge_conda_env": "/data/hcgu/.conda/envs/lm-eval",
    "bailian_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}


def _conn(cfg) -> dict:
    d = dict(_DEFAULTS)
    d.update({k: v for k, v in (getattr(cfg, "slurm", {}) or {}).items() if k in d})
    return d


def _ssh_base(conn: list[str] | dict) -> list[str]:
    c = conn
    return [
        "ssh", "-i", c["key"], "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=25", "-o", "ServerAliveInterval=10",
        "-p", str(c["port"]), f"{c['user']}@{c['host']}",
    ]


def _ssh(conn: dict, cmd: str, timeout: int = 90, retries: int = 4) -> tuple[int, str, str]:
    """Run a command over ssh, retrying ssh-LAYER failures (exit 255 / timeouts) with backoff.

    A flaky campus network must not abort a 10h autonomous run on a single transient blip.
    Only ssh's own connection errors (rc 255) and client timeouts are retried; a non-255 exit
    is the REMOTE command's real return code and is passed through unretried."""
    last: tuple[int, str, str] = (255, "", "ssh not attempted")
    for attempt in range(retries):
        try:
            p = subprocess.run(
                _ssh_base(conn) + [cmd], capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            last = (255, "", f"ssh client timed out after {timeout}s")
        else:
            if p.returncode != 255:  # remote command's own rc — done (success or real failure)
                return p.returncode, p.stdout, p.stderr
            last = (255, p.stdout, p.stderr)
        if attempt < retries - 1:
            backoff = min(2 ** attempt * 2, 30)
            log.warning("ssh transient failure (attempt %d/%d): %s; retry in %ds",
                        attempt + 1, retries, last[2].strip()[:120], backoff)
            time.sleep(backoff)
    return last


def _ssh_write(conn: dict, remote_path: str, content: str) -> None:
    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    cmd = f"mkdir -p $(dirname {remote_path!r}) && echo {b64} | base64 -d > {remote_path!r}"
    rc, _, err = _ssh(conn, cmd)
    if rc != 0:
        raise RuntimeError(f"ssh write {remote_path} failed: {err.strip()}")


def _render(template: str, mapping: dict[str, str]) -> str:
    out = template
    for k, v in mapping.items():
        out = out.replace(f"@{k}@", str(v))
    return out


def _train_args(spec_train: dict) -> dict[str, str]:
    t = spec_train or {}
    return {
        "MAX_LEN": t.get("cutoff_len", t.get("max_len", 1024)),
        "BS": t.get("per_device_train_batch_size", t.get("bs", 4)),
        "GRAD_ACCUM": t.get("gradient_accumulation_steps", t.get("grad_accum", 8)),
        "LR": t.get("learning_rate", t.get("lr", 1.5e-4)),
        "EPOCHS": t.get("num_train_epochs", t.get("epochs", 1.0)),
        "LORA_R": t.get("lora_rank", t.get("lora_r", 64)),
        "LORA_ALPHA": t.get("lora_alpha", 128),
    }


def _read_score(conn: dict, eval_dir: str, bench: str) -> tuple[float | None, dict]:
    """Read the canonical PostTrainBench score (native percentage scale) for this bench.

    The protocol-eval writes ``summary.json`` with the headline ``score`` (e.g. 100.0) plus a
    nested ``metrics`` block; that is the authoritative value. Fall back to metrics.json."""
    rc, out, _ = _ssh(conn, f"cat {eval_dir}/summary.json 2>/dev/null")
    if rc == 0 and out.strip():
        try:
            summ = json.loads(out)
            s = summ.get("score")
            if isinstance(s, (int, float)):
                return float(s), (summ.get("metrics") or {})
        except json.JSONDecodeError:
            pass
    rc, out, _ = _ssh(conn, f"cat {eval_dir}/metrics.json 2>/dev/null")
    if rc != 0 or not out.strip():
        return None, {}
    try:
        metrics = json.loads(out)
    except json.JSONDecodeError:
        return None, {}
    # nested results[posttrainbench_<bench>][SCORE_KEY] (fraction) -> percentage
    results = (metrics.get("results") or {}).get(f"posttrainbench_{bench}", {})
    key = SCORE_KEYS.get(bench)
    if key and isinstance(results.get(key), (int, float)):
        v = float(results[key])
        return (v * 100.0 if v <= 1.0 else v), metrics
    for k in ("accuracy", "score", "headline_score"):
        if isinstance(metrics.get(k), (int, float)):
            v = float(metrics[k])
            return (v * 100.0 if v <= 1.0 else v), metrics
    return None, metrics


def _sacct_state(conn: dict, job_id: str) -> str:
    rc, out, _ = _ssh(conn, f"sacct -j {job_id} --format=State --noheader 2>/dev/null | head -1")
    return out.strip().split()[0] if out.strip() else "PENDING"


def _terminal(state: str) -> bool:
    return state.startswith(("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF"))


def fan_out(*, cfg, study_id: str, specs: list) -> dict[str, dict]:
    """Submit each spec as an H100 Slurm job, poll to terminal, write local report.json."""
    conn = _conn(cfg)
    bench = cfg.target_bench
    if not bench:
        raise ValueError("slurm_h100 needs cfg.target_bench (the single PTB bench)")
    base_model = cfg.h100_base_model
    if not base_model:
        raise ValueError("slurm_h100 needs cfg.h100_base_model (H100 base-model path)")
    eval_limit = cfg.data_limit if cfg.data_limit is not None else (
        20 if cfg.ptb_profile == "smoke" else 200
    )
    template = Path(__file__).resolve().parent.parent / (
        "run-dataos/harnesses/posttrainbench/ptb_job.sbatch.tmpl"
    )
    tmpl = template.read_text(encoding="utf-8")

    log.info("fan_out: %d spec(s) -> H100 %s@%s qos=gpu_priv bench=%s base=%s eval_limit=%s",
             len(specs), conn["user"], conn["host"], bench, base_model, eval_limit)
    exp_root = Path(cfg.workspace) / cfg.project_id / study_id / "experiments"
    jobs: dict[str, dict] = {}  # name -> {job_id, run_root, slug, spec}
    for spec in specs:
        name = spec["name"] if isinstance(spec, dict) else spec.name
        slug = _slug(str(name))
        run_root = f"{conn['runs_base']}/{cfg.project_id}/{study_id}/{slug}"
        data = (spec.get("data") if isinstance(spec, dict) else {}) or {}
        # smoke = FAST validation: cap training data too (the profile already caps eval_limit).
        # Without this a "smoke" run still trains on the planner's full n_samples (e.g. 120k → hours).
        n_samples = int(data.get("n_samples", 60000))
        if cfg.ptb_profile == "smoke":
            n_samples = min(n_samples, 1000)
        mapping = {
            "BENCH": bench,
            "BENCHMARK_TAG": data.get("benchmark_tag", bench),
            "SOURCES": ",".join(data.get("sources", []) or []),
            "MIX": ",".join(f"{k}:{v}" for k, v in (data.get("mix") or {}).items()),
            "N_SAMPLES": n_samples,
            "RUN_ROOT": run_root,
            "BASE_MODEL": base_model,
            "TIME_LIMIT": conn["time_limit"],
            "EVAL_LIMIT": eval_limit,
            **_train_args(spec.get("train") if isinstance(spec, dict) else {}),
        }
        # A submit failure for ONE spec (ssh down after retries, sbatch rejected) is recorded as a
        # failed experiment and the loop continues — never crash the whole autonomous run.
        try:
            _ssh_write(conn, f"{run_root}/job.sbatch", _render(tmpl, mapping))
            rc, out, err = _ssh(conn, f"cd {run_root} && sbatch --parsable job.sbatch")
        except Exception as e:  # noqa: BLE001 — ssh write exhausted retries, etc.
            log.error("submit ERROR for %s: %s (run_root=%s)", name, e, run_root)
            jobs[name] = {"job_id": None, "run_root": run_root, "slug": slug, "spec": spec,
                          "error": str(e)}
            continue
        if rc != 0:
            log.error("submit FAILED for %s: %s (run_root=%s)", name, err.strip(), run_root)
            jobs[name] = {"job_id": None, "run_root": run_root, "slug": slug, "spec": spec,
                          "error": err.strip()}
            continue
        job_id = out.strip().splitlines()[-1]
        log.info("submitted %s -> job %s (run_root=%s)", name, job_id, run_root)
        jobs[name] = {"job_id": job_id, "run_root": run_root, "slug": slug, "spec": spec,
                      "n_samples": n_samples}

    # poll all to terminal (log only on state change to keep the log readable)
    pending = {n for n, j in jobs.items() if j.get("job_id")}
    last_state: dict[str, str] = {}
    while pending:
        time.sleep(cfg.poll_interval_s)
        for name in list(pending):
            st = _sacct_state(conn, jobs[name]["job_id"])
            jobs[name]["state"] = st
            if last_state.get(name) != st:
                log.info("job %s (%s): %s", jobs[name]["job_id"], name, st)
                last_state[name] = st
            if _terminal(st):
                pending.discard(name)

    from dataos_auto import judge  # lazy: judge imports _ssh from this module

    reports: dict[str, dict] = {}
    for name, j in jobs.items():
        slug = j["slug"]
        eval_dir = f"{j['run_root']}/eval/{bench}_limit{eval_limit}"
        score, metrics = (None, {})
        judge_info = None
        if j.get("job_id"):
            score, metrics = _read_score(conn, eval_dir, bench)
            log.info("%s: state=%s native_score=%s eval_dir=%s",
                     name, j.get("state"), score, eval_dir)
            if bench in JUDGE_BENCHES:
                # compute nodes have no internet, so the GPU job only wrote a 0.0 placeholder +
                # the generated predictions; the real score is the login-node Bailian judge.
                log.info("%s: running login judge (bench=%s, model=%s)...", name, bench,
                         conn["judge_model"])
                jscore, jmetrics = judge.run_login_judge(
                    conn=conn, eval_dir=eval_dir, bench=bench, model_alias=f"ptb_{bench}",
                    judge_model=conn["judge_model"], judge_workers=int(conn["judge_workers"]),
                    limit=eval_limit,
                )
                judge_info = {"score": jscore, "metrics": jmetrics}
                if jscore is None:
                    log.error("%s: login judge produced no score (%s)", name, jmetrics.get("error"))
                else:
                    log.info("%s: login judge score=%s", name, jscore)
                score, metrics = (jscore, jmetrics) if jscore is not None else (None, metrics)
        else:
            log.error("%s: no job submitted (%s)", name, j.get("error"))
        status = "completed" if score is not None else "failed"
        log.info("%s: report status=%s score=%s", name, status, score)
        report = {
            "id": _make_id("exp"),
            "name": name,
            "status": status,
            "score": score,
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "spec": j["spec"],
            # Study.summary() gates a row's score on stages.eval.status == "completed", and reads
            # the data-volume sweep dimension from dataset.rows — populate both so the reviewer's
            # digest (best / ranking / varied_keys) reflects these remote H100 jobs.
            "stages": {"train": {"status": status}, "eval": {"status": status}},
            "dataset": {"rows": j.get("n_samples")},
            "model": {"path": f"{j['run_root']}/final_model", "base_model": base_model,
                      "mode": "h100_slurm"},
            "eval": {"score": score, "per_task": {bench: score}, "metrics": metrics,
                     "path": eval_dir},
            "h100": {"job_id": j.get("job_id"), "state": j.get("state"),
                     "run_root": j["run_root"], "needs_login_judge": bench in JUDGE_BENCHES,
                     "judge": judge_info, "error": j.get("error")},
        }
        atomic_write_json(exp_root / slug / "report.json", report)
        reports[name] = report
    return reports
