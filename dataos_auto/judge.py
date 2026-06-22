"""Login-node API judge for PostTrainBench judge benches (healthbench, arenahardwriting).

H100 compute nodes have NO internet, so the Slurm GPU job only GENERATES answers; the LLM
judge — Bailian (DashScope, model ``qwen3.5-plus``) — runs HERE on the H100 login node over
ssh, against the in-branch vendored ``api_judge`` + ``posttrainbench_src``.

Keys hygiene (hard requirement): the Bailian key is read from a 0600 file on H100 and exported
into the judge process env *inline on the remote shell only*. It is never echoed, never written
to any rendered manifest / report.json, and never leaves H100.

:func:`run_login_judge` converts the generated predictions to the format each vendored judge
expects, runs the judge under the ``lm-eval`` conda env (it has ``openai`` + ``shortuuid`` +
``tiktoken``), and reads back the headline accuracy on a 0-100 scale. The caller patches the
local ``report.json`` with the judged score.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Self-contained H100 connection + ssh helper (the deterministic eval/judge tool the sub-agent
# invokes — it does NOT improvise judging). cfg.slurm can override any of these.
_DEFAULTS = {
    "user": "hcgu", "host": "114.214.215.159", "port": "22",
    "key": str(Path.home() / ".ssh" / "ghc_miyao"),
    "agentver": "/data/hcgu/users/tjh/dataos_agentver",
    "bailian_keys": "/data/hcgu/users/tjh/.bailian_keys",
    "judge_conda_env": "/data/hcgu/.conda/envs/lm-eval",
    "bailian_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "judge_model": "qwen3.5-plus", "judge_workers": "16",
}


def _conn(overrides: dict | None = None) -> dict:
    d = dict(_DEFAULTS)
    if overrides:
        d.update({k: v for k, v in overrides.items() if k in d})
    return d


def _ssh_base(c: dict) -> list[str]:
    return [
        "ssh", "-i", c["key"], "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=25", "-o", "ServerAliveInterval=10",
        "-p", str(c["port"]), f"{c['user']}@{c['host']}",
    ]


def _ssh(conn: dict, cmd: str, timeout: int = 90, retries: int = 4) -> tuple[int, str, str]:
    """Run a command over ssh, retrying ssh-LAYER failures (exit 255 / timeouts) with backoff.
    A non-255 exit is the remote command's real rc and is passed through unretried."""
    last: tuple[int, str, str] = (255, "", "ssh not attempted")
    for attempt in range(retries):
        try:
            p = subprocess.run(_ssh_base(conn) + [cmd], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            last = (255, "", f"ssh client timed out after {timeout}s")
        else:
            if p.returncode != 255:
                return p.returncode, p.stdout, p.stderr
            last = (255, p.stdout, p.stderr)
        if attempt < retries - 1:
            time.sleep(min(2 ** attempt * 2, 30))
    return last

# protocol-eval gen artifact (written into the run's eval dir) per judge bench
_PRED_FILE = {
    "arenahardwriting": "arena_predictions.jsonl",
    "healthbench": "healthbench_predictions.jsonl",
}
# vendored judge entrypoint per bench
_JUDGE_SCRIPT = {
    "arenahardwriting": "arena_bailian_judge.py",
    "healthbench": "health_bailian_judge.py",
}


def _eval_paths(conn: dict) -> dict[str, str]:
    """Derive the in-branch vendored eval paths on H100 from the deployment root."""
    root = f"{conn['agentver']}/run-dataos/eval"
    return {
        "api_judge": f"{root}/api_judge",
        "ptb_src": f"{root}/posttrainbench_src",
        "benchmarks": f"{root}/benchmarks",
    }


def _score_from_metrics(metrics: dict) -> float | None:
    """Headline accuracy -> 0-100 scale (winrate for arena, rubric accuracy for health)."""
    for k in ("accuracy", "score", "winrate", "arena_winrate"):
        v = metrics.get(k)
        if isinstance(v, (int, float)):
            return float(v) * 100.0 if v <= 1.0 else float(v)
    return None


def run_login_judge(
    *,
    conn: dict,
    eval_dir: str,
    bench: str,
    model_alias: str,
    judge_model: str = "qwen3.5-plus",
    judge_workers: int = 16,
    limit: int = -1,
    timeout: int = 7200,
) -> tuple[float | None, dict]:
    """Run the Bailian judge for one judge bench on the H100 login node.

    Returns ``(score_0_100 | None, metrics)``. ``None`` means the judge could not run or score
    (missing predictions / judge failure) — the caller keeps the bench as failed.
    """
    if bench not in _PRED_FILE:
        raise ValueError(f"{bench} is not a judge bench")
    p = _eval_paths(conn)
    keys_file = conn["bailian_keys"]
    conda_sh = "/data/software/miniconda3/etc/profile.d/conda.sh"
    env = conn.get("judge_conda_env", "/data/hcgu/.conda/envs/lm-eval")

    pred = f"{eval_dir}/{_PRED_FILE[bench]}"
    judge_out = f"{eval_dir}/judge"
    answer_file = f"{judge_out}/{bench}_judge_input.jsonl"
    script = f"{p['api_judge']}/{_JUDGE_SCRIPT[bench]}"
    log.info("login judge: bench=%s alias=%s model=%s workers=%s limit=%s",
             bench, model_alias, judge_model, judge_workers, limit)

    # 0) predictions must exist (the gen step ran)
    rc, out, _ = _ssh(conn, f"test -s {shlex.quote(pred)} && wc -l < {shlex.quote(pred)} || echo MISSING")
    if rc != 0 or out.strip() == "MISSING":
        log.error("login judge: predictions not found: %s", pred)
        return None, {"error": f"predictions not found: {pred}"}
    log.info("login judge: %s prediction rows -> converting + judging", out.strip())

    base_url = conn.get("bailian_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    # one remote shell: activate env -> convert -> judge.  The key is interpolated inline from the
    # 0600 file and exported only into this process; it is never printed.
    cmd = (
        f"set -eo pipefail; "
        f"source {conda_sh}; conda activate {shlex.quote(env)}; "
        f"mkdir -p {shlex.quote(judge_out)}; "
        f"python {shlex.quote(p['api_judge'])}/predictions_to_judge_input.py "
        f"--bench {bench} --in {shlex.quote(pred)} --out {shlex.quote(answer_file)} "
        f"--model-alias {shlex.quote(model_alias)}; "
        f"OPENAI_API_KEY=\"$(sed -n '1p' {shlex.quote(keys_file)})\" "
        f"OPENAI_BASE_URL={shlex.quote(base_url)} "
        f"BAILIAN_JUDGE_MODEL={shlex.quote(judge_model)} "
        f"POSTTRAINBENCH_EVAL_SRC={shlex.quote(p['ptb_src'])} "
        f"POSTTRAINBENCH_BENCHMARKS_DIR={shlex.quote(p['benchmarks'])} "
        f"python {shlex.quote(script)} "
        f"--ptb-eval-src {shlex.quote(p['ptb_src'])} "
        f"--answer-file {shlex.quote(answer_file)} "
        f"--model-alias {shlex.quote(model_alias)} "
        f"--output-dir {shlex.quote(judge_out)} "
        f"--judge-model {shlex.quote(judge_model)} "
        f"--judge-workers {int(judge_workers)} --limit {int(limit)} "
        f"> {shlex.quote(judge_out)}/judge.log 2>&1; "
        f"cat {shlex.quote(judge_out)}/{shlex.quote(model_alias)}.metrics.json"
    )
    rc, out, err = _ssh(conn, cmd, timeout=timeout)
    if rc != 0:
        # surface a short tail of the judge log for the report (no secrets in it)
        _, tail, _ = _ssh(conn, f"tail -5 {shlex.quote(judge_out)}/judge.log 2>/dev/null")
        log.error("login judge rc=%s; judge.log tail: %s", rc, tail.strip()[-400:])
        return None, {"error": f"judge rc={rc}", "log_tail": tail.strip()[-800:]}
    try:
        metrics = json.loads(out)
    except json.JSONDecodeError:
        log.error("login judge: metrics.json unparseable; stdout tail: %s", out.strip()[-200:])
        return None, {"error": "judge metrics.json unparseable", "stdout_tail": out.strip()[-400:]}
    score = _score_from_metrics(metrics)
    log.info("login judge done: bench=%s score=%s", bench, score)
    return score, metrics


def main(argv: list[str] | None = None) -> int:
    """Deterministic CLI for the sub-agent: judge one bench's generated predictions.

    e.g. ``python -m dataos_auto.judge --eval-dir <RUN_ROOT>/eval/<bench>_limit<N> \
         --bench healthbench --model-alias ptb_healthbench``
    Prints ``{"score": <0-100|null>, "metrics": {...}}``; exit 0 iff a score was produced."""
    import argparse

    ap = argparse.ArgumentParser(description="PostTrainBench login-node Bailian judge (deterministic)")
    ap.add_argument("--eval-dir", required=True, help="dir holding <bench>_predictions.jsonl")
    ap.add_argument("--bench", required=True, choices=sorted(_PRED_FILE))
    ap.add_argument("--model-alias", required=True, help="e.g. ptb_<bench>")
    ap.add_argument("--judge-model", default=_DEFAULTS["judge_model"])
    ap.add_argument("--judge-workers", type=int, default=int(_DEFAULTS["judge_workers"]))
    ap.add_argument("--limit", type=int, default=-1)
    a = ap.parse_args(argv)
    score, metrics = run_login_judge(
        conn=_conn(), eval_dir=a.eval_dir, bench=a.bench, model_alias=a.model_alias,
        judge_model=a.judge_model, judge_workers=a.judge_workers, limit=a.limit,
    )
    print(json.dumps({"score": score, "metrics": metrics}, ensure_ascii=False))
    return 0 if score is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
