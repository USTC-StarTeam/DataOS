"""Eval suite / external / guards regression (stub, no GPU). Workspace: env SUITE_WS.

Covers: suite combiner (macro/union/receipts/entry dirs), entry isolation (+all-fail raise),
protocol-consistency caution (incl. nested-stub no-false-alarm), data-side varied keys,
injected dataset format, empty-suite validator, duplicate/path-hostile entry names,
harness-stderr redaction, coverage caution, eval-only replay.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import dataos
from dataos_operator import Dataset

WS = Path(os.environ.get("SUITE_WS", "/tmp/dataos_suite_ws"))
shutil.rmtree(WS, ignore_errors=True)
WS.mkdir(parents=True)
proj = dataos.project(id="sv", workspace=str(WS))
proj.set_goal(instruct="suite verify")

harness = WS / "fake_harness.py"
harness.write_text(
    "import json,sys\n"
    "json.dump({'score':0.8,'metrics':{'bfcl::overall':0.8,'bfcl::ast':0.7}}, open(sys.argv[1],'w'))\n")

# (1) suite: stub entry (0.4) + external entry (0.8) -> macro 0.6, eval-only baseline model
SUITE = [
    {"name": "lm", "provider": "stub", "provider_config": {"score": 0.4}},
    {"name": "bfcl", "provider": "external",
     "command": f"{sys.executable} {harness} {{results_file}}",
     "harness": {"repo": "aisa-group/PostTrainBench", "commit": "abc123", "judge_model": None}},
]
s1 = proj.study(id="s1")
spec_a = {"name": "a", "eval": {"suite": SUITE, "model_name_or_path": "Qwen/Qwen3-1.7B"}}
s1.set_plan({"goal": "x", "experiments": [spec_a]})
ea = s1.experiment(spec_a)
ea.eval.run()
art = ea.eval.artifact
print("suite score =", art.score, "| per_task =", art.meta["per_task"],
      "| metrics =", sorted(art.metrics))
assert abs(art.score - 0.6) < 1e-9, art.score
assert art.meta["per_task"] == {"lm": 0.4, "bfcl": 0.8}
assert art.metrics.get("bfcl::overall") == 0.8
assert len(art.meta["suite"]) == 2 and art.meta["suite"][1]["harness"]["commit"] == "abc123"
assert (Path(art.path) / "bfcl" / "external_results.json").is_file(), "entry subdir missing"
rep = json.loads((ea.workspace.exp_dir / "report.json").read_text())
m = json.loads(ea.workspace.manifest_path().read_text())
assert abs(rep["score"] - 0.6) < 1e-9 and abs(rep["eval"]["macro_average"] - 0.6) < 1e-9
assert m["eval"]["backend"] == "suite" and abs(m["eval"]["macro_average"] - 0.6) < 1e-9
assert m["eval"]["suite"][1]["harness"]["commit"] == "abc123"
print("SUITE_OK")

# (2) isolation: one entry broken -> skipped, other survives; all broken -> raise
spec_b = {"name": "b", "eval": {"suite": [
    {"name": "ok", "provider": "stub", "provider_config": {"score": 0.4}},
    {"name": "bad", "provider": "external", "command": "false"},
], "model_name_or_path": "X"}}
s1.plan["experiments"].append(spec_b)
eb = s1.experiment(spec_b)
eb.eval.run()
assert "bad" in eb.eval.artifact.meta["skipped_tasks"] and eb.eval.artifact.score == 0.4
spec_c = {"name": "c", "eval": {"suite": [
    {"name": "bad", "provider": "external", "command": "false"}], "model_name_or_path": "X"}}
s1.plan["experiments"].append(spec_c)
ec = s1.experiment(spec_c)
err = None
try:
    ec.eval.run()
except Exception as exc:  # noqa: BLE001
    err = str(exc)
assert err and "all suite entries failed" in err, err
print("ISOLATION_OK")

# (3) protocol caution fires for heterogeneous suites; quiet for homogeneous study
dig1 = s1.summary()
assert dig1["caution"] and "DIFFERENT eval protocols" in dig1["caution"], dig1["caution"]
s2 = proj.study(id="s2")
base = {"name": "base", "eval": {"provider": "stub", "provider_config": {"score": 0.1},
                                 "model_name_or_path": "Qwen/Qwen3-1.7B"}}
t1 = {"name": "t1", "train": {"mode": "stub", "learning_rate": 1e-4},
      "eval": {"provider": "stub", "provider_config": {"score": 0.2}}}
t2 = {"name": "t2", "train": {"mode": "stub", "learning_rate": 2e-4},
      "eval": {"provider": "stub", "provider_config": {"score": 0.3}}}
s2.set_plan({"goal": "x", "experiments": [base, t1, t2]})
ds4 = Dataset.from_records([{"instruction": f"q{i}", "input": "", "output": "a"} for i in range(4)],
                           metadata={"dataset_format": "alpaca"})
ds8 = Dataset.from_records([{"instruction": f"q{i}", "input": "", "output": "a"} for i in range(8)],
                           metadata={"dataset_format": "alpaca"})
for s, d in ((base, None), (t1, ds4), (t2, ds8)):
    e = s2.experiment(s)
    if d is None:
        e.eval.run()
    else:
        e.data.set(d).run(); e.train.run(); e.eval.run()
dig2 = s2.summary()
print("s2 caution =", dig2["caution"], "| varied =", dig2["varied_keys"])
assert not (dig2["caution"] and "protocols" in (dig2["caution"] or "")), dig2["caution"]
# (4) data-side varied
assert "data.rows" in dig2["varied_keys"] and "learning_rate" in dig2["varied_keys"]
t1_entry = next(x for x in dig2["experiments"] if x["name"] == "t1")
assert t1_entry["data.rows"] == 4, t1_entry
# (5) injected format + fingerprint in report
rep_t1 = json.loads((s2.experiment(t1).workspace.exp_dir / "report.json").read_text())
assert rep_t1["dataset"]["format"] == "alpaca", rep_t1["dataset"]
assert rep_t1["dataset"]["fingerprint"], rep_t1["dataset"]
print("PROTOCOL_AND_DATA_OK")

# (6) empty suite -> the intended ValueError, not a misleading EvalConfig TypeError
s4 = proj.study(id="s4x")
spec_e = {"name": "e", "eval": {"suite": [], "model_name_or_path": "X"}}
s4.set_plan({"goal": "x", "experiments": [spec_e]})
err = None
try:
    s4.experiment(spec_e).eval.run()
except Exception as exc:  # noqa: BLE001
    err = str(exc)
assert err and "non-empty" in err, err

# (7) suite-of-stubs with differing fake scores -> NO protocol caution
s5 = proj.study(id="s5")
sa = {"name": "sa", "eval": {"suite": [{"name": "x", "provider": "stub", "provider_config": {"score": 0.1}}],
                             "model_name_or_path": "X"}}
sb = {"name": "sb", "eval": {"suite": [{"name": "x", "provider": "stub", "provider_config": {"score": 0.9}}],
                             "model_name_or_path": "X"}}
s5.set_plan({"goal": "x", "experiments": [sa, sb]})
for s in (sa, sb):
    s5.experiment(s).eval.run()
dig5 = s5.summary()
assert dig5["caution"] is None, dig5["caution"]

# (8) duplicate / path-hostile entry names -> uniquified slugs, nothing overwritten
s6 = proj.study(id="s6")
sd = {"name": "dup", "eval": {"suite": [
    {"name": "foo", "provider": "stub", "provider_config": {"score": 0.2}},
    {"name": "foo", "provider": "stub", "provider_config": {"score": 0.4}},
    {"name": "../esc/ape", "provider": "stub", "provider_config": {"score": 0.6}},
], "model_name_or_path": "X"}}
s6.set_plan({"goal": "x", "experiments": [sd]})
ed = s6.experiment(sd)
ed.eval.run()
pt = ed.eval.artifact.meta["per_task"]
assert pt == {"foo": 0.2, "foo#2": 0.4, "esc-ape": 0.6}, pt
assert abs(ed.eval.artifact.score - 0.4) < 1e-9
assert (Path(ed.eval.artifact.path) / "foo#2").is_dir()
assert not (Path(ed.eval.artifact.path).parent.parent / "esc").exists(), "path escaped eval dir!"

# (9) harness leaking a token to stderr then failing -> skipped + stderr log REDACTED;
#     coverage caution fires when bench sets differ
FAKE2 = "hf_LEAKLEAKLEAKLEAKLEAKLEAK002"
bad = WS / "bad_harness_external.py"
bad.write_text("import sys\nprint('oops token=" + FAKE2 + "', file=sys.stderr)\nsys.exit(1)\n")
s7 = proj.study(id="s7")
sl = {"name": "leak", "eval": {"suite": [
    {"name": "ok", "provider": "stub", "provider_config": {"score": 0.5}},
    {"name": "bad", "provider": "external", "command": f"{sys.executable} {bad}"},
], "model_name_or_path": "X"}}
sfull = {"name": "full", "eval": {"suite": [
    {"name": "ok", "provider": "stub", "provider_config": {"score": 0.5}},
    {"name": "bad", "provider": "external", "command": f"{sys.executable} {harness} {{results_file}}"},
], "model_name_or_path": "X"}}
s7.set_plan({"goal": "x", "experiments": [sl, sfull]})
el = s7.experiment(sl)
el.eval.run()
s7.experiment(sfull).eval.run()
assert "bad" in el.eval.artifact.meta["skipped_tasks"]
ws_hits = subprocess.run(["grep", "-rl", FAKE2, str(WS / "sv")], capture_output=True, text=True).stdout.strip()
assert ws_hits == "", f"harness stderr token leaked into: {ws_hits}"
stderr_log = (Path(el.eval.artifact.path) / "bad" / "external_stderr.log").read_text()
assert "<redacted>" in stderr_log and FAKE2 not in stderr_log
dig7 = s7.summary()
assert dig7["caution"] and "DIFFERENT bench sets" in dig7["caution"], dig7["caution"]
print("REVIEW_FIXES_OK")

# (10) eval-only baseline manifest -> --replay replays just the eval and PASSes
repo = Path(dataos.__file__).resolve().parents[1]
s8 = proj.study(id="s8")
sb2 = {"name": "b0", "eval": {"provider": "stub", "provider_config": {"score": 0.33},
                              "model_name_or_path": "Qwen/Qwen3-1.7B"}}
s8.set_plan({"goal": "x", "experiments": [sb2]})
e8 = s8.experiment(sb2)
e8.eval.run()
mf = e8.workspace.manifest_path()
r = subprocess.run([sys.executable, str(repo / "run-dataos/scripts/run_one.py"), "--replay", str(mf), "--tol", "0"],
                   capture_output=True, text=True)
print("replay exit =", r.returncode, "|", [ln for ln in r.stdout.splitlines() if "PASS" in ln or "FAIL" in ln])
assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
print("EVAL_ONLY_REPLAY_OK")
print("SUITE_VERIFY_ALL_OK")
