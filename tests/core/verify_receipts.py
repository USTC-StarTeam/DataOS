"""Receipt-honesty + concurrency regression (stub, no GPU). Workspace: env B1_WS.
Covers: atomic writes, liveness guard (alive/dead pid), config drift re-run, downstream
stale cascade + artifact-dir clearing, injected-source honesty, eval-only identity +
manifest flags (eval_only / visible_devices / operator origin), redacted-env guard,
external timeout receipts, replay coverage check, coverage None-filter, stale decision
layer (ranking/result), no-laundering via direct eval.run, injected-fingerprint drift,
resume_from_checkpoint clearing exception, stage-sourced device recording.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import warnings
from pathlib import Path

import dataos
from dataos_operator import Dataset

WS = Path(os.environ.get("B1_WS", "/tmp/dataos_b1_ws"))
shutil.rmtree(WS, ignore_errors=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
proj = dataos.project(id="b1", workspace=str(WS))
proj.set_goal(instruct="batch1 verify")
DS = Dataset.from_records([{"instruction": f"q{i}", "input": "", "output": "a"} for i in range(4)],
                          metadata={"dataset_format": "alpaca"})

# (1)+(6)+(7)+(12) full stub chain + injected honesty + eval-only identity + manifest fields
s1 = proj.study(id="s1")
spec = {"name": "e1", "data": {"source": "datasets/whatever-declared", "split": "train"},
        "train": {"mode": "stub", "learning_rate": 1e-4},
        "eval": {"provider": "stub", "provider_config": {"score": 0.5}}}
base = {"name": "b0", "eval": {"provider": "stub", "provider_config": {"score": 0.3},
                               "model_name_or_path": "Qwen/Qwen3-1.7B"}}
s1.set_plan({"goal": "x", "experiments": [spec, base]})
e1 = s1.experiment(spec)
e1.data.set(DS).run(); e1.train.run(); e1.eval.run()
m1 = json.loads(e1.workspace.manifest_path().read_text())
assert m1["data"]["source"] == "<injected>", m1["data"]["source"]
assert json.loads((e1.workspace.exp_dir / "data.json").read_text())["artifact"]["meta"]["declared_source"] == "datasets/whatever-declared"
assert m1["hardware"]["visible_devices"] == "4", m1["hardware"]
assert m1["eval_only"] is False
op_origin = m1["env"].get("dataos_operator")
print("operator origin:", op_origin)
assert not (WS / "b1" / "s1" / "experiments" / "e1" / f"report.json.{os.getpid()}.tmp").exists()
eb = s1.experiment(base)
eb.eval.run()
mb = json.loads(eb.workspace.manifest_path().read_text())
assert mb["eval_only"] is True, "eval-only flag missing"
res = json.loads((Path(eb.eval.artifact.path) / "results.json").read_text())
assert res["source_model_id"].startswith("evalonly:"), res["source_model_id"]
print("CHAIN_IDENTITY_DEVICE_OK")

# (3) config drift: reopen e1 with a DIFFERENT eval config -> drift detected -> re-runs
spec_changed = {"name": "e1", "data": spec["data"], "train": spec["train"],
                "eval": {"provider": "stub", "provider_config": {"score": 0.9}}}
e1b = s1.experiment(spec_changed)
assert e1b.eval.config_drift is True, "drift not detected"
old_eval_id = e1.eval.artifact.id
e1b.eval.run()
assert e1b.eval.artifact.score == 0.9 and e1b.eval.artifact.id != old_eval_id
assert e1b.train.config_drift is False
print("CONFIG_DRIFT_OK")

# (4)+(5) downstream invalidation + dir clearing
junk = e1b.workspace.model_dir / "adapter_config.json"
junk.write_text("{}")
e1b.data.set(DS).run(force=True)
tj = json.loads((e1b.workspace.exp_dir / "train.json").read_text())
ej = json.loads((e1b.workspace.exp_dir / "eval.json").read_text())
assert tj["status"] == "stale" and ej["status"] == "stale", (tj["status"], ej["status"])
e1b.train.run()
assert not junk.exists(), "stale artifact file survived the re-run"
e1b.eval.run()
assert e1b.eval.status == "completed"
print("STALE_CASCADE_AND_CLEAR_OK")

# (2)/(9) liveness guard: persisted running + alive pid -> refuses; dead pid -> proceeds
s2 = proj.study(id="s2")
spec2 = {"name": "live", "eval": {"provider": "stub", "provider_config": {"score": 0.1},
                                  "model_name_or_path": "X"}}
s2.set_plan({"goal": "x", "experiments": [spec2]})
el = s2.experiment(spec2)
el.eval.run()
ej_path = el.workspace.exp_dir / "eval.json"
state = json.loads(ej_path.read_text())
state["status"] = "running"
state["runner"] = {"pid": os.getpid(), "hostname": socket.gethostname()}
ej_path.write_text(json.dumps(state))
el2 = s2.experiment(spec2)
err = None
try:
    el2.eval.run()
except RuntimeError as exc:
    err = str(exc)
assert err and "already running" in err, err
state["runner"] = {"pid": 99999999, "hostname": socket.gethostname()}
ej_path.write_text(json.dumps(state))
el3 = s2.experiment(spec2)
el3.eval.run()
assert el3.eval.status == "completed"
print("LIVENESS_GUARD_OK")

# (10) redacted env dropped + (9) timeout keeps redacted receipts
s3 = proj.study(id="s3")
ok_harness = WS / "envcheck.py"
ok_harness.write_text(
    "import os,sys,json\n"
    "json.dump({'score': 1.0 if os.environ.get('LEAKKEY') is None else 0.0, 'metrics': {}},"
    " open(sys.argv[1],'w'))\n")
spec3 = {"name": "envg", "eval": {"provider": "external", "model_name_or_path": "X",
         "command": f"{sys.executable} {ok_harness} {{results_file}}",
         "env": {"LEAKKEY": "<redacted>"}}}
s3.set_plan({"goal": "x", "experiments": [spec3]})
e3 = s3.experiment(spec3)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    e3.eval.run()
assert e3.eval.artifact.score == 1.0, "redacted env value was exported!"
assert any("redaction placeholder" in str(c.message) for c in caught)
spec4 = {"name": "tmo", "eval": {"provider": "external", "model_name_or_path": "X",
         "command": "echo token=hf_TIMEOUTLEAK000000000000 >&2; sleep 30", "timeout": 2}}
s3.plan["experiments"].append(spec4)
e4 = s3.experiment(spec4)
err = None
try:
    e4.eval.run()
except Exception as exc:  # noqa: BLE001
    err = str(exc)
assert err and "timed out" in err and "hf_TIMEOUTLEAK" not in err, err
stderr_log = e4.workspace.eval_dir / "external_stderr.log"
assert stderr_log.exists(), "timeout left no logs"
assert "hf_TIMEOUTLEAK" not in stderr_log.read_text() and "<redacted>" in stderr_log.read_text()
print("ENV_GUARD_AND_TIMEOUT_OK")

# (11) replay coverage check: doctored manifest claiming an extra bench -> FAIL
s4 = proj.study(id="s4")
spec5 = {"name": "cov", "eval": {"suite": [
    {"name": "a", "provider": "stub", "provider_config": {"score": 0.4}},
    {"name": "b", "provider": "stub", "provider_config": {"score": 0.6}}],
    "model_name_or_path": "X"}}
s4.set_plan({"goal": "x", "experiments": [spec5]})
e5 = s4.experiment(spec5)
e5.eval.run()
mf = e5.workspace.manifest_path()
repo = Path(dataos.__file__).resolve().parents[2]
r = subprocess.run([sys.executable, str(repo / "scripts/run_one.py"),
                    "--replay", str(mf), "--tol", "0"], capture_output=True, text=True)
assert r.returncode == 0 and "eval.coverage:    OK" in r.stdout, r.stdout[-400:]
doctored = json.loads(mf.read_text())
doctored["eval"]["per_task"]["ghost_bench"] = 0.9
mf.write_text(json.dumps(doctored))
r2 = subprocess.run([sys.executable, str(repo / "scripts/run_one.py"),
                     "--replay", str(mf), "--tol", "1.0"], capture_output=True, text=True)
assert r2.returncode == 1 and "MISMATCH" in r2.stdout, r2.stdout[-400:]
print("REPLAY_COVERAGE_OK")

# (8) coverage caution counts only SCORED benches (None = skipped = missing coverage)
spec6 = {"name": "cov2", "eval": {"suite": [
    {"name": "a", "provider": "stub", "provider_config": {"score": 0.4}},
    {"name": "b", "provider": "stub", "provider_config": {"score": 0.6}}],
    "model_name_or_path": "X"}}
s4.plan["experiments"].append(spec6)
e6 = s4.experiment(spec6)
e6.eval.run()
rep6 = json.loads((e6.workspace.exp_dir / "report.json").read_text())
rep6["eval"]["per_task"]["b"] = None
(e6.workspace.exp_dir / "report.json").write_text(json.dumps(rep6))
dig2 = s4.summary()
assert dig2["caution"] and "DIFFERENT bench sets" in dig2["caution"], dig2["caution"]
print("COVERAGE_NONE_FILTER_OK")

# ===== review-driven additions =====
# (R1) stale chain: no headline score, excluded from ranking/best; result manifest nulled
s5 = proj.study(id="s5")
sa_ = {"name": "ra", "train": {"mode": "stub", "learning_rate": 1e-4},
       "eval": {"provider": "stub", "provider_config": {"score": 0.9}}}
sb_ = {"name": "rb", "train": {"mode": "stub", "learning_rate": 2e-4},
       "eval": {"provider": "stub", "provider_config": {"score": 0.2}}}
s5.set_plan({"goal": "x", "experiments": [sa_, sb_]})
for s in (sa_, sb_):
    e = s5.experiment(s)
    e.data.set(DS).run(); e.train.run(); e.eval.run()
ea_ = s5.experiment(sa_)
DS2 = Dataset.from_records([{"instruction": f"z{i}", "input": "", "output": "b"} for i in range(6)],
                           metadata={"dataset_format": "alpaca"})
ea_.data.set(DS2).run(force=True)
rep_a = json.loads((ea_.workspace.exp_dir / "report.json").read_text())
assert rep_a["status"] == "stale" and rep_a["score"] is None, (rep_a["status"], rep_a["score"])
m_a = json.loads(ea_.workspace.manifest_path().read_text())
assert m_a["result"]["score"] is None and m_a["result"]["model_id"] is None
dig5 = s5.summary()
assert dig5["best"]["name"] == "rb", dig5["best"]
print("STALE_DECISION_LAYER_OK")

# (R2) eval.run() directly on the stale chain refuses (no laundering)
err = None
try:
    ea_.eval.run()
except Exception as exc:  # noqa: BLE001
    err = str(exc)
if err is None and ea_.eval.status == "failed":
    err = ea_.eval.error
assert err and "not current" in err, err
ea_.train.run(); ea_.eval.run()
assert json.loads((ea_.workspace.exp_dir / "report.json").read_text())["status"] == "completed"
print("NO_LAUNDER_OK")

# (R3) injected-data drift: different prepared pool -> data re-runs, downstream stale
ea2 = s5.experiment(sa_)
DS3 = Dataset.from_records([{"instruction": f"w{i}", "input": "", "output": "c"} for i in range(8)],
                           metadata={"dataset_format": "alpaca"})
ea2.data.set(DS3).run()
assert json.loads((ea2.workspace.exp_dir / "data.json").read_text())["artifact"]["meta"]["rows"] == 8
assert json.loads((ea2.workspace.exp_dir / "train.json").read_text())["status"] == "stale"
print("INJECTED_DRIFT_OK")

# (R4) resume_from_checkpoint train keeps its dir; manifest carries drift flags + stage device
junk2 = ea2.workspace.model_dir / "checkpoint-100"
junk2.mkdir(parents=True, exist_ok=True)
ea2.train.config = {"mode": "stub", "resume_from_checkpoint": str(junk2)}
ea2.train.run()
assert junk2.exists(), "resume checkpoint dir was wiped"
m_a2 = json.loads(ea2.workspace.manifest_path().read_text())
assert "config_drift" in m_a2["train"] and "config_drift" in m_a2["eval"]
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
ea2._persist_report()
m_a3 = json.loads(ea2.workspace.manifest_path().read_text())
assert m_a3["hardware"]["visible_devices"] == "4", m_a3["hardware"]["visible_devices"]
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
print("CHECKPOINT_DRIFTFLAG_DEVICE_OK")
print("BATCH1_VERIFY_ALL_OK")
