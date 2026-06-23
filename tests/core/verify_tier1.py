"""Tier-1 regression (stub, no GPU). Workspace: env T1_WS.
(A) metric_winners: stderr/_sterr/all2 excluded; exact ties crown nobody; unique winner kept;
(B) save_to_disk sidecar: dataset_format survives save->load->inject->report;
(C) train-needs-data friendly error + failed-stage persistence.
"""
import json
import os
import shutil
from pathlib import Path

import dataos
from dataos_operator import Dataset

WS = Path(os.environ.get("T1_WS", "/tmp/dataos_tier1_ws"))
shutil.rmtree(WS, ignore_errors=True)
proj = dataos.project(id="t1", workspace=str(WS))
proj.set_goal(instruct="tier1 verify")

# (A) two exps: tied real metric, unique-winner real metric, noisy stderr/all2 metrics
study = proj.study(id="w")
a = {"name": "a", "eval": {"provider": "stub", "provider_config": {"score": 0.5}, "model_name_or_path": "X"}}
b = {"name": "b", "eval": {"provider": "stub", "provider_config": {"score": 0.4}, "model_name_or_path": "X"}}
study.set_plan({"goal": "x", "experiments": [a, b]})
for s in (a, b):
    study.experiment(s).eval.run()
for name, mets in (("a", {"t::acc": 0.9, "t::tie": 0.5, "t::stderr": 0.01, "bfcl::all2": 0.02, "u::x_sterr": 0.1}),
                   ("b", {"t::acc": 0.7, "t::tie": 0.5, "t::stderr": 0.30, "bfcl::all2": 0.50, "u::x_sterr": 0.9})):
    p = WS / "t1" / "w" / "experiments" / name / "report.json"
    r = json.loads(p.read_text())
    r["eval"]["metrics"] = mets
    p.write_text(json.dumps(r))
dig = study.summary()
print("winners:", dig["metric_winners"])
assert dig["metric_winners"] == {"t::acc": "a"}, dig["metric_winners"]
print("WINNERS_FILTER_OK")

# (B) sidecar metadata round-trip
src = Dataset.from_records([{"instruction": f"q{i}", "input": "", "output": "a"} for i in range(4)],
                           metadata={"dataset_format": "alpaca"})
saved = WS / "prepared"
src.save_to_disk(str(saved))
print("sidecar exists:", (saved / "dataos_metadata.json").is_file())
ds2 = dataos.load(str(saved))
print("loaded metadata dataset_format:", dict(ds2.metadata).get("dataset_format"))
study2 = proj.study(id="fmt")
spec = {"name": "e", "train": {"mode": "stub"}, "eval": {"provider": "stub", "provider_config": {"score": 0.1}}}
study2.set_plan({"goal": "x", "experiments": [spec]})
e = study2.experiment(spec)
e.data.set(ds2).run(); e.train.run(); e.eval.run()
rep = json.loads((e.workspace.exp_dir / "report.json").read_text())
print("report dataset.format =", rep["dataset"]["format"])
assert rep["dataset"]["format"] == "alpaca", rep["dataset"]
print("SIDECAR_FORMAT_OK")

# (C) train with no data -> friendly error; failed stage persists with started_at
study3 = proj.study(id="run")
bad = {"name": "bad", "train": {"mode": "stub"}, "eval": {"provider": "stub", "provider_config": {"score": 0.1}}}
study3.set_plan({"goal": "x", "experiments": [bad]})
eb = study3.experiment(bad)
err = None
try:
    eb.train.run()
except Exception as exc:  # noqa: BLE001
    err = str(exc)
assert err and "train needs data" in err, err
tj = json.loads(eb.workspace.stage_path("train").read_text())
assert tj["status"] == "failed" and tj["started_at"], tj["status"]
print("RUNNING_PERSIST_OK")
print("TIER1_VERIFY_ALL_OK")
