"""GPU suite test (box-only): REAL lm_eval_native (gsm8k, limit 2) + fake external harness in
one suite, eval-only on a cached model. Proves the native path inside the suite combiner
(seeds, per_task, namespacing) and the macro across heterogeneous backends.

Run on the GPU box with: CUDA_VISIBLE_DEVICES=<n> + HF mirror env. Env overrides:
GPU_WS (workspace), GPU_MODEL (default Qwen/Qwen3-1.7B)."""
import json
import os
import shutil
import sys
from pathlib import Path

import dataos

WS = Path(os.environ.get("GPU_WS", "/tmp/dataos_suitegpu_ws"))
MODEL = os.environ.get("GPU_MODEL", "Qwen/Qwen3-1.7B")
shutil.rmtree(WS, ignore_errors=True)
WS.mkdir(parents=True)
harness = WS / "fake_harness.py"
harness.write_text(
    "import json,sys\njson.dump({'score':0.9,'metrics':{'bfcl::overall':0.9}}, open(sys.argv[1],'w'))\n")

proj = dataos.project(id="sg", workspace=str(WS))
proj.set_goal(instruct="gpu suite verify")
study = proj.study(id="s")
spec = {"name": "mix", "eval": {
    "model_name_or_path": MODEL,
    "suite": [
        {"name": "lm", "mode": "lm_eval_native", "model": "hf", "tasks": ["gsm8k"],
         "limit": 2, "num_fewshot": 0, "apply_chat_template": True,
         "headline_metric": "exact_match,strict-match",
         "gen_kwargs": "max_gen_toks=256,max_new_tokens=256"},
        {"name": "bfcl", "provider": "external",
         "command": f"{sys.executable} {harness} {{results_file}}",
         "harness": {"repo": "aisa-group/PostTrainBench", "commit": "deadbeef"}},
    ]}}
study.set_plan({"goal": "x", "experiments": [spec]})
e = study.experiment(spec)
e.eval.run()
art = e.eval.artifact
print("per_task =", art.meta["per_task"])
print("macro    =", art.meta["macro_average"], "| score =", art.score)
print("seeds    =", art.meta.get("seeds"))
m = json.loads(e.workspace.manifest_path().read_text())
print("manifest: backend =", m["eval"]["backend"], "| macro =", m["eval"]["macro_average"],
      "| harness commit =", m["eval"]["suite"][1]["harness"]["commit"],
      "| visible_devices =", m["hardware"]["visible_devices"])
assert "gsm8k" in art.meta["per_task"] and art.meta["per_task"]["bfcl"] == 0.9
assert any(k.startswith("gsm8k::") for k in art.metrics)
assert art.meta["seeds"] and art.meta["seeds"].get("numpy_random_seed") == 1234
assert m["eval"]["suite"][1]["harness"]["commit"] == "deadbeef"
assert m["eval"]["macro_average"] == art.meta["macro_average"]
print("SUITE_GPU_OK")
