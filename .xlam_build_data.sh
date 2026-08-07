#!/bin/bash
#SBATCH --job-name=xlam-data
#SBATCH --nodes=1
#SBATCH --time=01:00:00
set -euo pipefail
RUN_ROOT=/data2/hcgu/dataos_agentver_runs/ptb-bfcl-4b2/round-0001/xlam-clean-singlecall-lora
POOL=/data2/hcgu/dataos_fixed_pools/processed/posttrainbench_fixed_pool_v2/xlam_function_calling_60k
AGENTVER=/data/hcgu/users/tjh/dataos_agentver
OUT="$RUN_ROOT/data/sft"
export DATAOS_OPERATOR_PLUGINS="$AGENTVER/resources/custom_operators:$RUN_ROOT/custom_operators"
export TMPDIR="/tmp/job_$SLURM_JOB_ID"
export TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/cache/inductor"
export TRITON_CACHE_DIR="$RUN_ROOT/cache/triton"
export TORCH_HOME="$RUN_ROOT/cache/torch"
mkdir -p "$TMPDIR" "$RUN_ROOT/cache" "$RUN_ROOT/data"
source "$AGENTVER/configs/environments/PTB_H100_slurm/envs/dataos_sft/.venv/bin/activate"
python - <<'PY'
import json
import re
from pathlib import Path

from datasets import Dataset as HFDataset
from datasets import load_from_disk
from dataos_operator import Dataset
from dataos_operator.operators.mapper.function_call_normalizer import FunctionCallNormalizer
from dataos_operator.operators.selector.deduplicate_selector import DeduplicateSelector
from dataos_operator.operators.selector.function_call_filter import FunctionCallFilter
from dataos_operator.operators.selector.random_sample_selector import RandomSampleSelector

run_root = Path("/data2/hcgu/dataos_agentver_runs/ptb-bfcl-4b2/round-0001/xlam-clean-singlecall-lora")
source = load_from_disk(
    "/data2/hcgu/dataos_fixed_pools/processed/posttrainbench_fixed_pool_v2/"
    "xlam_function_calling_60k"
)["train"]

ported = []
port_rejects = 0
for row in source:
    prefix = "Available tools:\n"
    raw_tools = row.get("input", "")
    if not isinstance(raw_tools, str) or not raw_tools.startswith(prefix):
        port_rejects += 1
        continue
    try:
        tools = json.loads(raw_tools[len(prefix):])
    except json.JSONDecodeError:
        port_rejects += 1
        continue
    ported.append(
        {
            "query": row.get("instruction", ""),
            "tools": tools,
            "answers": row.get("output", ""),
            "source_row_index": row.get("source_row_index"),
        }
    )

normalized = FunctionCallNormalizer(
    tools_key="tools", call_key="answers", query_key="query"
).run(Dataset.from_records(ported))

# Mechanical bridge: the normalizer emits tagged assistant messages while the fixed
# filter consumes a JSON list in `output`; restore `query` for the fixed dedup key.
bridged = []
bridge_rejects = 0
multi_call_rejects = 0
for row in normalized:
    messages = row.get("messages") or []
    user = next((m.get("content") for m in messages if m.get("role") == "user"), None)
    assistant = next(
        (m.get("content") for m in messages if m.get("role") == "assistant"), None
    )
    calls = []
    if isinstance(assistant, str):
        for payload in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", assistant, re.S):
            try:
                calls.append(json.loads(payload))
            except json.JSONDecodeError:
                pass
    if not isinstance(user, str) or not user.strip() or not calls:
        bridge_rejects += 1
        continue
    if len(calls) != 1:
        multi_call_rejects += 1
        continue
    call = calls[0]
    if not isinstance(call.get("arguments"), dict) or not call.get("name"):
        bridge_rejects += 1
        continue
    bridged.append(
        {
            **row,
            "query": user,
            "output": json.dumps(calls, ensure_ascii=False),
        }
    )

valid = FunctionCallFilter(output_key="output").run(Dataset.from_records(bridged))
deduped = DeduplicateSelector(
    key="query", keep="first", case_sensitive=True
).run(valid)
sampled = RandomSampleSelector(sample_size=30000, strict=False, seed=42).run(deduped)

alpaca = []
for row in sampled:
    calls = json.loads(row["output"])
    call = calls[0]
    envelope = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call["arguments"],
                },
            }
        ]
    }
    alpaca.append(
        {
            "instruction": row["query"],
            "input": "Available tools:\n" + json.dumps(row.get("tools", []), ensure_ascii=False),
            "output": json.dumps(envelope, ensure_ascii=False),
        }
    )

def parse_prediction(text):
    text = text.strip()
    candidates = [(text, False)]
    candidates += [
        (m.group(1).strip(), True)
        for m in re.finditer(
            r"```(?:tool_calls?|function_call)\s*\n(.*?)```", text, re.S | re.I
        )
    ]
    candidates += [
        (m.group(1).strip(), True)
        for m in re.finditer(r"<tool_calls?>\s*(.*?)\s*</tool_calls?>", text, re.S)
    ]
    for candidate, allow_bare in candidates:
        cleaned = re.sub(r"</?tool_calls?>", "", candidate.strip()).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        calls = payload.get("tool_calls")
        if calls:
            function = calls[0].get("function", {})
            name = function.get("name")
            arguments = function.get("arguments", {})
        elif isinstance(payload.get("tool_call"), dict):
            call = payload["tool_call"]
            function = call.get("function", {})
            name = call.get("name") or function.get("name")
            arguments = call.get("arguments") or function.get("arguments", {})
        elif allow_bare and payload.get("name") and "arguments" in payload:
            name, arguments = payload["name"], payload["arguments"]
        else:
            return None
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return {"function": name, "arguments": arguments}
    return None

precheck = [parse_prediction(row["output"]) for row in alpaca[:3]]
if len(precheck) != 3 or any(value is None for value in precheck):
    raise RuntimeError(f"BFCL format pre-check failed: {precheck}")

out = run_root / "data/sft"
if out.exists():
    import shutil
    shutil.rmtree(out)
HFDataset.from_list(alpaca).save_to_disk(str(out))
(out / "train.json").write_text(
    json.dumps(alpaca, ensure_ascii=False), encoding="utf-8"
)
(out / "dataset_info.json").write_text(
    json.dumps(
        {
            "train": {
                "file_name": "train.json",
                "columns": {"prompt": "instruction", "query": "input", "response": "output"},
            }
        },
        indent=2,
    ),
    encoding="utf-8",
)
metrics = {
    "source_rows": len(source),
    "ported_rows": len(ported),
    "port_rejects": port_rejects,
    "normalized_rows": len(normalized),
    "bridge_rejects": bridge_rejects,
    "multi_call_rejects": multi_call_rejects,
    "valid_single_call_rows": len(valid),
    "deduplicated_rows": len(deduped),
    "sampled_rows": len(alpaca),
    "format_precheck": "passed: 3/3 targets parsed by BFCL-equivalent parser",
    "target_example": alpaca[0] if alpaca else None,
}
(run_root / "data/build_metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
PY
