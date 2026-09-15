# Qwen3.8-27B PTB runtime

This isolated environment does not upgrade the frozen PTB/Curation environments.
Run: `configs/runs/ptb-bfcl-qwen38-27b-sonnet5-xhigh.json`.

- Agent host checkout: `/zssd/home/tangjiahui/2026/DataOS-local`, branch `release`.
- Remote checkout: `/data/hcgu/users/tjh/dataos_agentver`, branch `release`.
- Model: `/data2/hcgu/tjh_models/Qwen3.8-27B`.
- Official model revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Venv: `/data2/hcgu/dataos_environments/PTB_QWEN38_H100_slurm/.venv`.
- Preparation logs: `/data2/hcgu/dataos_diagnostics/qwen38-27b-20260915`.
- Claude Code: `claude-sonnet-5`, `xhigh` effort.
- BFCL: all 100 exec_simple examples, unchanged task questions and scorer.
- Resources: gpu01, gpu_priv (the cluster's privileged QoS), 8 H100s, 64 CPUs.
- Run budget: 24 hours, excluding download and environment preparation.

The model retains `model_type=qwen3_5`, but needs LLaMA-Factory's **qwen3_8**
template, including reasoning instructions and XML tool calls. The evaluation template is
the official model revision's `chat_template.jinja`, vendored as `qwen3_8.jinja`.
Its default thinking mode is retained; this is distinct from the Claude agent's effort.
vLLM uses qwen3_coder tool parsing, qwen3 reasoning parsing, a 32K context, and the
allocated GPU count as tensor parallelism. Training and evaluation must retain these
same model-specific conventions before/after adaptation.

Startup uses a 30-minute readiness allowance, eager execution and Triton GDN prefill.
Job 42829 exposed a cold FlashInfer compilation timeout, compiler temporary-storage
exhaustion, and orphaned workers retaining GPU memory across retries. Each server now
owns a private process group; retry/completion cleanup terminates that whole group.
vLLM and FlashInfer caches are placed under the evaluation output directory on run storage.
These are serving changes; BFCL questions, generation limits, and scoring are unchanged.
Triton's transient launcher cache is node-local under the Slurm job's TMPDIR: sharing
this cache across eight workers on the network filesystem caused stale file handles.
The server PATH includes its venv bin directory so installed JIT helpers such as ninja
remain available without shell activation. Active eval handles cannot be overwritten
by another submission; interrupted reconnaissance can resume its saved CLI session.

LLaMA-Factory is pinned to upstream commit `97b32d3133b501432141a82949d5c7bc4d94f23a`.
Its source archive is staged under the preparation directory referenced by requirements.txt.
Install requirements into the isolated venv, then use `scripts/sync_envs.py` to install
the declared DataOS tools. Capture the resolved dependency versions after installation.
The gpu01 driver was verified as 595.58.03 through Slurm job 42802.

This experiment adapts an already post-trained 27B model with a larger compute budget;
it is not directly comparable to PTB's original small-base-model, 1-H100/10-hour setting.
