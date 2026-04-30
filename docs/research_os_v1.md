# DataOS Research OS v1

DataOS Research OS is the project-level control plane above the existing single-run
`dataos.runtime.run_task()` path. It lets humans, Claude Code skills, and DataOS Agent submit
research tasks into one shared local state store.

## State Model

- SQLite state lives at `<workspace>/dataos.db`.
- Existing artifacts remain unchanged: `dataset`, `model`, `eval`, `result`, `experiment`.
- Project-level state uses Research OS records instead of new artifact types:
  - `project`
  - `hypothesis`
  - `research_task`
  - `candidate_operator`
  - `agent_task`
  - `agent_proposal`
  - `event`

## Minimal CLI Flow

Create a project:

```bash
dataos project create \
  --workspace workspace \
  --goal "Improve GSM8K SFT with data-centric experiments" \
  --dataset-path /path/to/hf_dataset \
  --task-template dataos/templates/sft_v1.json \
  --budget '{"max_runs": 4}'
```

`max_runs` / `max_experiments` caps non-cancelled `experiment_run` tasks. Cancelled runs release
budget; completed, failed, pending, scheduled, and running runs all count.
`constraints.max_concurrent_tasks` caps scheduled/running tasks for one project, so one campaign
does not occupy the whole workspace queue.

Submit a stub experiment task:

```bash
dataos task submit \
  --workspace workspace \
  --project-id "$PROJECT_ID" \
  --type experiment_run \
  --spec '{"stub_result": {"score": 0.5}}'
```

Run ready tasks in one scheduler + worker step:

```bash
dataos scheduler run-ready --workspace workspace --max-workers 4
```

Each scheduler tick first reconciles dependency state. Pending tasks whose dependencies are
`failed`, `cancelled`, or already `blocked` become `blocked` instead of remaining silently pending.
If an upstream failure is requeued or otherwise repaired, blocked downstream tasks whose blocking
dependencies cleared are moved back to `pending`.

Limit scheduler work to one project when multiple campaigns share a workspace:

```bash
dataos scheduler run-ready --workspace workspace --project-id "$PROJECT_ID"
```

Inspect active or released resource leases:

```bash
dataos resource leases --workspace workspace --project-id "$PROJECT_ID" --status active
```

Advance the project-level control loop in one step:

```bash
dataos project step --workspace workspace "$PROJECT_ID" \
  --agent-workers 4 \
  --task-workers 4
```

Inspect project state and audit events:

```bash
dataos project state --workspace workspace "$PROJECT_ID"
dataos event list --workspace workspace --project-id "$PROJECT_ID"
```

`dataos project summary` includes `task_status_counts` and `agent_task_status_counts`, which are
the quickest way to inspect pending, scheduled, running, blocked, failed, and completed work.

Projects start as `created` and automatically move to `running` when a research task is submitted
or DataOS Agent creates agent tasks. The transition is recorded as a `project_running` event.

Pause or resume a campaign. Paused, completed, and cancelled projects are not scheduled and do not
receive new DataOS Agent tasks:

```bash
dataos project set-status --workspace workspace "$PROJECT_ID" --status paused
dataos project set-status --workspace workspace "$PROJECT_ID" --status running
```

Pausing a project releases leases for `scheduled` tasks that have not started yet and requeues
them to `pending`, so paused work does not keep GPUs reserved. Workers also re-check project
activity before starting any remaining `scheduled` task.

Setting a project to `cancelled` cancels open research tasks, releases leases, and cancels open
agent tasks. Setting a project to `completed` is allowed only after open research and agent work is
closed. `completed` and `cancelled` are terminal states; they cannot be moved back to `running`
through ordinary status updates. Report proposals that update the project summary use the same
lifecycle guard.

Cancel queued or scheduled work and release any active lease:

```bash
dataos task cancel --workspace workspace "$TASK_ID" --reason "Superseded by a better run."
```

Recover stuck work by requeueing a single task or scanning for stale scheduled/running tasks:

```bash
dataos task requeue --workspace workspace "$TASK_ID" --reason "Worker did not start."
dataos scheduler recover-stale --workspace workspace --project-id "$PROJECT_ID" \
  --older-than-seconds 3600
```

Requeueing and scheduling both require an active project, so task-level commands cannot restart
paused, completed, or cancelled campaigns. Terminal research tasks (`completed`, `failed`, and
`cancelled`) cannot be moved back to active states through raw status updates; retry failed or
cancelled work through `task requeue` so budget, leases, and project activity checks still apply.

## DataOS Agent Flow

The manager creates expert agent tasks. Expert tasks produce proposals only; proposals become
official Research OS tasks only after manager or human acceptance.
Accepting proposals requires an active project (`created` or `running`). Rejecting proposals is
allowed on inactive projects so humans can clean up stale recommendations. A proposal must be
attached to the same project and role as the agent task that produced it, and accepted/rejected/
superseded proposals are terminal audit records.

```bash
dataos agent manager-tick --workspace workspace "$PROJECT_ID"
dataos agent run-pending --workspace workspace --project-id "$PROJECT_ID" --max-workers 4
dataos agent proposal-list --workspace workspace --project-id "$PROJECT_ID" --status proposed
dataos agent proposal-accept --workspace workspace "$PROPOSAL_ID"
```

For the common Manager -> parallel specialists -> Manager merge loop:

```bash
dataos agent cycle --workspace workspace --project-id "$PROJECT_ID" --max-workers 4
```

If a specialist task failed and should be retried after fixing credentials or prompts:

```bash
dataos agent run-pending --workspace workspace --project-id "$PROJECT_ID" --retry-failed
```

Commands that run agent LLM calls require:

- `STARTEAM_BASE_URL`
- `STARTEAM_API_KEY`
- optional `STARTEAM_MODEL`

Manual proposal review commands do not require LLM credentials.

## Candidate Operator Gate

Candidate operators are tracked in Research OS state, not promoted into the stable operator
registry automatically. An `experiment_run` can reference a candidate operator only after the
candidate is `validated` or `promoted`.

Operator owners only need to satisfy this macro contract:

```text
CandidateOperatorSpec -> implementation/ref -> ValidationReport
```

External operator builders can write validation state back into the control plane:

```bash
dataos candidate list --workspace workspace --project-id "$PROJECT_ID" --status proposed
dataos candidate set-status --workspace workspace "$CANDIDATE_OPERATOR_ID" \
  --status validated \
  --validation-report '{"ok": true, "metric": 0.82}'
```

## Worker Scope

- Local workers can execute `stub_result` tasks and `experiment_run` tasks via `run_task()`.
- SSH-assigned tasks require `spec.ssh_command` in v1.
- GPU assignment is owned by the scheduler/resource lease layer, not by DataOS Agent or skills.
