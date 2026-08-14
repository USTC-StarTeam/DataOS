# Archived run configs — 2026-08-14

Every run config written before `qos` became a run-level choice. They were migrated in place
first (the 18 that pointed at the deleted `PTB_H100_slurm_normal` environment now carry
`"qos": "gpu_normal"`, the rest `"qos": "gpu_priv"`), so each one here still loads — but none of
them is a template. Copy `configs/runs/_template.json` for new runs.

Kept for provenance: a run's workspace name is its only link back to what it asked for, and
several of these are the configs behind results still on disk (`workspace/_archive-g3b-*`).
