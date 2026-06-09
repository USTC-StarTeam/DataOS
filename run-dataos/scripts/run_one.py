#!/usr/bin/env python
"""跑一个 project/study 里的单个实验,或从 manifest **复现**一个实验。

**正常**(并行/串行的单元,并行时每进程钉一张卡):重开 project + study,加载**已准备好的**数据集
(`exp.data.set(ds)` 逃生口,跳过 spec 的 recipe),对一个 spec 做 data + train + eval:

    CUDA_VISIBLE_DEVICES=6 python scripts/run_one.py \
        --project qwen-gsm8k --study lr-sweep --spec lr-1e-4 --data prepared/ --workspace workspace

**复现**:`--replay manifest.json` —— 在独立 workspace 按 manifest 的声明式数据(source+revision+recipe)+
train/eval config + seeds 重跑,并验证 **数据 fingerprint 一致** 且 **|Δscore| <= --tol**(stub 默认 0 = 精确)。
注:用 `set()` 逃生口注入数据(`source == "<injected>"`)的实验无法从 manifest 重跑数据。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dataos


def _run_one(args: argparse.Namespace) -> int:
    study = dataos.project(id=args.project, workspace=args.workspace).study(id=args.study)
    specs = {s.get("name"): s for s in (study.plan or {}).get("experiments", [])}
    spec = specs.get(args.spec)
    if spec is None:
        raise SystemExit(f"spec {args.spec!r} 不在 study {args.study!r} 里;现有: {sorted(specs)}")

    ds = dataos.load(args.data, split=args.split, limit=args.limit)
    exp = study.experiment(spec)
    exp.data.set(ds).run()   # 逃生口:注入已准备好的数据集,跳过 spec["data"] 的 recipe
    exp.train.run()          # config 默认取 spec["train"]
    exp.eval.run()           # config 默认取 spec["eval"]
    summary = exp.summary()
    print(
        f"[run_one] {args.spec}: status={summary['status']} "
        f"model={(summary.get('model') or {}).get('mode')} "
        f"score={(summary.get('eval') or {}).get('score')} -> {summary['id']}"
    )
    return 0


def _replay(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.replay).read_text(encoding="utf-8"))
    who = manifest.get("who") or {}
    data = manifest.get("data") or {}
    data_cfg = dict(data.get("config") or {})
    if not data_cfg.get("source") or data.get("source") == "<injected>":
        raise SystemExit(
            "replay 需要声明式数据(spec['data'] 有 source);此 manifest 用了 set() 注入,无法重跑数据。"
        )

    spec = {
        "name": who.get("name") or "replay",
        "data": data_cfg,
        "train": dict((manifest.get("train") or {}).get("config") or {}),
        "eval": dict((manifest.get("eval") or {}).get("config") or {}),
    }
    # 复现到独立 workspace,避免覆盖原实验(默认 manifest 同级 _replay/)
    workspace = (
        args.workspace
        if args.workspace != "workspace"
        else str(Path(args.replay).resolve().parent / "_replay")
    )
    study = dataos.project(id=who.get("project") or "replay", workspace=workspace).study(
        id=who.get("study") or "replay"
    )
    study.set_plan({"goal": "replay", "experiments": [spec]})
    exp = study.experiment(spec)
    exp.data.run()
    exp.train.run()
    exp.eval.run()

    new_fp = (exp.data.artifact.meta or {}).get("fingerprint")
    old_fp = data.get("fingerprint")
    new_score = exp.eval.artifact.score
    old_score = (manifest.get("result") or {}).get("score")

    fp_ok = old_fp is None or new_fp == old_fp
    if old_score is None or new_score is None:
        score_ok = old_score is None and new_score is None
        score_detail = f"old={old_score} new={new_score}"
    else:
        score_ok = abs(float(new_score) - float(old_score)) <= args.tol
        score_detail = f"|{new_score} - {old_score}| <= {args.tol}"

    ok = fp_ok and score_ok
    print(f"[replay] data.fingerprint: {'OK' if fp_ok else 'MISMATCH'} (old={old_fp} new={new_fp})")
    print(f"[replay] eval.score:       {'OK' if score_ok else 'FAIL'} ({score_detail})")
    print(f"[replay] {'PASS' if ok else 'FAIL'} -> {workspace}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="跑一个 DataOS project/study 里的单个实验,或从 manifest 复现一个实验。"
    )
    ap.add_argument("--replay", help="manifest.json 路径:从它复现并验证(忽略 --project/--study/--spec/--data)")
    ap.add_argument("--tol", type=float, default=0.0, help="--replay 时 score 容差(stub 默认 0 = 精确)")
    ap.add_argument("--project", help="project id(已存在则重开)")
    ap.add_argument("--study", help="study id(已存在则重开)")
    ap.add_argument("--spec", help="study plan 里某个实验的 name")
    ap.add_argument("--data", help="已准备好的数据集: hf://repo | 文件 | save_to_disk 目录")
    ap.add_argument("--workspace", default="workspace")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.replay:
        return _replay(args)
    missing = [f"--{f}" for f in ("project", "study", "spec", "data") if not getattr(args, f)]
    if missing:
        raise SystemExit(f"缺参数: {missing}(或用 --replay manifest.json 复现)")
    return _run_one(args)


if __name__ == "__main__":
    raise SystemExit(main())
