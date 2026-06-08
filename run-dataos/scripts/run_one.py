#!/usr/bin/env python
"""跑一个 study 里的单个实验,端到端。

重开 study,加载**已准备好的**数据集,对指定 spec 做 data(逃生口注入)+ train + eval + report。
这是串行循环和并行运行通用的"单元"—— 并行时每个进程钉一张卡:

    CUDA_VISIBLE_DEVICES=6 python scripts/run_one.py \
        --study lr-sweep --spec lr-1e-4 --data prepared/ --workspace workspace

数据配方(算子)是你的事 —— 用 dataos_operator 构建 Dataset、`save_to_disk` 后把目录作为
--data 传进来。本脚本用 `exp.data.set(ds)`(**逃生口**:注入现成 Dataset、跳过 spec 里的 recipe)
保证"准备一次、多进程复用",因此对各种配方都通用、不会按 spec 重跑数据处理。
"""

from __future__ import annotations

import argparse

import dataos


def main() -> int:
    ap = argparse.ArgumentParser(description="跑一个 DataOS study 里的单个实验。")
    ap.add_argument("--study", required=True, help="study id(已存在则重开)")
    ap.add_argument("--spec", required=True, help="study plan 里某个实验的 name")
    ap.add_argument("--data", required=True, help="已准备好的数据集: hf://repo | 文件 | save_to_disk 目录")
    ap.add_argument("--workspace", default="workspace")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    study = dataos.study(id=args.study, workspace=args.workspace)
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


if __name__ == "__main__":
    raise SystemExit(main())
