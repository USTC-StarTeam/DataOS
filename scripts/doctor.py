#!/usr/bin/env python
"""DataOS 环境自检 —— 在驱动实验前先跑。

它能抓住第一跑常踩的坑:cwd 里的 ./dataos 盖掉了安装的包、缺核心依赖或 train/eval extra、
llamafactory-cli 不在 PATH、没有可见 GPU。

    python scripts/doctor.py        # 用安装了 dataos 的那个 python
"""

from __future__ import annotations

import importlib.util as iu
import os
import re
import shutil


def _pyproject_version() -> str | None:
    """The version pinned in the repo's pyproject.toml."""
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as handle:
            for line in handle:
                matched = re.match(r'version\s*=\s*"([^"]+)"', line.strip())
                if matched:
                    return matched.group(1)
    except OSError:
        return None
    return None


def has(mod: str) -> bool:
    try:
        return iu.find_spec(mod) is not None
    except Exception:
        return False


def main() -> int:
    ok = True
    print("# DataOS 自检\n")

    try:
        import dataos
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] import dataos 失败: {exc}")
        print("       请用安装了 dataos 的环境,例如 .venv/bin/python scripts/doctor.py")
        return 1
    print(f"[ok]   import dataos -> {dataos.__file__}")

    cwd_pkg = os.path.join(os.getcwd(), "src", "dataos", "__init__.py")
    if os.path.exists(cwd_pkg) and os.path.realpath(cwd_pkg) != os.path.realpath(dataos.__file__):
        print(f"[WARN] 你的 cwd 里有另一个 ./dataos ({cwd_pkg}),可能盖掉安装的包。")
        print("       本次加载到的是对的树,但从这里直接 `python -c` 可能就不是了 ——")
        print("       为稳妥起见,脚本从干净目录(如 /tmp)里跑。")

    # 版本漂移检查:抓"树被部分覆盖"(误 git clone / 旧 rsync 留下混合树)——真实事故:
    # 远端被 clone 打回 0.1.0,新旧文件混在一起,import 看似成功、API 却缺。
    version = getattr(dataos, "__version__", None)
    expected = _pyproject_version()
    if expected and version != expected:
        ok = False
        print(f"[FAIL] 版本漂移: import 的 dataos = {version},仓库 pyproject = {expected}")
        print("       树被覆盖/混合过——从开发机 `rsync -azI`(忽略时间戳)全量重推,删 __pycache__ 后重跑。")
    else:
        note = "(= pyproject)" if expected else "(未找到 pyproject,跳过比对)"
        print(f"[ok]   dataos 版本: {version} {note}")

    for mod in ("datasets", "pyarrow", "dataos_operator"):
        present = has(mod)
        ok = ok and present
        print(f"[{'ok' if present else 'FAIL'}] 核心依赖: {mod}{'' if present else '  <-- 必需'}")
    try:
        import dataos_operator  # noqa: PLC0415

        op_version = getattr(dataos_operator, "__version__", None) or "?"
        print(f"[ok]   dataos_operator 版本: {op_version} ({dataos_operator.__file__})")
    except Exception:  # noqa: BLE001 - covered by the FAIL above
        pass

    api = (
        "catalog", "load", "project",
        "Project", "Study", "Experiment",
        "Stage", "DataStage", "TrainStage", "EvalStage",
        "Artifact", "Data", "Model", "Eval",
    )
    missing = [n for n in api if not hasattr(dataos, n)]
    ok = ok and not missing
    print(f"[{'ok' if not missing else 'FAIL'}] 公开接口: {'全部就位' if not missing else f'缺 {missing}'}")

    train_ok = has("llamafactory")
    cli_ok = shutil.which("llamafactory-cli") is not None
    print(f"[{'ok' if train_ok else '--'}] train extra: llamafactory {'已装' if train_ok else '缺失(真实训练需要 [train])'}")
    if train_ok and not cli_ok:
        print("[info] 装了 llamafactory 但 `llamafactory-cli` 不在 PATH —— 真实训练前用**绝对** PATH:")
        print('       PATH="$(pwd)/.venv/bin:$PATH"(相对 .venv/bin 会让 shutil.which 返回相对路径而出错)')
    print(f"[{'ok' if has('lm_eval') else '--'}] eval extra:  lm_eval {'已装' if has('lm_eval') else '缺失(真实评估需要 [eval])'}")

    if has("torch"):
        import torch  # noqa: PLC0415
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"[{'ok' if n else '--'}] GPU: torch 看到 {n} 张 CUDA 设备")
    else:
        print("[--]   GPU: 未装 torch(只能跑 stub)")

    print()
    if ok:
        extras = "真实训练+评估就绪" if (train_ok and has("lm_eval")) else "stub 就绪(装 [train,eval] 才能真跑)"
        print(f"OK —— {extras}")
    else:
        print("上面有问题 —— 真实实验前先修掉")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
