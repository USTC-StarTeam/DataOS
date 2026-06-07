# 执行 —— 运行模式与并行

串行还是并行,是**你**的事,不归 DataOS。study 就是个目录,`study.compare()` 读每个实验盘上的
`report.json`,所以无论你怎么驱动,SDK 都一样。挑合适的模式。

## 光谱

```
一个大脚本(全内存)     ↔     一实验一进程      ↔     一阶段一调用
   批处理、最快                自适应 / 并行甜点区          最灵活
```

状态在调用之间靠两种方式传:

- **同进程(一个脚本):** 内存里的 `study` / `exp` / `model` 对象顺序流转。最简单 —— 适合计划
  已固定的情形。
- **跨进程(分开跑):** 文件系统是 join。`dataos.study(id=...)` 是 **open-or-create** —— 新进程
  重开 study、把 `plan.json` 读回;结果在 `report.json` 里。这就是"中途停下来判断"和并行能成立的
  原因。没有 DataOS session 或守护进程 —— 状态要么在内存、要么在磁盘。

## 串行(一个进程)

```python
study = dataos.study(id="lr-sweep"); study.set_plan(plan)
for spec in study.plan["experiments"]:
    exp = study.experiment(spec)
    ds = build_dataset(spec)        # 你的配方(见 data-prep.md)
    exp.train(ds); exp.eval(); exp.finalize()
study.compare(); study.finalize(conclusion="…")
```

## 并行(一实验一进程,各钉一张卡)

自然单位是**一实验一进程**,因为 `train -> eval -> finalize` 共享内存里的 `exp.model` —— 把
train 和 eval 拆到不同进程就得从盘上重建 `Model`。把共享数据集准备并保存一次,然后用自带的
`scripts/run_one.py`:

```bash
python -c "import dataos; dataos.study(id='lr-sweep').set_plan(__import__('json').load(open('plan.json')))"
# …… 用算子构建数据集,然后 ds.save_to_disk('prepared/') ……
CUDA_VISIBLE_DEVICES=6 python scripts/run_one.py --study lr-sweep --spec lr-1e-4 --data prepared/ &
CUDA_VISIBLE_DEVICES=5 python scripts/run_one.py --study lr-sweep --spec lr-5e-5 --data prepared/ &
wait
python -c "import dataos, json; print(json.dumps(dataos.study('lr-sweep').compare(), ensure_ascii=False, indent=2))"
```

DataOS 既不调度、也不知道并行 —— 分卡和并发度完全是你的事。`compare()` 只是 glob 现有的
`report.json`,所以容忍还在跑或缺失的实验。

## 先跑 doctor

真实运行前,`python scripts/doctor.py` 确认你 import 的是对的 `dataos`(cwd 里的 `./dataos`
会盖掉安装的包)、核心依赖 + `train` / `eval` extra 在不在、`llamafactory-cli` 在不在 PATH、
GPU 可不可见。它能抓住每次第一跑都会咬人的那些坑。
