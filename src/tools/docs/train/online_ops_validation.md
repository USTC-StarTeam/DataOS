# 在线算子验证记录(2026-07-24 ~ 07-25,3090 集群)

模型 Qwen3-1.7B-Base + LoRA(r16/α32),同预算同 seed 对拍。训练由 dataos-train
online-ops 分支驱动,评测走 PTB 的 lm-eval 路径。判定口径:实现正确性(链路)与
方法收益(分数)分开——前者全部验证通过,后者见下表。

## 架构验证(全部真机通过)

| 项 | 证据 |
|---|---|
| reweight 挂载点 | 加权 loss 实测生效;重构后回归冒烟通过 |
| select 挂载点 | `epoch_subset_select`/`loss_history_selector`:75/100 步签名(epoch1 全量 50 + epoch2 收窄 25) |
| step 挂载点 | 回调每步触发(协议保留,observe-only) |
| row_indices 逐行索引 | `__row_index__` 注入生效;两个集成 bug(RemoveColumnsCollator 剥键、`__getitems__` 绕过)已修 |
| entry-point 发现 | `dataos_train.online_ops` 组的 Operator 适配器在真训练里被解析并驱动决策 |
| modeljob(loss/gradient) | 8 卡分片打分数千条实测;completion-only CE == HF 内建 loss(差 0.001);sparse JL 投影保范比 0.96~1.01 |

## 算子收益矩阵(分数,基线为同预算 random/base)

### gpqa(预算 1000 条,seed42,3 epoch;random = 18.8)

| 选法 | 分数 | Δ |
|---|---|---|
| LESS(池子切片 D_val) | 21.2 | +2.4(噪声内) |
| LESS(代表性 D_val) | 8.5 | -10.3 |
| loss-hi(最难) | 11.6 | -7.2 |
| loss-lo(最易) | 11.4 | -7.4 |
| reweight(linupper,4 种子) | — | +3.1(≈1.7σ,不显著) |

### humaneval(预算 926 条,seed42,3 epoch;random = 38.4)

| 选法 | 分数 | Δ |
|---|---|---|
| LESS(池子切片 D_val) | 15.2 | -23.2 |
| LESS(代表性 D_val) | 19.5 | -18.9 |
| loss-hi(最难) | 12.2 | -26.2 |
| loss-lo(最易) | 33.5 | -4.9 |
| reweight(linupper,2 种子,vs base 21.95) | 17.65 | -4.3 |

### aime(预算 1000 条,seed42;random = 3.4,整体近地板)

| 选法 | 分数 | Δ |
|---|---|---|
| LESS(池子切片 D_val) | 3.4 | 0.0 |

## 结论

1. **实现无误,方法在该规模(1.7B + LoRA + 千条池)无稳定收益。** LESS、loss
   课程、reweight 三线证据一致;没有算子在任一 bench 上显著超过随机采样。
2. **梯度选择的结果极不稳定**:gpqa 上两种 D_val 的选中集 Jaccard 0.866,分数却
   21.2 vs 8.5——~13% 的行差异摆动 12 个点,信号弱、噪声主导。
3. **loss 课程两个 bench 一致为负**:loss-hi(选难)大幅有害(-26.2 / -7.2),与
   课程学习文献一致(高 loss ≈ 噪声/学不动);loss-lo 也一致为负(-4.9 / -7.4)——
   砍掉最难的一段同样损失信息量。
4. 池子已贴合任务时,random 的多样性就是强基线;任何 top-N 收窄都在赌信号质量,
   而该规模下信号不存在。
