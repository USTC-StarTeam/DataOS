# Selector

预留给样本选择类算子。

典型方向：

- 随机选择。
- 长度选择。
- 重复样本识别。
- 基于 evaluator 分数选择。
- 基于论文方法做主动选择或代表性选择。

当前示例：

- `length_filter`：保留文本长度位于指定区间内的 records。
- `random_sample_selector`：从输入 dataset 中随机选择固定数量的 records。
- `deduplicate_selector`：按指定字段值去重并保留首个或最后一个 record。
