# TODO

## 即将实现的算子

- [ ] Evaluator：大模型或规则打分。
- [ ] Generator：生成故事等新数据。
- [ ] Mapper：修改数据集格式和字段 schema。
- [ ] Mixer：混合多个数据源或多个算子结果。
- [ ] Clusterer：聚类并打新 tag。
- [ ] Reweighter：为样本或样本组加权。
- [ ] Refiner：用大模型修改回答或样本内容。
- [ ] Selector：随机选择、去重、长度选择和基于 evaluator 的选择。

## 仓库建设

- [x] 初始化 Python 包结构。
- [x] 建立算子目录。
- [x] 编写 README。
- [x] 加入 uv 开发环境。
- [x] 加入 CI、lint、format check、typecheck、unit test。
- [x] 加入框架级单测，防止目录和基础契约漂移。
- [x] 加入算子布局校验。
- [x] 加入三层算子抽象：统一入口、最细执行粒度、语义算子层。
- [x] 加入算子配置保存、资源配置识别和字段路径工具。
- [ ] 确定数据样本的最小 schema，例如 `id`、`text`、`answer`、`metadata`。
- [ ] 确定完整算子配置格式。
- [ ] 整理论文和方法调研到 `docs/research/`。
- [ ] 增加小型脱敏 fixture 数据。
