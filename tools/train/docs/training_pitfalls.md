# Training pitfalls — verified on 3090 × Qwen3-1.7B-Base (LoRA SFT)

这些坑全部在 3090 对拍中实测复现并定位，按根因记录。触发面：任何 backend=llamafactory 的
SFT 任务，尤其小基座 + LoRA + 长文本数据。

## 1. 截断教模型不终止(cutoff_len < 答案长度)

- 现象:训练 loss 正常,但生成永不停止、评测卡死、输出重复乱码。
- 根因:`cutoff_len` 小于样本长度时,标签在答案中间被切断,`<|im_end|>` 从未出现在标签
  末尾——模型学到"答案不需要终止"。实测:aime 数据 66% 超 512 token,模型全员退化为
  无限生成。
- 修法:**数据按 token 长度过滤 + cutoff 与数据匹配**。guard:`validate()` 现在对
  ">25% 样本超 cutoff(按 chars/3 估)" 给出 warning(见 backends/llamafactory)。

## 2. 基座未训练的特殊 token(qwen3 的 `<think>`/`<tool_call>`)

- 现象:模型在模板特殊 token 位置输出乱码(`具有战士`/`𫟦`/`即时发生`),内容本身正常。
- 根因:`<think>`(151667)/`<tool_call>`(151657)在 Qwen3-Base 里未训练;LoRA 默认
  不更新 embedding/lm_head,模型只能用乱码近似。
  - 补充实证:LF 的 `additional_target: lm_head` 训练的权重**不会**被 LF export 合入
    final_model(export 只合 LoRA 层,丢 modules_to_save);peft 手动合并可解,但小数据
    下嵌入行也推不动多少。
  - **bfcl 专项结论(干净实验:≤1024 过滤数据 + additional_target=lm_head + peft 手动
    合并)**:模型依然只会输出乱码——**1.7B-Base + LoRA + ~1k 样本学不动
    `<tool_call>`**。该 bench 在此模型规模下的可行路径:full-FT(全参数含嵌入)、
    更大的 function-calling 数据量(万级+),或换用已训练过该 token 的模型。
- 修法:**非推理数据用 `qwen3_nothink` 模板**(不引入未训练 token),立竿见影。
  `<tool_call>` 是独立的另一题(function calling 场景),需要更多数据或 full-FT 再议。

## 3. eval 的 vllm 显存按 H100 硬编码

- 现象:24G 卡上 vllm 起不来:"KV cache 2.40 GiB < 3.50 GiB required"。
- 根因:`run_lmeval_eval.py` 的 `DIRECT_GPU_MEMORY = {"humaneval": 0.3, ...}` 按
  80G 卡调的(0.3×80=24G 够 32k 上下文;0.3×24=7.2G 不够)。
- 修法:24G 卡把 util 提到 0.8(3090 上的 runtime 副本已改;runtime 契约侧后续宜改成
  按卡型适配)。

## 4. 数据形状 ≠ scorer 格式契约(gpqa 线全程实证)

- 现象:训练 loss 正常、模型"会答",但分数贴地。
- 根因链(逐层剥开):数据是长段落(camel)而 bench 要字母 → 换 MCQ 格式(题面+字母)
  仍贴地 → **scorer 的提取协议是 `ANSWER: X` 正则匹配**(utils.py `_extract_choice`),
  训练输出必须逐字对齐这个格式。同时注意 prompt 说 "Think step by step"——数据最好也带
  简短推理,格式与评测指令一致。
- 修法:**先看 scorer 源码再配数据**(runtime 的 `scoring.format_contract` 就是干这个的)。
  数据↔评测是逆流水线契约:冻结 scorer 是不动点,上游一切对齐它。

## 5. LESS/梯度选择:D_val(参照集)是灵魂,切片 ≠ 任务参照

- 现象:LESS 式梯度余弦选择出的 top-k 训练,**比随机选还差**(humaneval:LESS 15.2 vs
  random 38.4)。
- 根因:D_val 用"候选池前 100 条"只是普通切片,选出来的是与它冗余的**窄子集**;
  梯度余弦无错,但参照物本身不代表目标任务。
- 修法:**D_val 必须是目标任务的代表样本**(bench 的训练集 / few-shot),而不是从
  候选池里随手切。这就是 task 契约里 `selection_target` 该从 bench train split 注入、
  而不是大脑自由发挥的原因。LESS 论文的 D_val 也是目标任务的 few-shot。
- 复核(2026-07-25,humaneval,seed42,同预算 926 条):换了代表性 D_val(100 条任务
  代表样本)后 LESS 19.5,比切片 D_val 的 15.2 好,但仍远低于 random 38.4(≈5σ)。
  两种 D_val 的选中集 Jaccard 0.91——梯度相似度排序被池子侧特征主导,D_val 只能
  挪动边缘行。**最终判定:LESS 在 1.7B-Base + LoRA + 千条池规模下打不过随机采样,
  与 D_val 选法无关**;池子已贴合任务时 random 保住多样性,top-N 反而收窄多样性。
  实现本身(梯度签名/稀疏投影/打分链路)已验证无误,是方法在该规模不适用。

## 6. HF Trainer 管线里给 batch 注入自定义键的两个暗坑(row_indices 接线实录)

往 HF Trainer 的批次里注入自定义键(我们注入的是 `__row_index__` 逐行索引)时,单独
冒烟 dataset 包装和 collator 包装都能过,真训练才被剥光——两个坑叠在一起:

- **RemoveColumnsCollator 剥键**:`Trainer._get_dataloader` 会用
  `_get_collator_with_removed_columns` 把你的 collator 再包一层最外层,凡不在模型
  `forward` 签名里的键一律删掉。解法:override `_get_collator_with_removed_columns`,
  把自定义键 append 进 `signature_columns` 白名单(commit `8f066fe`)。
- **`__getitems__` 批量路径绕过 `__getitem__`**:torch 的 `_MapDatasetFetcher` 发现
  dataset 有 `__getitems__` 就优先走它;`datasets.Dataset` 恰好有,而包装类的
  `__getattr__` 委托会把内层这个 dunder 暴露出去,于是注入逻辑被完全绕过(表象:
  包装类构造了、`__getitem__` 一次都没被调)。解法:显式覆写 `__getitems__` 并在
  批量返回里逐个注入(commit `2e49875`)。
- 验证手段:200 条小数据 + 2 epoch + 按观测 loss 收窄一半的 select 算子,步数签名
  `75/100`(epoch1 全量 50 步 + epoch2 半量 25 步)是干净利落的判据。

## 7. 评测吞吐参考(3090)

- 训练:1.7B LoRA,bs4×cutoff512,~2s/step(250 步 ≈ 8 分钟)。
- eval:humaneval(164 题,短答案)~6 分钟;gpqamain(448 题,长答案)~2-4 小时/模型——
  扫参建议 `--limit 200`。
- aime2025(30 题,long-CoT 模型):单次 eval **可达 5 小时级**(模型对每题生成近 16k
  token 上限的长推理,且容易被 vllm KV cache 限并发)——aime 类实验要把 eval 时长算进
  预算,或先截断生成长度再评。
