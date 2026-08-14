# 图片多模态基础算子

第一批图片算子统一使用下面的 record 结构：

```json
{
  "messages": [
    {"role": "user", "content": "<image>\n描述图片"},
    {"role": "assistant", "content": "一张示例图片"}
  ],
  "images": ["images/example.png"],
  "metadata": {"images": []}
}
```

`messages` 中的 `<image>` 与顶层 `images` 按出现顺序一一对应。文件解析和哈希目前只
处理本地图片；路径规范化可以保留 HTTP、S3、GS 和 OSS URI，但不会主动下载远端文件。

## 已实现算子

| 算子 | kind | 作用 |
| --- | --- | --- |
| `ImageTextMessageMapper` | mapper | 将字段型或 OpenAI content-block 样本统一成 `messages + images` |
| `ImagePathNormalizerMapper` | mapper | 解析相对路径、转绝对路径、可选检查文件存在性 |
| `ImageMetadataEvaluator` | evaluator | 解码图片并记录格式、宽高、像素数、宽高比、字节数和完整性 |
| `ImageHashEvaluator` | evaluator | 流式计算 SHA-256 等文件哈希，并合并进图片 metadata |
| `ImageIntegrityFilter` | selector | 删除不存在或无法解码的图片样本 |
| `ImageTextAlignmentFilter` | selector | 校验 `<image>` 数量与 `images` 数量一致 |
| `ImageCountFilter` | selector | 按单条样本的图片数量过滤 |
| `ImageShapeFilter` | selector | 按宽、高、总像素数过滤 |
| `ImageAspectRatioFilter` | selector | 按宽高比过滤 |
| `ImageSizeFilter` | selector | 按编码文件字节数过滤 |
| `ImageFormatFilter` | selector | 按解码后的图片格式白名单过滤 |
| `ImageExactDeduplicateSelector` | selector | 使用预计算文件哈希做精确去重 |

## 推荐处理顺序

```text
ImageTextMessageMapper
  -> ImagePathNormalizerMapper
  -> ImageMetadataEvaluator(strict=False)
  -> ImageIntegrityFilter
  -> ImageTextAlignmentFilter
  -> ImageCount/Shape/AspectRatio/Size/Format filters
  -> ImageHashEvaluator
  -> ImageExactDeduplicateSelector
```

图片解码依赖 Pillow：

```bash
uv sync --extra image
```

多图片过滤器默认要求全部图片满足条件；设置 `strategy="any"` 可改为任意一张满足即可。
`keep_missing` 和 `keep_empty` 分别控制元数据字段缺失与图片列表为空时是否保留样本。

## 最小示例

```python
from dataos_operator import Dataset
from dataos_operator.operators.evaluator import ImageHashEvaluator, ImageMetadataEvaluator
from dataos_operator.operators.mapper import ImagePathNormalizerMapper, ImageTextMessageMapper
from dataos_operator.operators.selector import (
    ImageExactDeduplicateSelector,
    ImageIntegrityFilter,
    ImageTextAlignmentFilter,
)

data = Dataset.from_records(
    [{"query": "这是什么？", "answer": "一只猫。", "images": ["cat.png"]}]
)
data = ImageTextMessageMapper(source_format="fields").run(data)
data = ImagePathNormalizerMapper(media_dir="/data/images").run(data)
data = ImageMetadataEvaluator(strict=False).run(data)
data = ImageIntegrityFilter().run(data)
data = ImageTextAlignmentFilter().run(data)
data = ImageHashEvaluator().run(data)
data = ImageExactDeduplicateSelector().run(data)
```

导出 LLaMA-Factory ShareGPT 数据时，适配器会同时写出 `messages` 与 `images` 列，并在
`dataset_info.json` 中声明图片列。
