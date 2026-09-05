# Indexed multimodal curation

The indexed path complements the existing record-oriented text path. It preserves the original
Hugging Face features and image payloads while operators select source row indices.

```python
from dataos_operator import ConversationFeatureEvaluator, WeightedReservoirSelector
from dataos_operator.integrations.curationbench import load_pool, save_submission

pool = load_pool(dataset_path)
featured = ConversationFeatureEvaluator(texts_key="texts").run(pool)
selected = WeightedReservoirSelector(
    sample_size=10_000,
    weight_key="metadata.conversation.richness_weight",
    strata_keys=["subset", "metadata.conversation.format"],
    quotas={
        "coco|mcq": 4_000,
        "coco|long_form": 3_500,
        "ocr_vqa|multi_turn": 1_500,
        "textvqa|short_vqa": 1_000,
    },
    seed=42,
).run(featured)
save_submission(selected, output_path, manifest_path=manifest_path)
```

The numbers above are an example recipe, not a built-in strategy. Task agents choose quotas from
their own evidence. The operators provide reusable mechanics:

- `ConversationFeatureEvaluator`: paired-turn statistics, format/capability tags, question-group
  identity, coordinate-output signal, and a bounded richness weight.
- `WeightedReservoirSelector`: streaming Efraimidis-Spirakis sampling with exact strata quotas.
- `StratifiedQuotaSelector`: explicit or capacity-weighted source/task/format allocation.
- `GroupAwareSelector`: atomic MCQ rotations or a configurable per-image cap.
- `CompositeDeduplicateSelector`: exact multi-field identity; shared images are not implicitly
  discarded.
- `AnchorRefreshSelector`: retain a controlled reference dose and replace within identical
  strata.

Use `Dataset` for existing text pipelines. Use `DatasetView` when the source is a large Arrow
dataset or when rebuilding records could alter image features. Record-oriented operators can still
be applied after an explicitly bounded shortlist is materialized.
