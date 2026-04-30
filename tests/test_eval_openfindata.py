from __future__ import annotations

from datasets import Dataset

from dataos.eval.registry import get_eval_task
from dataos.eval.tasks.openfindata import OpenFinDataTask
from dataos.eval.types import EvalConfig, EvalTaskConfig, ModelResponse


def _config() -> EvalConfig:
    return EvalConfig(
        provider="native",
        task=EvalTaskConfig(
            name="openfindata",
            config={
                "question_field": "query",
                "answer_field": "gold",
                "task_type_field": "kind",
            },
        ),
    )


def _default_config() -> EvalConfig:
    return EvalConfig(
        provider="native",
        task=EvalTaskConfig(name="openfindata"),
    )


def test_openfindata_scores_choice_extraction_and_short_answer_with_field_overrides():
    dataset = Dataset.from_dict(
        {
            "query": [
                "请指出哪个数据有明显错误。 A. 无明显错误数据 B. 英国10年期国债收益率错误",
                "请列出以下内容中提及的股市概念板块：无人驾驶概念股强势走高。",
                "请列出以下内容中提及的股市概念板块：机器人与人工智能板块走强。",
                "英国10年期国债收益率是多少？",
            ],
            "gold": [
                "A",
                "无人驾驶",
                "机器人，人工智能",
                "4.328%",
            ],
            "kind": [
                "金融判别/选择题",
                "实体/概念抽取",
                "实体/概念抽取",
                "金融问答",
            ],
        }
    )
    task = OpenFinDataTask()
    config = _config()
    requests = task.build_requests(dataset, config)
    responses = [
        ModelResponse(
            sample_id=requests[0].sample_id,
            index=requests[0].index,
            request_type=requests[0].request_type,
            text="答案：无明显错误数据",
        ),
        ModelResponse(
            sample_id=requests[1].sample_id,
            index=requests[1].index,
            request_type=requests[1].request_type,
            text="根据文本内容，提及的概念板块为：\n\n**无人驾驶**",
        ),
        ModelResponse(
            sample_id=requests[2].sample_id,
            index=requests[2].index,
            request_type=requests[2].request_type,
            text="文中提到的概念板块有：\n- 机器人\n- 人工智能",
        ),
        ModelResponse(
            sample_id=requests[3].sample_id,
            index=requests[3].index,
            request_type=requests[3].request_type,
            text="最终答案：4.328%",
        ),
    ]

    bundle = task.score_responses(dataset, requests, responses, config)

    assert bundle.metric_name == "exact_match"
    assert bundle.score == 1.0
    assert bundle.summary["correct"] == 4
    assert bundle.summary["num_scored"] == 4
    assert bundle.summary["num_skipped"] == 0
    assert bundle.summary["type_counts"] == {"choice": 1, "extraction": 2, "qa": 1}
    assert bundle.samples[0].normalized_prediction["letters"] == ["A"]
    assert bundle.samples[1].normalized_prediction == ["无人驾驶"]
    assert bundle.samples[2].normalized_prediction == ["人工智能", "机器人"]
    assert bundle.samples[3].normalized_prediction == "4.328%"


def test_openfindata_choice_accepts_option_letter_for_full_text_reference():
    dataset = Dataset.from_dict(
        {
            "question": ["请选择金融任务类型。"],
            "A": ["金融知识"],
            "B": ["金融合规"],
            "answer": ["金融合规"],
            "source_file": ["financial_facts.json"],
        }
    )
    task = OpenFinDataTask()
    config = _default_config()
    requests = task.build_requests(dataset, config)
    responses = [
        ModelResponse(
            sample_id=requests[0].sample_id,
            index=requests[0].index,
            request_type=requests[0].request_type,
            text="B",
        )
    ]

    bundle = task.score_responses(dataset, requests, responses, config)

    assert bundle.score == 1.0
    assert bundle.samples[0].normalized_prediction["letters"] == ["B"]
    assert bundle.samples[0].normalized_reference["letters"] == ["B"]


def test_openfindata_builds_opencompass_style_prompt_from_option_columns():
    dataset = Dataset.from_dict(
        {
            "question": ["请选择金融任务类型。"],
            "A": ["金融知识"],
            "B": ["金融合规"],
            "answer": ["B"],
            "source_file": ["financial_facts.json"],
        }
    )
    task = OpenFinDataTask()
    requests = task.build_requests(dataset, _default_config())

    assert requests[0].prompt == "请选择金融任务类型。\nA. 金融知识\nB. 金融合规\n答案: "
    assert requests[0].meta["output_type"] == "choice"
    assert requests[0].meta["choice_options"] == {"A": "金融知识", "B": "金融合规"}


def test_openfindata_infers_extraction_type_from_opencompass_source_file():
    dataset = Dataset.from_dict(
        {
            "question": ["请列出以下内容中提及的股市概念板块：无人驾驶概念股强势走高。"],
            "answer": ["无人驾驶"],
            "source_file": ["entity_recognition.json"],
        }
    )
    task = OpenFinDataTask()
    requests = task.build_requests(dataset, _default_config())

    assert requests[0].prompt == "请列出以下内容中提及的股市概念板块：无人驾驶概念股强势走高。\n答案: "
    assert requests[0].meta["output_type"] == "extraction"


def test_openfindata_choice_requires_unique_final_answer():
    dataset = Dataset.from_dict(
        {
            "query": ["请选择金融任务类型。 A. 金融知识 B. 金融合规 C. 金融分析"],
            "gold": ["C"],
            "kind": ["选择题"],
        }
    )
    task = OpenFinDataTask()
    config = _config()
    requests = task.build_requests(dataset, config)
    responses = [
        ModelResponse(
            sample_id=requests[0].sample_id,
            index=requests[0].index,
            request_type=requests[0].request_type,
            text="A 不对，B 也不合适，因此最终答案是 C。",
        )
    ]

    bundle = task.score_responses(dataset, requests, responses, config)

    assert bundle.score == 1.0
    assert bundle.samples[0].normalized_prediction["letters"] == ["C"]


def test_openfindata_choice_does_not_credit_ambiguous_reasoning():
    dataset = Dataset.from_dict(
        {
            "query": ["请选择金融任务类型。 A. 金融知识 B. 金融合规 C. 金融分析"],
            "gold": ["C"],
            "kind": ["选择题"],
        }
    )
    task = OpenFinDataTask()
    config = _config()
    requests = task.build_requests(dataset, config)
    responses = [
        ModelResponse(
            sample_id=requests[0].sample_id,
            index=requests[0].index,
            request_type=requests[0].request_type,
            text="A 和 B 都看起来有道理，C 也可能正确。",
        )
    ]

    bundle = task.score_responses(dataset, requests, responses, config)

    assert bundle.score == 0.0
    assert bundle.samples[0].normalized_prediction["letters"] == []


def test_openfindata_skips_nan_reference_answers():
    dataset = Dataset.from_dict(
        {
            "query": ["请回答合规问题。", "请选择金融任务类型。 A. 金融知识 B. 金融合规"],
            "gold": ["nan", "B"],
            "kind": ["金融问答", "选择题"],
        }
    )
    task = OpenFinDataTask()
    config = _config()
    requests = task.build_requests(dataset, config)
    responses = [
        ModelResponse(
            sample_id=requests[0].sample_id,
            index=requests[0].index,
            request_type=requests[0].request_type,
            text="这里是一段长回答。",
        ),
        ModelResponse(
            sample_id=requests[1].sample_id,
            index=requests[1].index,
            request_type=requests[1].request_type,
            text="B",
        ),
    ]

    bundle = task.score_responses(dataset, requests, responses, config)

    assert bundle.score == 1.0
    assert bundle.summary["num_examples"] == 2
    assert bundle.summary["num_scored"] == 1
    assert bundle.summary["num_skipped"] == 1
    assert bundle.samples[0].meta["is_skipped"] is True
    assert bundle.samples[0].metrics == {}
    assert bundle.samples[1].meta["is_skipped"] is False


def test_openfindata_is_registered():
    assert isinstance(get_eval_task("openfindata"), OpenFinDataTask)
