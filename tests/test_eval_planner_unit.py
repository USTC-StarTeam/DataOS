"""Unit tests for the NL planner: gallery, intent, resolver. No lm-eval needed."""
from __future__ import annotations

from typing import Any

import pytest

from dataos.eval.planner import (
    EvalIntent,
    EvalPlanItem,
    builtin_gallery,
    keyword_intent,
    llm_intent,
    plan_eval,
    resolve_tasks,
)


class _ScriptedLLM:
    def __init__(self, payload: Any):
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def generate_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> Any:
        self.calls.append((system_prompt, user_prompt))
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class TestKeywordIntent:
    def test_recognizes_math_synonyms_zh_and_en(self):
        intent = keyword_intent("评估模型在数学和推理上的表现")
        assert "math" in intent.topics
        assert "reasoning" in intent.topics

    def test_code_goal_enables_unsafe_code(self):
        intent = keyword_intent("看看模型写 Python code 的能力")
        assert "code" in intent.topics
        assert intent.allow_unsafe_code is True

    def test_word_boundary_avoids_substring_matches(self):
        intent = keyword_intent("decode the barcode reader output")
        assert "code" not in intent.topics

    def test_empty_goal_returns_empty_topics(self):
        intent = keyword_intent("")
        assert intent.topics == ()


class TestLLMIntent:
    def test_uses_llm_payload_when_valid(self):
        llm = _ScriptedLLM(
            {
                "topics": ["math", "code"],
                "skills": ["multi_step", "python"],
                "allow_unsafe_code": True,
                "rationale": "user wants math and python",
            }
        )
        intent = llm_intent("anything", llm)
        assert set(intent.topics) == {"math", "code"}
        assert intent.allow_unsafe_code is True
        assert intent.rationale.startswith("user wants")
        assert llm.calls, "LLM should have been called"

    def test_falls_back_to_keyword_when_llm_raises(self):
        llm = _ScriptedLLM(RuntimeError("boom"))
        intent = llm_intent("看看数学能力", llm)
        assert "math" in intent.topics
        assert "keyword fallback" in intent.rationale

    def test_falls_back_to_keyword_when_llm_returns_garbage(self):
        llm = _ScriptedLLM(["not", "a", "dict"])
        intent = llm_intent("评估推理", llm)
        assert "reasoning" in intent.topics


class TestResolver:
    def test_math_intent_picks_gsm8k(self):
        intent = EvalIntent(raw_goal="math reasoning", topics=("math", "reasoning"))
        plan = resolve_tasks(intent, max_tasks=4)
        tasks = [item.task for item in plan]
        assert "gsm8k" in tasks
        assert plan[0].task in {"gsm8k", "bbh_cot_zeroshot_logical_deduction_three_objects"}

    def test_only_runnable_keeps_multiple_choice_for_transformers_string(self):
        """Default adapter='transformers' is conservatively recognized as
        logprob-capable, so multiple_choice tasks (mmlu) are kept."""
        intent = EvalIntent(raw_goal="general knowledge", topics=("knowledge",))
        plan = resolve_tasks(intent, only_runnable=True, adapter="transformers")
        assert any(item.task == "mmlu" for item in plan), [i.task for i in plan]

    def test_only_runnable_excludes_multiple_choice_for_unknown_string_adapter(self):
        """Adapter strings other than 'transformers' are conservatively
        treated as generate-only — multiple_choice tasks must be filtered."""
        intent = EvalIntent(raw_goal="general knowledge", topics=("knowledge",))
        plan = resolve_tasks(intent, only_runnable=True, adapter="openai_compatible")
        assert all(item.task != "mmlu" for item in plan), [i.task for i in plan]

    def test_only_runnable_excludes_multiple_choice_for_live_generate_only_adapter(self):
        """Live adapter without score_continuations -> multiple_choice filtered."""

        class _GenerateOnly:
            name = "live_generate_only"

            def generate_until(self, model_artifact, requests, config):
                return []

        intent = EvalIntent(raw_goal="general knowledge", topics=("knowledge",))
        plan = resolve_tasks(intent, only_runnable=True, adapter=_GenerateOnly())
        assert all(item.task != "mmlu" for item in plan), [i.task for i in plan]

    def test_only_runnable_keeps_multiple_choice_for_live_scoring_adapter(self):
        """Live adapter implementing score_continuations -> multiple_choice kept."""

        class _ScoringAdapter:
            name = "live_scoring"

            def score_continuations(self, model_artifact, pairs, config):
                return [(0.0, True) for _ in pairs]

        intent = EvalIntent(raw_goal="general knowledge", topics=("knowledge",))
        plan = resolve_tasks(intent, only_runnable=True, adapter=_ScoringAdapter())
        assert any(item.task == "mmlu" for item in plan), [i.task for i in plan]

    def test_runnable_false_includes_multiple_choice(self):
        intent = EvalIntent(raw_goal="general knowledge", topics=("knowledge",))
        plan = resolve_tasks(intent, only_runnable=False, adapter="openai_compatible")
        assert any(item.task == "mmlu" for item in plan)

    def test_resolver_logs_filtered_capability_drops(self, caplog):
        """Master's observability requirement: when capability-driven filtering
        happens, log one line naming the dropped tasks and the adapter."""
        intent = EvalIntent(raw_goal="general knowledge", topics=("knowledge",))
        with caplog.at_level("INFO", logger="dataos.eval.planner"):
            resolve_tasks(intent, only_runnable=True, adapter="openai_compatible")
        records = [r for r in caplog.records if r.name == "dataos.eval.planner"]
        assert records, "expected an INFO log from dataos.eval.planner"
        message = records[0].getMessage()
        assert "openai_compatible" in message
        assert "score_continuations" in message
        assert "mmlu" in message

    def test_resolver_does_not_log_when_no_drops(self, caplog):
        intent = EvalIntent(raw_goal="math", topics=("math",))
        with caplog.at_level("INFO", logger="dataos.eval.planner"):
            resolve_tasks(intent, only_runnable=True, adapter="transformers")
        assert not [r for r in caplog.records if r.name == "dataos.eval.planner"]

    def test_unsafe_code_required_for_humaneval(self):
        intent_no = EvalIntent(raw_goal="code", topics=("code",), allow_unsafe_code=False)
        plan_no = resolve_tasks(intent_no)
        assert all(item.task != "humaneval" for item in plan_no)

        intent_yes = EvalIntent(raw_goal="code", topics=("code",), allow_unsafe_code=True)
        plan_yes = resolve_tasks(intent_yes)
        assert any(item.task == "humaneval" for item in plan_yes)
        humaneval = next(item for item in plan_yes if item.task == "humaneval")
        assert humaneval.confirm_run_unsafe_code is True

    def test_no_topic_match_falls_back(self):
        intent = EvalIntent(raw_goal="random unrelated request", topics=("nothing",))
        plan = resolve_tasks(intent)
        tasks = [item.task for item in plan]
        assert "gsm8k" in tasks  # default fallback ships gsm8k

    def test_task_name_in_goal_boosts_score(self):
        intent = EvalIntent(raw_goal="please run gsm8k", topics=())
        plan = resolve_tasks(intent)
        assert plan[0].task == "gsm8k"

    def test_resolver_carries_gallery_defaults(self):
        intent = EvalIntent(raw_goal="instruction following", topics=("instruction_following",))
        plan = resolve_tasks(intent, max_tasks=1)
        assert plan and plan[0].task == "ifeval"
        assert plan[0].max_new_tokens == 1280
        assert plan[0].headline_metric == "prompt_level_strict_acc,none"


class TestPlanEval:
    def test_zero_to_plan_with_keyword_path(self):
        plan = plan_eval("评估数学和指令遵循", max_tasks=3)
        tasks = {item.task for item in plan}
        assert "gsm8k" in tasks
        assert "ifeval" in tasks

    def test_llm_path_is_used_when_client_provided(self):
        llm = _ScriptedLLM(
            {
                "topics": ["code"],
                "skills": [],
                "allow_unsafe_code": True,
                "rationale": "code only",
            }
        )
        plan = plan_eval("anything", llm_client=llm)
        assert any(item.task == "humaneval" for item in plan)
        assert llm.calls, "LLM client should have been used"


class TestGallery:
    def test_builtin_has_unique_task_names(self):
        gallery = builtin_gallery()
        names = [entry.task for entry in gallery.entries]
        assert len(names) == len(set(names))

    def test_filter_by_output_type(self):
        gallery = builtin_gallery().filter(output_types=("generate_until",))
        assert all(entry.output_type == "generate_until" for entry in gallery.entries)
        assert any(entry.task == "gsm8k" for entry in gallery.entries)
