from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset

from dataos.eval.providers.base import EvalRunContext
from dataos.eval.tasks.code_execution import (
    extract_python_code,
    normalize_code_tests,
    run_python_code_tests,
)
from dataos.eval.types import EvalConfig, EvalRequest, EvalResultBundle, EvalSampleResult, ModelResponse

DEFAULT_PROMPT_TEMPLATE = """\
You are solving a Python programming problem.
Return only complete Python code. Do not include Markdown fences or explanations.

Problem:
{question}
{starter_section}"""


class LiveCodeBenchTask:
    name = "livecodebench"
    version = "v1"
    supported_request_types = {"generate_until"}

    def __init__(self) -> None:
        self._sandbox_tmp_dir: Path | None = None

    def set_runtime_context(self, context: EvalRunContext) -> None:
        self._sandbox_tmp_dir = context.sandbox.tmp_dir

    def default_split(self) -> str:
        return "test"

    def build_requests(self, dataset: Dataset, config: EvalConfig) -> list[EvalRequest]:
        task_config = _task_config(config)
        prompt_template = str(task_config.get("prompt_template", DEFAULT_PROMPT_TEMPLATE))
        requests: list[EvalRequest] = []
        for index, row in enumerate(dataset):
            question, question_field = _extract_question(row, index=index, task_config=task_config)
            starter_code = _extract_starter_code(row, task_config=task_config)
            tests, tests_field = _extract_tests(row, index=index, task_config=task_config)
            test_count = len(normalize_code_tests(tests))
            if test_count == 0:
                raise ValueError(f"Row {index} requires at least one executable LiveCodeBench test")
            prompt = _build_prompt(
                question=question,
                starter_code=starter_code,
                prompt_template=prompt_template,
            )
            requests.append(
                EvalRequest(
                    sample_id=str(row.get("id", index)),
                    index=index,
                    request_type="generate_until",
                    prompt=prompt,
                    meta={
                        "question_field": question_field,
                        "starter_code_present": bool(starter_code),
                        "tests_field": tests_field,
                        "test_count": test_count,
                    },
                )
            )
        return requests

    def score_responses(
        self,
        dataset: Dataset,
        requests: list[EvalRequest],
        responses: list[ModelResponse],
        config: EvalConfig,
    ) -> EvalResultBundle:
        rows = list(dataset)
        if len(rows) != len(requests):
            raise ValueError("Request count does not match evaluation dataset size")
        if len(rows) != len(responses):
            raise ValueError("Response count does not match evaluation dataset size")

        task_config = _task_config(config)
        sandbox_tmp_dir = _resolve_sandbox_tmp_dir(task_config, self._sandbox_tmp_dir)
        timeout_seconds = float(task_config.get("timeout_seconds", task_config.get("timeout", 5.0)))

        samples: list[EvalSampleResult] = []
        passed_samples = 0
        total_passed_tests = 0
        total_failed_tests = 0
        execution_error_count = 0

        for index, (row, request, response) in enumerate(zip(rows, requests, responses)):
            tests, tests_field = _extract_tests(row, index=index, task_config=task_config)
            generated_code = extract_python_code(response.text or "")
            execution_result = run_python_code_tests(
                code=generated_code,
                tests=tests,
                sandbox_tmp_dir=sandbox_tmp_dir,
                sample_id=request.sample_id,
                timeout_seconds=timeout_seconds,
            )
            sample_pass = 1.0 if execution_result.passed else 0.0
            passed_samples += int(execution_result.passed)
            total_passed_tests += execution_result.passed_tests
            total_failed_tests += execution_result.failed_tests
            execution_error_count += int(execution_result.execution_error is not None)
            samples.append(
                EvalSampleResult(
                    sample_id=request.sample_id,
                    index=index,
                    prompt=request.prompt,
                    prediction=response.text,
                    reference=tests,
                    normalized_prediction=generated_code,
                    normalized_reference=None,
                    metrics={
                        "pass@1": sample_pass,
                        "correct": sample_pass,
                        "passed_tests": float(execution_result.passed_tests),
                        "failed_tests": float(execution_result.failed_tests),
                        "execution_error": float(execution_result.execution_error is not None),
                    },
                    meta={
                        "is_correct": execution_result.passed,
                        "tests_field": tests_field,
                        "execution_error": execution_result.execution_error,
                        "test_results": execution_result.test_results,
                    },
                )
            )

        total = len(samples)
        pass_at_1 = passed_samples / total if total else 0.0
        return EvalResultBundle(
            metric_name="pass@1",
            score=pass_at_1,
            summary={
                "pass@1": pass_at_1,
                "passed_tests": total_passed_tests,
                "failed_tests": total_failed_tests,
                "execution_error": execution_error_count,
                "num_examples": total,
                "correct": passed_samples,
            },
            samples=samples,
        )


def _task_config(config: EvalConfig) -> dict[str, Any]:
    if config.task is None:
        return {}
    return dict(config.task.config)


def _extract_question(
    row: dict[str, Any],
    *,
    index: int,
    task_config: dict[str, Any],
) -> tuple[str, str]:
    configured_field = task_config.get("question_field")
    field_names = _field_candidates(
        configured_field,
        ("question", "description", "prompt", "problem", "problem_description"),
    )
    for field_name in field_names:
        value = row.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip(), field_name
    available = ", ".join(field_names)
    raise ValueError(f"Row {index} requires a non-empty question/description field from: {available}")


def _extract_starter_code(row: dict[str, Any], *, task_config: dict[str, Any]) -> str:
    configured_field = task_config.get("starter_code_field")
    field_names = _field_candidates(
        configured_field,
        ("starter_code", "starter", "code", "function_signature"),
    )
    for field_name in field_names:
        value = row.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_tests(
    row: dict[str, Any],
    *,
    index: int,
    task_config: dict[str, Any],
) -> tuple[Any, str]:
    configured_field = task_config.get("tests_field")
    field_names = _field_candidates(
        configured_field,
        ("public_tests", "tests", "public_test_cases", "test_cases"),
    )
    for field_name in field_names:
        value = row.get(field_name)
        if _has_tests(value):
            return value, field_name
    available = ", ".join(field_names)
    raise ValueError(f"Row {index} requires executable tests from one of: {available}")


def _build_prompt(*, question: str, starter_code: str, prompt_template: str) -> str:
    starter_section = ""
    if starter_code:
        starter_section = f"\nStarter code:\n```python\n{starter_code}\n```"
    return prompt_template.format(question=question, starter_code=starter_code, starter_section=starter_section)


def _resolve_sandbox_tmp_dir(task_config: dict[str, Any], runtime_tmp_dir: Path | None) -> Path:
    if runtime_tmp_dir is not None:
        return runtime_tmp_dir
    configured = task_config.get("sandbox_tmp_dir")
    if configured is not None:
        return Path(str(configured))
    raise ValueError(
        "LiveCodeBench eval requires sandbox tmp runtime context or "
        "eval_config.task.config.sandbox_tmp_dir"
    )


def _field_candidates(configured_field: Any, defaults: tuple[str, ...]) -> tuple[str, ...]:
    if configured_field is None:
        return defaults
    configured = str(configured_field).strip()
    if not configured:
        return defaults
    return (configured, *(field for field in defaults if field != configured))


def _has_tests(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return True
