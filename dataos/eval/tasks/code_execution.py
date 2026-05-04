from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(?P<code>.*?)```", re.IGNORECASE | re.DOTALL)
ANY_CODE_FENCE_RE = re.compile(r"```\s*\n(?P<code>.*?)```", re.DOTALL)
RESULT_PREFIX = "__DATAOS_CODE_EXECUTION_RESULT__="


@dataclass(slots=True)
class CodeTestCase:
    kind: str
    code: str | None = None
    stdin: str | None = None
    expected_stdout: str | None = None


@dataclass(slots=True)
class CodeExecutionResult:
    passed: bool
    passed_tests: int
    failed_tests: int
    execution_error: str | None = None
    test_results: list[dict[str, Any]] = field(default_factory=list)


def extract_python_code(text: str) -> str:
    for pattern in (CODE_FENCE_RE, ANY_CODE_FENCE_RE):
        match = pattern.search(text)
        if match:
            return match.group("code").strip()
    return text.strip()


def normalize_code_tests(raw_tests: Any, *, entry_point: str | None = None) -> list[CodeTestCase]:
    if raw_tests is None:
        return []
    if isinstance(raw_tests, str):
        parsed = _parse_json_if_possible(raw_tests)
        if parsed is not None:
            return normalize_code_tests(parsed, entry_point=entry_point)
        code = extract_python_code(raw_tests)
        return [CodeTestCase(kind="unit", code=code)] if code else []
    if isinstance(raw_tests, (list, tuple)):
        cases: list[CodeTestCase] = []
        for item in raw_tests:
            cases.extend(normalize_code_tests(item, entry_point=entry_point))
        return cases
    if isinstance(raw_tests, dict):
        test_type = str(raw_tests.get("testtype", "")).strip().lower()
        for key in ("public_tests", "tests", "test_cases", "cases"):
            if key in raw_tests:
                return normalize_code_tests(raw_tests[key], entry_point=entry_point)
        if "asserts" in raw_tests:
            return normalize_code_tests(raw_tests["asserts"], entry_point=entry_point)
        if test_type == "functional":
            code = _functional_assertion_code(raw_tests, entry_point=entry_point)
            return [CodeTestCase(kind="unit", code=code)]
        code = _first_string_value(raw_tests, ("code", "test", "assert", "assertion"))
        if code is not None:
            return normalize_code_tests(code, entry_point=entry_point)
        stdin = _first_present_value(raw_tests, ("input", "stdin"))
        expected = _first_present_value(
            raw_tests,
            ("output", "expected", "expected_output", "stdout"),
        )
        if stdin is not None and expected is not None:
            return [
                CodeTestCase(
                    kind="stdin_stdout",
                    stdin=_coerce_stdio(stdin),
                    expected_stdout=_coerce_stdio(expected),
                )
            ]
    raise ValueError(f"Unsupported code test format: {type(raw_tests).__name__}")


def run_python_code_tests(
    *,
    code: str,
    tests: Any,
    sandbox_tmp_dir: Path | str,
    sample_id: str,
    timeout_seconds: float = 5.0,
    entry_point: str | None = None,
) -> CodeExecutionResult:
    test_cases = normalize_code_tests(tests, entry_point=entry_point)
    if not test_cases:
        raise ValueError("At least one code test is required")

    base_dir = Path(sandbox_tmp_dir) / "livecodebench"
    work_dir = base_dir / f"{_safe_name(sample_id)}-{uuid4().hex[:8]}"
    work_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = work_dir / "candidate.py"
    candidate_path.write_text(extract_python_code(code), encoding="utf-8")

    if all(case.kind == "stdin_stdout" for case in test_cases):
        return _run_stdin_stdout_tests(
            candidate_path=candidate_path,
            test_cases=test_cases,
            work_dir=work_dir,
            timeout_seconds=timeout_seconds,
        )
    if all(case.kind == "unit" for case in test_cases):
        return _run_unit_tests(
            candidate_path=candidate_path,
            test_cases=test_cases,
            work_dir=work_dir,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError("Mixed unit and stdin/stdout tests are not supported")


def _run_unit_tests(
    *,
    candidate_path: Path,
    test_cases: list[CodeTestCase],
    work_dir: Path,
    timeout_seconds: float,
) -> CodeExecutionResult:
    runner_path = work_dir / "unit_runner.py"
    tests_payload = [case.code or "" for case in test_cases]
    runner_path.write_text(
        _unit_runner_source(candidate_path=candidate_path, tests_payload=tests_payload),
        encoding="utf-8",
    )
    completed = _run_subprocess(runner_path, work_dir, timeout_seconds)
    return _result_from_completed_process(completed, total_tests=len(test_cases))


def _run_stdin_stdout_tests(
    *,
    candidate_path: Path,
    test_cases: list[CodeTestCase],
    work_dir: Path,
    timeout_seconds: float,
) -> CodeExecutionResult:
    wrapper_path = work_dir / "stdin_runner.py"
    wrapper_path.write_text(_stdin_runner_source(candidate_path=candidate_path), encoding="utf-8")

    results: list[dict[str, Any]] = []
    execution_errors: list[str] = []
    for index, case in enumerate(test_cases):
        try:
            completed = _run_subprocess(
                wrapper_path,
                work_dir,
                timeout_seconds,
                stdin=case.stdin or "",
            )
        except TimeoutError as exc:
            error = str(exc)
            execution_errors.append(error)
            results.append({"index": index, "passed": False, "error": error, "kind": case.kind})
            continue
        if completed.returncode != 0:
            error = _compact_process_error(completed)
            execution_errors.append(error)
            results.append({"index": index, "passed": False, "error": error, "kind": case.kind})
            continue
        actual = _normalize_stdout(completed.stdout)
        expected = _normalize_stdout(case.expected_stdout or "")
        passed = actual == expected
        results.append(
            {
                "index": index,
                "passed": passed,
                "kind": case.kind,
                "actual_stdout": actual,
                "expected_stdout": expected,
            }
        )

    passed_tests = sum(1 for result in results if result["passed"])
    failed_tests = len(results) - passed_tests
    return CodeExecutionResult(
        passed=failed_tests == 0,
        passed_tests=passed_tests,
        failed_tests=failed_tests,
        execution_error="\n".join(execution_errors) if execution_errors else None,
        test_results=results,
    )


def _run_subprocess(
    runner_path: Path,
    work_dir: Path,
    timeout_seconds: float,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = _subprocess_env(work_dir)
    try:
        return subprocess.run(
            [sys.executable, "-I", str(runner_path)],
            input=stdin,
            cwd=work_dir,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"code execution timed out after {timeout_seconds:g}s") from exc


def _result_from_completed_process(
    completed: subprocess.CompletedProcess[str],
    *,
    total_tests: int,
) -> CodeExecutionResult:
    if completed.returncode != 0:
        error = _compact_process_error(completed)
        return CodeExecutionResult(
            passed=False,
            passed_tests=0,
            failed_tests=total_tests,
            execution_error=error,
            test_results=[
                {"index": index, "passed": False, "error": error, "kind": "unit"}
                for index in range(total_tests)
            ],
        )

    payload = _extract_result_payload(completed.stdout)
    if payload is None:
        error = "code execution did not emit a result payload"
        return CodeExecutionResult(
            passed=False,
            passed_tests=0,
            failed_tests=total_tests,
            execution_error=error,
            test_results=[
                {"index": index, "passed": False, "error": error, "kind": "unit"}
                for index in range(total_tests)
            ],
        )
    results = payload.get("test_results", [])
    if not isinstance(results, list):
        results = []
    passed_tests = sum(1 for result in results if isinstance(result, dict) and result.get("passed"))
    failed_tests = max(total_tests - passed_tests, 0)
    execution_error = payload.get("execution_error")
    return CodeExecutionResult(
        passed=failed_tests == 0 and not execution_error,
        passed_tests=passed_tests,
        failed_tests=failed_tests,
        execution_error=str(execution_error) if execution_error else None,
        test_results=results,
    )


def _unit_runner_source(*, candidate_path: Path, tests_payload: list[str]) -> str:
    return f"""\
from __future__ import annotations

import json
import sys
import types
import traceback
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import *

{_restriction_source()}

CANDIDATE_PATH = Path({str(candidate_path)!r})
TESTS = {json.dumps(tests_payload, ensure_ascii=False)}
RESULT_PREFIX = {RESULT_PREFIX!r}


def _emit(payload):
    print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False, sort_keys=True))


module = types.ModuleType("candidate")
module.__dict__.update({{
    "Counter": Counter,
    "defaultdict": defaultdict,
    "deque": deque,
}})
sys.modules["candidate"] = module
try:
    source = CANDIDATE_PATH.read_text(encoding="utf-8")
    exec(compile(source, str(CANDIDATE_PATH), "exec"), module.__dict__, module.__dict__)
except BaseException:
    error = traceback.format_exc(limit=20)
    _emit({{
        "execution_error": error,
        "test_results": [
            {{"index": index, "passed": False, "kind": "unit", "error": error}}
            for index in range(len(TESTS))
        ],
    }})
    raise SystemExit(0)

namespace = {{"candidate": module, **module.__dict__}}
test_results = []
execution_errors = []
for index, test_code in enumerate(TESTS):
    try:
        exec(test_code, namespace, namespace)
    except AssertionError:
        error = traceback.format_exc(limit=20)
        test_results.append({{
            "index": index,
            "passed": False,
            "kind": "unit",
            "error": error,
            "error_type": "assertion",
        }})
    except BaseException:
        error = traceback.format_exc(limit=20)
        execution_errors.append(error)
        test_results.append({{
            "index": index,
            "passed": False,
            "kind": "unit",
            "error": error,
            "error_type": "execution",
        }})
    else:
        test_results.append({{"index": index, "passed": True, "kind": "unit"}})

_emit({{
    "execution_error": "\\n".join(execution_errors) if execution_errors else None,
    "test_results": test_results,
}})
"""


def _stdin_runner_source(*, candidate_path: Path) -> str:
    return f"""\
from __future__ import annotations

import runpy
from pathlib import Path

{_restriction_source()}

runpy.run_path(str(Path({str(candidate_path)!r})), run_name="__main__")
"""


def _restriction_source() -> str:
    return r'''
import builtins
import io
import os
import socket
import subprocess
from pathlib import Path

_ALLOWED_ROOT = Path(__file__).resolve().parent
_ORIGINAL_OPEN = builtins.open
_ORIGINAL_IO_OPEN = io.open
_ORIGINAL_PATH_OPEN = Path.open
_ORIGINAL_OS_OPEN = os.open
_ORIGINAL_OS_REMOVE = os.remove
_ORIGINAL_OS_UNLINK = os.unlink
_ORIGINAL_OS_MKDIR = os.mkdir
_ORIGINAL_OS_MAKEDIRS = os.makedirs
_ORIGINAL_OS_RMDIR = os.rmdir
_ORIGINAL_OS_RENAME = os.rename
_ORIGINAL_OS_REPLACE = os.replace


def _is_write_mode(mode):
    mode_text = str(mode)
    return any(flag in mode_text for flag in ("w", "a", "x", "+"))


def _assert_write_allowed(file, mode):
    if not _is_write_mode(mode):
        return
    try:
        path = Path(file)
    except TypeError:
        return
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    try:
        path.relative_to(_ALLOWED_ROOT)
    except ValueError as exc:
        raise PermissionError("writes outside the sandbox tmp directory are disabled") from exc


def _guarded_open(file, mode="r", *args, **kwargs):
    _assert_write_allowed(file, mode)
    return _ORIGINAL_OPEN(file, mode, *args, **kwargs)


def _guarded_io_open(file, mode="r", *args, **kwargs):
    _assert_write_allowed(file, mode)
    return _ORIGINAL_IO_OPEN(file, mode, *args, **kwargs)


def _guarded_path_open(self, mode="r", *args, **kwargs):
    _assert_write_allowed(self, mode)
    return _ORIGINAL_PATH_OPEN(self, mode, *args, **kwargs)


def _guarded_os_open(file, flags, mode=0o777, *args, **kwargs):
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
    if flags & write_flags:
        _assert_write_allowed(file, "w")
    return _ORIGINAL_OS_OPEN(file, flags, mode, *args, **kwargs)


def _guarded_remove(path, *args, **kwargs):
    _assert_write_allowed(path, "w")
    return _ORIGINAL_OS_REMOVE(path, *args, **kwargs)


def _guarded_unlink(path, *args, **kwargs):
    _assert_write_allowed(path, "w")
    return _ORIGINAL_OS_UNLINK(path, *args, **kwargs)


def _guarded_mkdir(path, mode=0o777, *args, **kwargs):
    _assert_write_allowed(path, "w")
    return _ORIGINAL_OS_MKDIR(path, mode, *args, **kwargs)


def _guarded_makedirs(name, mode=0o777, exist_ok=False):
    _assert_write_allowed(name, "w")
    return _ORIGINAL_OS_MAKEDIRS(name, mode=mode, exist_ok=exist_ok)


def _guarded_rmdir(path, *args, **kwargs):
    _assert_write_allowed(path, "w")
    return _ORIGINAL_OS_RMDIR(path, *args, **kwargs)


def _guarded_rename(src, dst, *args, **kwargs):
    _assert_write_allowed(src, "w")
    _assert_write_allowed(dst, "w")
    return _ORIGINAL_OS_RENAME(src, dst, *args, **kwargs)


def _guarded_replace(src, dst, *args, **kwargs):
    _assert_write_allowed(src, "w")
    _assert_write_allowed(dst, "w")
    return _ORIGINAL_OS_REPLACE(src, dst, *args, **kwargs)


def _blocked(*args, **kwargs):
    raise RuntimeError("network and subprocess access are disabled during code evaluation")


class _BlockedSocket:
    def __init__(self, *args, **kwargs):
        _blocked()


builtins.open = _guarded_open
io.open = _guarded_io_open
Path.open = _guarded_path_open
os.open = _guarded_os_open
os.remove = _guarded_remove
os.unlink = _guarded_unlink
os.mkdir = _guarded_mkdir
os.makedirs = _guarded_makedirs
os.rmdir = _guarded_rmdir
os.rename = _guarded_rename
os.replace = _guarded_replace
os.system = _blocked
socket.socket = _BlockedSocket
socket.create_connection = _blocked
subprocess.Popen = _blocked
subprocess.run = _blocked
subprocess.call = _blocked
subprocess.check_call = _blocked
subprocess.check_output = _blocked
'''


def _extract_result_payload(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                payload = json.loads(line[len(RESULT_PREFIX) :])
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None
    return None


def _compact_process_error(completed: subprocess.CompletedProcess[str]) -> str:
    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    message = stderr or stdout or f"process exited with code {completed.returncode}"
    return message[-4000:]


def _subprocess_env(work_dir: Path) -> dict[str, str]:
    env: dict[str, str] = {
        "HOME": str(work_dir),
        "TMPDIR": str(work_dir),
        "PYTHONNOUSERSITE": "1",
        "NO_PROXY": "*",
        "no_proxy": "*",
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "http_proxy": "",
        "https_proxy": "",
    }
    if "PATH" in os.environ:
        env["PATH"] = os.environ["PATH"]
    return env


def _parse_json_if_possible(value: str) -> Any | None:
    text = value.strip()
    if not text or text[0] not in "[{":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _functional_assertion_code(raw_test: dict[str, Any], *, entry_point: str | None) -> str:
    if not entry_point:
        raise ValueError("Functional code tests require a metadata func_name/entry_point")
    raw_input = _first_present_value(raw_test, ("input", "stdin"))
    raw_output = _first_present_value(raw_test, ("output", "expected", "expected_output", "stdout"))
    if raw_input is None or raw_output is None:
        raise ValueError("Functional code tests require input and output values")
    args = _parse_functional_args(_coerce_stdio(raw_input))
    expected = _parse_case_value(_coerce_stdio(raw_output).strip())
    return f"""\
_DATAOS_ENTRY_POINT = {entry_point!r}
_DATAOS_ARGS = {args!r}
_DATAOS_EXPECTED = {expected!r}
_DATAOS_TARGET = globals().get(_DATAOS_ENTRY_POINT)
if _DATAOS_TARGET is None:
    _DATAOS_SOLUTION = globals().get("Solution")
    if _DATAOS_SOLUTION is None:
        raise AssertionError(f"missing function or Solution class for {{_DATAOS_ENTRY_POINT}}")
    _DATAOS_TARGET = getattr(_DATAOS_SOLUTION(), _DATAOS_ENTRY_POINT)
_DATAOS_ACTUAL = _DATAOS_TARGET(*_DATAOS_ARGS)
assert _DATAOS_ACTUAL == _DATAOS_EXPECTED, (
    f"expected {{_DATAOS_EXPECTED!r}}, got {{_DATAOS_ACTUAL!r}}"
)
"""


def _parse_functional_args(value: str) -> list[Any]:
    stripped = value.strip()
    if not stripped:
        return []
    return [_parse_case_value(line.strip()) for line in stripped.splitlines() if line.strip()]


def _parse_case_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _first_string_value(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _first_present_value(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _coerce_stdio(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)
    return str(value)


def _normalize_stdout(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())[:64].strip("-")
    return safe or "sample"
