from __future__ import annotations

import textwrap

from dataos_operator.core import validate_operator_layout


def test_current_operator_layout_is_valid() -> None:
    assert validate_operator_layout() == []


def test_validation_accepts_temp_operator_package(tmp_path, monkeypatch) -> None:
    package_root = tmp_path / "sample_pkg"
    operator_root = package_root / "operators"
    selector_root = operator_root / "selector"
    selector_root.mkdir(parents=True)

    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (operator_root / "__init__.py").write_text("", encoding="utf-8")
    (selector_root / "__init__.py").write_text("", encoding="utf-8")
    (selector_root / "sample_selector.py").write_text(
        textwrap.dedent(
            '''
            from __future__ import annotations

            from collections.abc import Sequence

            from dataos_operator.core import Record, SelectorOperator


            class SampleSelector(SelectorOperator):
                """保持所有输入 record 不变。"""

                def select(self, data: Sequence[Sequence[Record]]) -> Sequence[Record]:
                    return [record for dataset in data for record in dataset]
            '''
        ),
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))

    assert validate_operator_layout("sample_pkg.operators") == []


def test_validation_rejects_kind_directory_mismatch(tmp_path, monkeypatch) -> None:
    package_root = tmp_path / "bad_pkg"
    operator_root = package_root / "operators"
    selector_root = operator_root / "selector"
    selector_root.mkdir(parents=True)

    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (operator_root / "__init__.py").write_text("", encoding="utf-8")
    (selector_root / "__init__.py").write_text("", encoding="utf-8")
    (selector_root / "wrong_kind.py").write_text(
        textwrap.dedent(
            '''
            from dataos_operator.core import EvaluatorOperator, Record


            class WrongKind(EvaluatorOperator):
                """这个类声明了 evaluator 语义。"""

                def evaluate_record(self, record: Record) -> Record:
                    return record
            '''
        ),
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))

    issues = validate_operator_layout("bad_pkg.operators")

    assert len(issues) == 1
    assert "必须等于目录名" in issues[0].message
