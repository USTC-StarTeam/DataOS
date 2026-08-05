from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_OPERATOR_DIRS = {
    "clusterer",
    "evaluator",
    "generator",
    "mapper",
    "mixer",
    "refiner",
    "reweighter",
    "selector",
}


# 验证 operators 下保留全部预期算子类别目录。
def test_expected_operator_directories_exist() -> None:
    operators_root = ROOT / "dataos_operator" / "operators"

    assert operators_root.is_dir()
    assert {
        path.name
        for path in operators_root.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    } == EXPECTED_OPERATOR_DIRS


# 验证每个算子类别目录都有 README 和包初始化文件。
def test_each_operator_directory_has_readme_and_package_init() -> None:
    operators_root = ROOT / "dataos_operator" / "operators"

    for name in EXPECTED_OPERATOR_DIRS:
        operator_dir = operators_root / name
        assert (operator_dir / "README.md").is_file()
        assert (operator_dir / "__init__.py").is_file()


# 验证仓库关键文档、示例数据和 CI 文件存在。
def test_project_documents_exist() -> None:
    expected_files = [
        ROOT / "README.md",
        ROOT / "dataos_operator" / "dataset.py",
        ROOT / "dataset" / "README.md",
        ROOT / "docs" / "README.md",
        ROOT / "docs" / "architecture.md",
        ROOT / "docs" / "operator_contribution.md",
        ROOT / "docs" / "research" / "README.md",
        ROOT / "examples" / "datasets" / "README.md",
        ROOT / "examples" / "datasets" / "gsm8k_target_100.jsonl",
        ROOT / "tests" / "fixtures" / "README.md",
        ROOT / "tests" / "fixtures" / "gsm8k_target_100.jsonl",
        ROOT / ".github" / "workflows" / "ci.yml",
    ]

    for path in expected_files:
        assert path.is_file()
