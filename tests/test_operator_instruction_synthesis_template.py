from __future__ import annotations

from datasets import DatasetDict, Dataset

from dataos.core.sandbox import Sandbox
from dataos.operators.instruction_synthesis_template import InstructionSynthesisTemplateOperator
from operator_test_utils import build_source


def test_instruction_synthesis_template_adds_deterministic_variants(tmp_path):
    source = build_source(
        tmp_path,
        DatasetDict({"train": Dataset.from_dict({"instruction": ["What is 2 + 2?"], "response": ["4"]})}),
    )
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = InstructionSynthesisTemplateOperator().run([source], sandbox, templates=["brief_answer", "answer_only"])
    loaded = artifact.load()["train"]
    assert loaded.num_rows == 3
    assert artifact.meta["instruction_synthesis_added_rows"] == 2
    assert "dataos_synth_template" in loaded.column_names
