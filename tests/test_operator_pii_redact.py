from __future__ import annotations

from datasets import Dataset

from dataos.core.sandbox import Sandbox
from dataos.operators.pii_redact import PiiRedactOperator
from operator_test_utils import build_source


def test_pii_redact_masks_email_phone_and_url(tmp_path):
    source = build_source(
        tmp_path,
        Dataset.from_dict({
            "instruction": ["email me at alice@example.com or visit https://example.com"],
            "response": ["call +86 138 1234 5678"],
        }),
    )
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = PiiRedactOperator().run([source], sandbox)
    row = artifact.load()[0]
    assert "[EMAIL]" in row["instruction"]
    assert "[URL]" in row["instruction"]
    assert "[PHONE]" in row["response"]
    assert artifact.meta["pii_redact_counts"]["train"]["email"] == 1
    assert artifact.meta["pii_redact_counts"]["train"]["url"] == 1
