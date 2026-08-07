"""The recon source profiler must never silently zero a loadable source: layered
extraction (known keys -> sharegpt turns -> structural inference) with explicit
extraction metadata and a degraded flag when half the sample has no target text."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "profile_source",
    Path(__file__).resolve().parents[2] / "resources/recon/posttrainbench/profile_source.py",
)
ps = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ps)


@pytest.mark.parametrize("row,want_key", [
    ({"instruction": "q", "output": "a"}, "key:output"),                      # alpaca
    ({"problem": "q", "generated_solution": "a"}, "key:generated_solution"),  # openmathinstruct2
    ({"question": "q", "answer": "a"}, "key:answer"),                         # qa
    ({"messages": [{"role": "user", "content": "q"},
                   {"role": "assistant", "content": "a"}]}, "turns:assistant"),  # messages
    ({"conversations": [{"from": "human", "value": "q"},
                        {"from": "gpt", "value": "a"}]}, "turns:assistant"),     # sharegpt
    ({"foo": "x", "body": "a" * 100}, "inferred_longest:body"),               # unknown schema
])
def test_extract_output_covers_schemas(row, want_key):
    out, mode = ps._extract_output(row)
    assert out.strip()
    assert mode == want_key


def test_extract_output_never_silent_with_text_fields():
    out, mode = ps._extract_output({"totally_unknown_field": "some text"})
    assert out == "some text"
    assert mode.startswith("inferred_")


def test_extract_context_problem_key():
    ctx, mode = ps._extract_context({"problem": "2+2?", "generated_solution": "4"})
    assert "2+2?" in ctx
    assert mode == "keys:problem"


def _fake_pool(rows):
    """Wrap rows as a DatasetDict-shaped fake: {'train': rows-with-ds-API}."""

    class FakeDS(list):
        def shuffle(self, seed=None): return self

        def select(self, rng): return FakeDS([self[i] for i in rng])

    return {"train": FakeDS(rows)}


def test_profile_marks_degraded_when_extraction_fails(monkeypatch):
    # rows with no string fields at all -> extraction must fail everywhere
    monkeypatch.setattr(ps, "_load_full",
                        lambda *_a: _fake_pool([{"n": 1, "arr": [1, 2]}] * 10))
    p = ps.profile_source("/fake", "fake_source", n_sample=10)
    assert "degraded" in p
    assert p["extraction"]["empty_outputs"] == 10
    assert p["samples_raw"]  # the raw rows are still visible for manual judgment


def test_profile_not_degraded_on_good_schema(monkeypatch):
    monkeypatch.setattr(
        ps, "_load_full",
        lambda *_a: _fake_pool(
            [{"problem": f"q{i}", "generated_solution": f"work... \\boxed{{{i}}}"} for i in range(10)]
        ),
    )
    p = ps.profile_source("/fake", "openmathinstruct2", n_sample=10)
    assert "degraded" not in p
    assert p["sample"]["output"]          # a real sample is present
    assert p["domain"]["scores"]["math"] == 1.0  # \boxed{} rows count as math evidence
    assert p["shape"] == "qa"
    assert p["splits"] == {"train": 10}
    assert p["lengths"]["target_chars"]["max"] > 0


@pytest.mark.parametrize("schema,want", [
    (["chosen", "messages", "prompt", "rejected"], "preference"),
    (["translation"], "translation"),
    (["conversations", "system", "tools"], "chat_tools"),
    (["conversations", "system"], "chat"),
    (["answerKey", "choices", "question"], "mcq"),
    (["instruction", "input", "output"], "alpaca"),
    (["problem", "generated_solution"], "qa"),
    (["label", "text"], "classification"),
    (["text"], "raw_text"),
    (["blob_a", "blob_b", "blob_c", "blob_d", "blob_e"], "other"),
])
def test_shape_classification(schema, want):
    assert ps._shape(schema) == want
