"""Unit tests for dataos.research.agent specialist-prompt helpers."""
from __future__ import annotations

import pytest

from dataos.research.agent import (
    _OPERATOR_AWARE_ROLES,
    _build_operator_catalog,
    _role_system_prompt,
)


_SAMPLE_DESCRIPTIONS = [
    {"name": "dedup", "description": "Exact deduplication on a composite key over one or more columns."},
    {"name": "less", "description": "Select an influential training subset for a target task via LESS."},
    {"name": "sample", "description": "Deterministically cap the size of one or more dataset splits."},
    {"name": "format_convert", "description": "Convert SFT rows between alpaca, sharegpt, and messages schemas."},
]


def test_build_operator_catalog_names():
    catalog = _build_operator_catalog(_SAMPLE_DESCRIPTIONS)
    names = {entry["name"] for entry in catalog}
    assert names == {"dedup", "less", "sample", "format_convert"}


def test_build_operator_catalog_summary_format():
    catalog = _build_operator_catalog(_SAMPLE_DESCRIPTIONS)
    dedup = next(e for e in catalog if e["name"] == "dedup")
    assert dedup["summary"].startswith("dedup — ")


def test_build_operator_catalog_truncates_long_entries():
    long_desc = [{"name": "x" * 20, "description": "y" * 70}]
    catalog = _build_operator_catalog(long_desc)
    assert len(catalog[0]["summary"]) <= 80


def test_build_operator_catalog_from_real_registry():
    from dataos.operators import build_default_registry
    descriptions = build_default_registry().describe_all()
    catalog = _build_operator_catalog(descriptions)
    names = {e["name"] for e in catalog}
    assert "dedup" in names
    assert "less" in names


def test_experiment_planner_prompt_contains_operator_names():
    catalog = _build_operator_catalog(_SAMPLE_DESCRIPTIONS)
    prompt = _role_system_prompt("experiment_planner", catalog)
    assert "dedup" in prompt
    assert "less" in prompt


def test_operator_designer_prompt_contains_operator_names():
    catalog = _build_operator_catalog(_SAMPLE_DESCRIPTIONS)
    prompt = _role_system_prompt("operator_designer", catalog)
    assert "dedup" in prompt
    assert "less" in prompt


def test_critic_prompt_contains_operator_names():
    catalog = _build_operator_catalog(_SAMPLE_DESCRIPTIONS)
    prompt = _role_system_prompt("critic", catalog)
    assert "dedup" in prompt
    assert "less" in prompt


def test_non_operator_aware_roles_unchanged():
    catalog = _build_operator_catalog(_SAMPLE_DESCRIPTIONS)
    for role in ("hypothesis", "evaluation_analyst", "report"):
        base = _role_system_prompt(role)
        with_catalog = _role_system_prompt(role, catalog)
        assert base == with_catalog, f"role={role!r} prompt should not change with catalog"
        assert "dedup" not in with_catalog


def test_empty_catalog_falls_back_to_base():
    base = _role_system_prompt("experiment_planner")
    empty = _role_system_prompt("experiment_planner", [])
    assert base == empty


def test_operator_aware_roles_set_contents():
    assert "experiment_planner" in _OPERATOR_AWARE_ROLES
    assert "operator_designer" in _OPERATOR_AWARE_ROLES
    assert "critic" in _OPERATOR_AWARE_ROLES
    assert "hypothesis" not in _OPERATOR_AWARE_ROLES
