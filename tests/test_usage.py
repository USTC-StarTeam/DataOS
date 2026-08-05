"""Tests for the progressive-disclosure usage catalog."""

from __future__ import annotations

import json

import pytest

import dataos_operator
from dataos_operator import catalog, guide, operator_docs, reference, usage


def test_catalog_is_compact_index() -> None:
    text = catalog()
    assert isinstance(text, str)
    assert "Operator catalog" in text
    # One bullet per operator; effect line present, but no parameter tables.
    assert "- `length_filter`" in text
    assert "| param |" not in text
    # Compact: far smaller than the full reference.
    assert len(text) < len(reference()) // 5


def test_catalog_dict_rows() -> None:
    rows = catalog(fmt="dict")
    assert isinstance(rows, list) and rows
    row = rows[0]
    assert set(row) == {"name", "kind", "mode", "effect"}


def test_usage_is_single_operator_detail() -> None:
    text = usage("loss_selector")
    assert isinstance(text, str)
    assert "loss_selector" in text
    assert "Constructor:" in text  # has the how-to detail
    # Single operator only — no other operator leaks in.
    assert "length_filter" not in text


def test_usage_dict_and_online_example() -> None:
    entry = usage("loss_selector", fmt="dict")
    assert isinstance(entry, dict)
    assert entry["name"] == "loss_selector"
    assert entry["mode"] == "online"
    json.dumps(entry["online_config_example"])  # must be serializable


def test_usage_unknown_raises_with_pointer() -> None:
    with pytest.raises(ValueError, match="catalog"):
        usage("does_not_exist")


def test_usage_accepts_class_name_case_insensitive() -> None:
    entry = usage("LengthFilter", fmt="dict")
    assert entry["name"] == "length_filter"


def test_guide_is_shipped() -> None:
    text = guide()
    assert "DataOS Operator — Usage Guide" in text
    assert "How operators are used" in text


def test_filters_and_modes() -> None:
    selectors = catalog(kind="selector", fmt="dict")
    assert selectors and all(r["kind"] == "selector" for r in selectors)

    online = operator_docs(mode="online")
    assert online and all(e["mode"] == "online" for e in online)
    assert {"loss_selector", "doremi_mixer", "loss_reweighter"} <= {e["name"] for e in online}


def test_reference_is_the_full_manual() -> None:
    text = reference()
    assert dataos_operator.__version__  # smoke
    assert "DataOS Operator — Usage Guide" in text
    assert "# Operators" in text
    assert "length_filter" in text and "loss_selector" in text
