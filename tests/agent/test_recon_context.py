"""What of the recon reaches the leader, and what stays on disk.

Measured on a real run's recon.json: 237K total, of which the annotated per-source profiles are
175K. Inlining that table (even once) into a persistent dialog is the worst case — it occupies
the history for the rest of the run and is still gone after a compaction. The leader gets the
shortlist, the ids the plan gate accepts, and the target shape; the table stays in the file it
already lives in.
"""

import json

import pytest

from dataos.agent.loop import _recon_context

_RECON = {
    "eval_format_contract": {
        "bench": "b",
        "summary": "S" * 400,
        "parser": "utils.py:50 process_results takes results[0] and " + "P" * 600,
        "format_sensitive": True,
        "expected_target_shape": "T" * 370,
        "target_example": "E" * 230,
    },
    "baseline_diagnosis": {"cause": "capability", "evidence": "E" * 700},
    "candidate_sources": [{"source": f"s{i}", "why": "w", "conversion": "c"} for i in range(7)],
    "source_profiles": [
        {"source": f"s{i}", "shape": "alpaca", "n_rows": 1000, "note": "N" * 250,
         "samples": ["R" * 2000]}
        for i in range(106)
    ],
}


@pytest.fixture
def view(tmp_path):
    return _recon_context(_RECON, recon_path=tmp_path / "recon.json")


def test_shortlist_and_every_legal_id_are_inline(view):
    """The shortlist is the intended starting point; the ids are what the gate accepts."""
    assert len(view["candidate_sources"]) == 7
    assert view["legal_source_ids"] == [f"s{i}" for i in range(106)]


def test_the_profile_table_is_never_inlined(view, tmp_path):
    assert "source_profiles" not in view
    assert view["recon_path"] == str(tmp_path / "recon.json")
    assert "N" * 250 not in json.dumps(view)  # no per-source notes
    assert "R" * 2000 not in json.dumps(view)  # no raw samples


def test_parser_internals_go_to_the_experimenter_not_the_leader(view):
    """file:line into the scorer is only usable by whoever re-checks targets against it."""
    contract = view["eval_format_contract"]
    assert "parser" not in contract
    assert contract["expected_target_shape"] and contract["target_example"]
    assert contract["format_sensitive"] is True


def test_the_view_is_the_same_every_round(tmp_path):
    """No round-dependent branch: information the leader needs must not expire after round 1."""
    a = _recon_context(_RECON, recon_path=tmp_path / "r.json")
    b = _recon_context(_RECON, recon_path=tmp_path / "r.json")
    assert a == b


def test_view_stays_small(view):
    """Sanity bound: the whole view is a few KB, not tens of KB."""
    assert len(json.dumps(view, ensure_ascii=False, indent=2)) < 8000


def test_no_recon_passes_through(tmp_path):
    assert _recon_context(None, recon_path=tmp_path / "r.json") is None
