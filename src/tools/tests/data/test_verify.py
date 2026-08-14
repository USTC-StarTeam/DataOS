"""The output contract — the check the data stage never had.

Each case here is a way the stage has actually failed, or the shape of the failure it would have
caught. The one that motivates the module is `test_a_recipe_that_filtered_everything_out_fails`:
that build exited 0, and the failure surfaced one job later as a training job dying in 52 seconds
on an empty dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataos_data.verify import MANIFEST, main, materialize_alpaca, verify


def _sft(tmp_path: Path, rows: list[dict] | None, *, alpaca: bool = True) -> Path:
    """A data/sft dir. ``alpaca=False`` is the normal recipe now: it writes the HF form only."""
    from datasets import Dataset

    d = tmp_path / "data" / "sft"
    d.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows or []).save_to_disk(str(d))
    if alpaca:
        (d / "train.json").write_text(json.dumps(rows or []), encoding="utf-8")
    return d


_ROWS = [{"instruction": "q", "input": "", "output": "a"}] * 3


def test_both_forms_present_and_agreeing_is_the_contract(tmp_path):
    m = verify(_sft(tmp_path, _ROWS))
    assert m["rows"] == 3
    assert len(m["fingerprint"]) == 64  # sha256 of train.json — the recipe's real identity


def test_a_reserved_multimodal_token_is_refused(tmp_path):
    """The expensive half, and the one that actually got through.

    That node's NEXT build completed cleanly and the training job died 52 seconds later on
    `ValueError: The number of images does not match the number of <image> tokens` — Gemma 3
    reserves the string. The batch before lost an arenahardwriting node to `<video>`.
    """
    rows = [{"instruction": "describe <image>", "input": "", "output": "ok"}]
    with pytest.raises(SystemExit, match="<image>"):
        verify(_sft(tmp_path, rows), template="gemma3")


def test_the_scan_is_off_unless_a_template_is_named(tmp_path):
    """No template, no claim: a deployment whose template tolerates the string is not this one."""
    rows = [{"instruction": "describe <image>", "input": "", "output": "ok"}]
    assert verify(_sft(tmp_path, rows))["rows"] == 1


def test_the_hf_form_is_required_because_the_converter_reads_it(tmp_path):
    """A recipe that writes only train.json fails at TRAIN time, which reads as a train problem."""
    d = tmp_path / "data" / "sft"
    d.mkdir(parents=True)
    (d / "train.json").write_text(json.dumps(_ROWS), encoding="utf-8")
    with pytest.raises(SystemExit, match="save_to_disk"):
        verify(d)


def test_the_alpaca_form_is_derived_when_the_recipe_wrote_only_the_dataset(tmp_path):
    """The recipe writes ONE form. It used to be told to write two, the second a hand
    transcription of the first — a whole class of build failure that was only transcription."""
    d = _sft(tmp_path, _ROWS, alpaca=False)
    m = verify(d)
    assert m["rows"] == 3 and m["alpaca"] == "derived"
    assert json.loads((d / "train.json").read_text()) == [
        {"instruction": "q", "input": "", "output": "a", "system": "", "history": []}
    ] * 3
    # and the LF registry that names it, written by the converter's own writer
    assert json.loads((d / "dataset_info.json").read_text())["dataos_sft"]["file_name"] == "train.json"


def test_a_recipe_that_wrote_its_own_alpaca_form_keeps_it(tmp_path):
    """It is the authority on its own rows; deriving over it would silently reshape them."""
    d = _sft(tmp_path, _ROWS)
    before = (d / "train.json").read_text()
    assert verify(d)["alpaca"] == "written by the recipe"
    assert (d / "train.json").read_text() == before
    assert materialize_alpaca(d) is None


def test_a_messages_dataset_projects_through_the_converters_own_alpaca(tmp_path):
    """Derived by the SAME function the training job's converter uses, so the two agree because
    they are one implementation — not because two were checked against each other."""
    rows = [{"messages": [{"role": "user", "content": "hi"},
                          {"role": "assistant", "content": "yo"}]}] * 2
    d = verify(_sft(tmp_path, rows, alpaca=False)) and (tmp_path / "data" / "sft")
    written = json.loads((d / "train.json").read_text())
    assert written[0]["instruction"] == "hi" and written[0]["output"] == "yo"


def test_the_derived_form_still_faces_every_check(tmp_path):
    """Deriving does not exempt the data from the scan — the reserved token came from a SOURCE
    row, so it survives any transcription, hand-written or not."""
    rows = [{"instruction": "describe <video>", "input": "", "output": "ok"}]
    with pytest.raises(SystemExit, match="<video>"):
        verify(_sft(tmp_path, rows, alpaca=False), template="gemma3")


def test_the_two_forms_must_be_the_same_rows(tmp_path):
    d = _sft(tmp_path, _ROWS)
    (d / "train.json").write_text(json.dumps(_ROWS[:1]), encoding="utf-8")
    with pytest.raises(SystemExit, match="disagree"):
        verify(d)


def test_the_recipes_own_funnel_is_kept_not_replaced(tmp_path):
    """Recipes record a per-source funnel worth having; what they cannot do is agree on a name.

    The same 2026-08-14 batch wrote `data_stats.json` in one run root and `data_manifest.json` in
    another, for the same thing. The name is fixed here; the content is merged under one key.
    """
    stats = {"openmathinstruct2": {"in": 105154, "after_dedup": 66166, "capped": 25000}}
    m = verify(_sft(tmp_path, _ROWS), stats=stats)
    assert m["recipe_stats"] == stats
    assert m["rows"] == 3  # the recipe's own numbers cannot overwrite the measured ones


def test_the_manifest_lands_next_to_the_data_under_one_name(tmp_path, capsys):
    d = _sft(tmp_path, _ROWS)
    assert main(["--dir", str(d)]) == 0
    written = json.loads((tmp_path / MANIFEST).read_text(encoding="utf-8"))
    assert written["rows"] == 3
    assert json.loads(capsys.readouterr().out)["ok"] is True
