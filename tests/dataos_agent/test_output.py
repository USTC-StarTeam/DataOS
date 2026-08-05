import pytest
from pydantic import BaseModel

from dataos_agent._output import parse_output
from dataos_agent.errors import OutputParseError


class Plan(BaseModel):
    name: str
    steps: int


def test_raw_json():
    p = parse_output('{"name": "exp1", "steps": 3}', Plan)
    assert p.name == "exp1" and p.steps == 3


def test_fenced_block():
    text = "Here is the plan:\n```json\n{\"name\": \"a\", \"steps\": 2}\n```\nDone."
    p = parse_output(text, Plan)
    assert p.name == "a" and p.steps == 2


def test_embedded_in_prose_and_thinking():
    text = 'let me think... the answer is {"name": "b", "steps": 1} — final.'
    p = parse_output(text, Plan)
    assert p.name == "b" and p.steps == 1


def test_picks_the_valid_object_among_several():
    text = 'noise {"unrelated": 1} then {"name": "c", "steps": 9} end'
    p = parse_output(text, Plan)
    assert p.name == "c" and p.steps == 9


def test_missing_field_raises():
    with pytest.raises(OutputParseError):
        parse_output('{"name": "x"}', Plan)          # steps missing


def test_no_json_raises():
    with pytest.raises(OutputParseError):
        parse_output("there is no json here at all", Plan)


def test_reads_from_file(tmp_path):
    f = tmp_path / "result.json"
    f.write_text('{"name": "fromfile", "steps": 7}')
    p = parse_output(str(f), Plan)
    assert p.name == "fromfile" and p.steps == 7
