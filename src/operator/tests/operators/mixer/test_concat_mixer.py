from __future__ import annotations

from dataos_operator.operators.mixer.concat_mixer import ConcatMixer


def test_concat_mixer_accepts_single_dataset() -> None:
    operator = ConcatMixer()

    result = operator.run([
        {"id": "a", "text": "alpha"},
        {"id": "b", "text": "beta"},
    ])

    assert result.to_records() == [
        {"id": "a", "text": "alpha"},
        {"id": "b", "text": "beta"},
    ]
    assert result.metadata["kind"] == "mixer"
    assert result.metadata["input_records"] == 2
    assert result.metadata["output_records"] == 2


def test_concat_mixer_accepts_multiple_datasets_and_preserves_order() -> None:
    operator = ConcatMixer()

    result = operator.run(
        [
            [{"id": "a"}, {"id": "b"}],
            [{"id": "c"}],
            [{"id": "d"}, {"id": "e"}],
        ]
    )

    assert result.to_records() == [
        {"id": "a"},
        {"id": "b"},
        {"id": "c"},
        {"id": "d"},
        {"id": "e"},
    ]
    assert result.metadata["input_items"] == 3
    assert result.metadata["output_records"] == 5


def test_concat_mixer_can_handle_empty_datasets() -> None:
    operator = ConcatMixer()

    result = operator.run([[], [{"id": "a"}], []])

    assert result.to_records() == [{"id": "a"}]
