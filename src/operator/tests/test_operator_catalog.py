from __future__ import annotations

from dataos_operator.core import list_operator_specs


def test_list_operator_specs_contains_length_filter() -> None:
    specs = list_operator_specs()

    assert specs == [
        spec
        for spec in specs
        if spec.name != "" and spec.kind and spec.description and spec.operator_class is not None
    ]
    assert any(
        spec.name == "length_filter"
        and spec.kind == "selector"
        and spec.operator_class == "dataos_operator.operators.selector.length_filter.LengthFilter"
        for spec in specs
    )



def test_list_operator_specs_contains_concat_mixer() -> None:
    specs = list_operator_specs()

    assert any(
        spec.name == "concat_mixer"
        and spec.kind == "mixer"
        and spec.operator_class == "dataos_operator.operators.mixer.concat_mixer.ConcatMixer"
        for spec in specs
    )
