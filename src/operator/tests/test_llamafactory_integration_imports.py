from __future__ import annotations


def test_llamafactory_integration_namespace_is_lazy() -> None:
    import dataos_operator.integrations.llamafactory as llamafactory
    from dataos_operator.integrations.llamafactory.collator import (
        DataOSLanguageModelingCollator,
    )

    assert "DataOSCallback" in llamafactory.__all__
    assert DataOSLanguageModelingCollator.__name__ == "DataOSLanguageModelingCollator"
