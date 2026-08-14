"""本地插件目录(自定义算子)加载机制的测试。"""

from __future__ import annotations

import textwrap

from dataos_operator.core.catalog import (
    PLUGIN_ENV_VAR,
    list_operator_specs,
    resolve_plugin_dirs,
)

PLUGIN_SRC = textwrap.dedent(
    '''
    import json
    from collections.abc import Sequence

    from dataos_operator.core.contracts import Dataset, SelectorOperator


    class FunctionCallFilter(SelectorOperator):
        """Keep only rows whose ``output`` is a valid function-call list.

        Implementation: json.loads(output) must be a list of {"name","arguments"} dicts.
        Dependency: pure-rule operator.
        """

        operator_name = "demo_call_filter"

        def __init__(self, *, output_key: str = "output", **kwargs):
            super().__init__(output_key=output_key, **kwargs)
            self.output_key = output_key

        def select(self, data):
            rows = [r for ds in data for r in ds]
            return [r for r in rows if self._ok(self.get_field(r, self.output_key, ""))]

        @staticmethod
        def _ok(out):
            if not isinstance(out, str):
                return False
            try:
                calls = json.loads(out)
            except (ValueError, TypeError):
                return False
            return isinstance(calls, list) and all(
                isinstance(c, dict) and "name" in c and "arguments" in c for c in calls
            )
    '''
)


def _write_plugin(directory, name="function_call_filter.py", src=PLUGIN_SRC):
    (directory / name).write_text(src, encoding="utf-8")


def test_plugin_dir_adds_operator(tmp_path):
    _write_plugin(tmp_path)
    base = list_operator_specs(include_plugins=False)
    withp = list_operator_specs(plugin_dirs=[str(tmp_path)])
    names = {spec.name for spec in withp}
    assert "demo_call_filter" in names
    assert len(withp) == len(base) + 1
    spec = next(s for s in withp if s.name == "demo_call_filter")
    assert spec.kind == "selector"
    assert spec.description.startswith("Keep only rows")


def test_plugin_via_env(tmp_path, monkeypatch):
    _write_plugin(tmp_path)
    monkeypatch.setenv(PLUGIN_ENV_VAR, str(tmp_path))
    names = {spec.name for spec in list_operator_specs()}
    assert "demo_call_filter" in names
    assert resolve_plugin_dirs() == [tmp_path]


def test_include_plugins_false_ignores_env(tmp_path, monkeypatch):
    _write_plugin(tmp_path)
    monkeypatch.setenv(PLUGIN_ENV_VAR, str(tmp_path))
    names = {spec.name for spec in list_operator_specs(include_plugins=False)}
    assert "demo_call_filter" not in names


def test_bad_plugin_is_skipped(tmp_path):
    (tmp_path / "broken.py").write_text("def (this is not valid python", encoding="utf-8")
    _write_plugin(tmp_path)
    names = {spec.name for spec in list_operator_specs(plugin_dirs=[str(tmp_path)])}
    assert "demo_call_filter" in names  # 坏插件被跳过,好插件照常


def test_underscore_files_not_scanned(tmp_path):
    (tmp_path / "_helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    _write_plugin(tmp_path)
    # 不应因下划线辅助文件报错,且算子仍被发现
    names = {spec.name for spec in list_operator_specs(plugin_dirs=[str(tmp_path)])}
    assert "demo_call_filter" in names


def test_builtin_name_collision_is_ignored(tmp_path):
    collide = PLUGIN_SRC.replace(
        'operator_name = "demo_call_filter"',
        'operator_name = "length_filter"',  # 与内置算子同名
    )
    _write_plugin(tmp_path, name="fake_length_filter.py", src=collide)
    specs = list_operator_specs(plugin_dirs=[str(tmp_path)])
    length_specs = [s for s in specs if s.name == "length_filter"]
    assert len(length_specs) == 1  # 只有一个 length_filter
    # 保留的是内置那个,不是插件(插件同名被忽略)
    assert length_specs[0].operator_class.startswith("dataos_operator.operators")
