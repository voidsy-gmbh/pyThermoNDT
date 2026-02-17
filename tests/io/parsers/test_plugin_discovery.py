from collections.abc import Generator
from pathlib import Path

import pytest

import pythermondt.io.parsers as parsers
from pythermondt.data import DataContainer
from pythermondt.io.parsers import EdevisParser, HDF5Parser, SimulationParser
from pythermondt.io.utils import IOPathWrapper


class SuccessfulPluginParser:
    supported_extensions = (".good",)

    @staticmethod
    def parse(data: IOPathWrapper) -> DataContainer:
        return DataContainer()


class _FakeEntryPoint:
    def __init__(self, name: str, module: str, parser_cls: type[SuccessfulPluginParser]):
        self.name = name
        self.module = module
        self._parser_cls = parser_cls

    def load(self) -> type[SuccessfulPluginParser]:
        return self._parser_cls


class _FailingEntryPoint:
    def __init__(self, name: str, module: str, error_message: str):
        self.name = name
        self.module = module
        self._error_message = error_message

    def load(self) -> type[SuccessfulPluginParser]:
        raise RuntimeError(self._error_message)


@pytest.fixture(autouse=True)
def clear_registry_cache() -> Generator[None]:
    """Ensure parser registry cache does not leak between tests."""
    parsers._get_registry.cache_clear()
    yield
    parsers._get_registry.cache_clear()


def test_load_parser_plugins(monkeypatch, caplog):
    """Test plugin loading emits warning for failures and debug log for successes."""

    def fake_entry_points(*, group: str):
        assert group == "pythermondt.parsers"
        return [
            _FakeEntryPoint("good-plugin", "tests.fake_plugin", SuccessfulPluginParser),
            _FailingEntryPoint("broken-plugin", "tests.broken_plugin", "plugin import failed"),
        ]

    monkeypatch.setattr(parsers, "entry_points", fake_entry_points)

    with caplog.at_level("DEBUG", logger="pythermondt.io.parsers"):
        msg = "Loaded parser plugin 'good-plugin' from 'tests.fake_plugin'"
        with pytest.warns(UserWarning, match=msg) as caught:
            plugins = parsers._load_parser_plugins()

    assert plugins == (SuccessfulPluginParser,)
    assert "Loaded parser plugin 'good-plugin' from 'tests.fake_plugin'" in caplog.messages
    assert len(caught) == 1
    assert Path(caught[0].filename).name == Path(__file__).name
    assert caught[0].lineno > 0


def test_get_all_parsers(monkeypatch):
    """Test registry includes built-in parsers plus successfully loaded plugins."""

    def fake_entry_points(*, group: str):
        assert group == "pythermondt.parsers"
        return [
            _FakeEntryPoint("good-plugin", "tests.fake_plugin", SuccessfulPluginParser),
            _FailingEntryPoint("broken-plugin", "tests.broken_plugin", "plugin import failed"),
        ]

    monkeypatch.setattr(parsers, "entry_points", fake_entry_points)

    with pytest.warns(UserWarning, match="Failed to load parser plugin 'broken-plugin': plugin import failed"):
        all_parsers = parsers.get_all_parsers()

    assert HDF5Parser in all_parsers
    assert SimulationParser in all_parsers
    assert EdevisParser in all_parsers
    assert SuccessfulPluginParser in all_parsers
    assert parsers.find_parser_for_extension(".good") is SuccessfulPluginParser
    assert parsers.find_parser_for_extension("good") is SuccessfulPluginParser


def test_get_all_parsers_builtins(monkeypatch):
    """Test built-in parsers remain available even if all plugins fail to load."""

    def fake_entry_points(*, group: str):
        assert group == "pythermondt.parsers"
        return [
            _FailingEntryPoint("broken-plugin-a", "tests.broken_plugin_a", "failure a"),
            _FailingEntryPoint("broken-plugin-b", "tests.broken_plugin_b", "failure b"),
        ]

    monkeypatch.setattr(parsers, "entry_points", fake_entry_points)

    with pytest.warns(UserWarning, match="Failed to load parser plugin") as caught:
        all_parsers = parsers.get_all_parsers()

    assert len(caught) == 2
    assert all_parsers == (HDF5Parser, SimulationParser, EdevisParser)
