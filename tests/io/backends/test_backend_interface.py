from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from pythermondt.io import BaseBackend, IOPathWrapper, LocalBackend


@dataclass
class BackendConfig:
    """Configuration for backend testing."""

    backend_cls: type[BaseBackend]
    is_remote: bool
    init_kwargs: dict[str, Any]
    requires_mock: bool = False
    mock_setup: Callable | None = None


BACKEND_CONFIGS = [
    BackendConfig(
        backend_cls=LocalBackend,
        is_remote=False,
        init_kwargs={"pattern": "tests/assets", "recursive": True},
        requires_mock=False,
    ),
]


@pytest.fixture(params=BACKEND_CONFIGS, ids=lambda x: x.backend_cls.__name__)
def configured_backend(request) -> Generator[tuple[BaseBackend, BackendConfig], None, None]:
    """Create backend from configuration."""
    config = request.param

    # Setup mock if needed
    mock_context = None
    if config.requires_mock and config.mock_setup:
        mock_context = config.mock_setup()
        mock_context.__enter__()

    # Add tmp_path to kwargs if backend needs it
    kwargs = config.init_kwargs.copy()
    backend = config.backend_cls(**kwargs)

    yield backend, config

    # Cleanup
    backend.close()
    if mock_context:
        mock_context.__exit__(None, None, None)


def _setup_temp_file(path: Path) -> str:
    """Helper to create a temporary file for testing and return its path as a string."""
    with open(path, "w") as f:
        f.write("test content")
    return str(path)


def test_remote_source(configured_backend):
    backend, config = configured_backend
    assert backend.remote_source == config.is_remote


def test_read_file(tmp_path: Path, configured_backend):
    backend, _ = configured_backend
    path = _setup_temp_file(tmp_path / "sample.txt")
    result = backend.read_file(path)
    assert isinstance(result, IOPathWrapper)
    assert result.file_obj.read().decode() == "test content"
