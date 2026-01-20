from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any

import pytest

from pythermondt.io import BaseBackend, IOPathWrapper, LocalBackend


@dataclass
class TestConfig:
    """Configuration for backend testing."""

    backend_cls: type[BaseBackend]
    is_remote: bool
    init_kwargs: dict[str, Any]
    requires_mock: bool = False
    mock_setup: Callable | None = None


BACKEND_CONFIGS = [
    TestConfig(
        backend_cls=LocalBackend,
        is_remote=False,
        init_kwargs={"pattern": "tests/assets", "recursive": True},
        requires_mock=False,
    ),
]


@pytest.fixture(params=BACKEND_CONFIGS, ids=lambda x: x.backend_cls.__name__)
def configured_backend(request) -> Generator[tuple[BaseBackend, TestConfig], None, None]:
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


@pytest.fixture
def test_files(configured_backend, tmp_path):
    """Auto-create test files for the configured backend.

    Returns dict mapping logical names to actual paths.
    """
    backend = configured_backend

    files = {}
    test_data = {
        "sample.txt": b"test content",
        "data.bin": b"\x00\x01\x02\x03",
        "large.tiff": b"fake thermal data" * 100,
    }

    for filename, content in test_data.items():
        # For local Backend: create file in tmp path
        if isinstance(backend, LocalBackend):
            file_path = tmp_path / filename
            with open(file_path, "wb") as f:
                f.write(content)
            file_path = str(file_path)
        # Else write using backend
        else:
            file_path = filename
            backend.write_file(IOPathWrapper(content), file_path)

        files[filename] = file_path

    return files
