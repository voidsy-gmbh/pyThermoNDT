"""Fixtures for backend tests."""

from collections.abc import Generator
from dataclasses import dataclass
from typing import cast

import pytest

from pythermondt.io import BaseBackend, IOPathWrapper, LocalBackend, S3Backend


@dataclass
class TestConfig:
    """Configuration for backend testing."""

    backend_cls: type[BaseBackend]
    is_remote: bool


@dataclass
class TestFile:
    """Configuration for test files."""

    filename: str
    content: bytes


BACKENDS = [
    TestConfig(backend_cls=LocalBackend, is_remote=False),
]

TEST_FILES = {
    "sample.txt": b"test content",
    "data.bin": b"\x00\x01\x02\x03",
    "large.tiff": b"fake thermal data" * 100,
}


@pytest.fixture(params=BACKENDS, ids=lambda x: x.backend_cls.__name__)
def backend(request, tmp_path) -> Generator[tuple[BaseBackend, TestConfig], None, None]:
    """Create backend from configuration."""
    config = cast(TestConfig, request.param)

    # Setup the backend instance
    if config.backend_cls == LocalBackend:
        backend = LocalBackend(pattern=str(tmp_path))
    elif config.backend_cls == S3Backend:
        backend = S3Backend(bucket="ffg-bp", prefix="example2_writing_data/")
    else:
        raise NotImplementedError(f"Backend {config.backend_cls} not implemented in test fixture.")

    yield backend, config

    # Cleanup
    backend.close()


def _prepare_file(backend_instance: BaseBackend, name: str, content: bytes, tmp_path):
    file_path = tmp_path / name
    # Prepare file based on backend type
    if isinstance(backend_instance, LocalBackend):
        file_path = tmp_path / name
        file_path.write_bytes(content)
        return str(file_path)
    # Else write using backend
    else:
        backend_instance.write_file(IOPathWrapper(content), name)
        return name


@pytest.fixture(params=TEST_FILES.items(), ids=lambda x: x)
def test_file(request, backend, tmp_path):
    """Auto-create test files for the configured backend.

    Returns dict mapping logical names to actual paths.
    """
    name, content = request.param
    backend_instance, _ = backend
    file_path = _prepare_file(backend_instance, name, content, tmp_path)
    return file_path, content


@pytest.fixture
def test_files_all(backend, tmp_path):
    """All test files for bulk operations."""
    backend_instance, _ = backend
    return {name: _prepare_file(backend_instance, name, content, tmp_path) for name, content in TEST_FILES.items()}
