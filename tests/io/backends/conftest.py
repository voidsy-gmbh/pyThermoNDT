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
    TestConfig(backend_cls=S3Backend, is_remote=True),
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


@pytest.fixture(params=TEST_FILES.items(), ids=lambda x: x)
def test_file(request, backend, tmp_path):
    """Auto-create test files for the configured backend.

    Returns dict mapping logical names to actual paths.
    """
    name, content = request.param
    backend, _ = backend

    if isinstance(backend, LocalBackend):
        file_path = tmp_path / name
        with open(file_path, "wb") as f:
            f.write(content)
        file_path = str(file_path)
    # Else write using backend
    else:
        file_path = name
        backend.write_file(IOPathWrapper(content), file_path)

    return file_path, content
