"""Fixtures for backend tests."""

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from pythermondt.io import BaseBackend, IOPathWrapper, LocalBackend, S3Backend


@dataclass
class TestConfig:
    """Configuration for backend testing."""

    backend_cls: type[BaseBackend]
    is_remote: bool


BACKENDS = [
    TestConfig(backend_cls=LocalBackend, is_remote=False),
]

# Single file test data
TEST_FILES = {
    "sample.txt": b"test content",
    "data.bin": b"\x00\x01\x02\x03",
    "large.tiff": b"fake thermal data" * 100,
}

# Multi-file scenarios for list/filter tests
FILE_SCENARIOS = {
    "mixed_types": {
        "sample.txt": b"test content",
        "data.bin": b"\x00\x01\x02\x03",
        "large.tiff": b"fake thermal data" * 100,
    },
    "single_type": {
        "thermal1.tiff": b"data1",
        "thermal2.tiff": b"data2",
        "thermal3.tiff": b"data3",
    },
    "many_files": {f"file{i:03d}.bin": b"x" * i for i in range(15)},
}


@pytest.fixture(params=BACKENDS, ids=lambda x: x.backend_cls.__name__)
def backend_config(request, tmp_path: Path) -> Generator[tuple[BaseBackend, TestConfig], None, None]:
    """Create backend from configuration."""
    config = cast(TestConfig, request.param)

    if config.backend_cls == LocalBackend:
        backend_instance = LocalBackend(pattern=str(tmp_path))
    elif config.backend_cls == S3Backend:
        backend_instance = S3Backend(bucket="test-bucket", prefix="test/")
    else:
        raise NotImplementedError(f"Backend {config.backend_cls} not implemented")

    yield backend_instance, config
    backend_instance.close()


def _prepare_file(backend_instance: BaseBackend, name: str, content: bytes, tmp_path: Path) -> str:
    """Prepare file and return path."""
    if isinstance(backend_instance, LocalBackend):
        file_path = tmp_path / name
        file_path.write_bytes(content)
        return str(file_path)
    else:
        backend_instance.write_file(IOPathWrapper(content), name)
        return name


@pytest.fixture(params=TEST_FILES.items(), ids=lambda x: x[0])
def test_file(request, backend_config, tmp_path: Path) -> tuple[str, bytes]:
    """Single test file - returns (path, content) tuple."""
    name, content = request.param
    backend_instance, _ = backend_config
    file_path = _prepare_file(backend_instance, name, content, tmp_path)
    return file_path, content


@pytest.fixture
def test_files_all(backend_config, tmp_path: Path) -> dict[str, str]:
    """All test files - returns dict of {name: path}."""
    backend_instance, _ = backend_config
    return {name: _prepare_file(backend_instance, name, content, tmp_path) for name, content in TEST_FILES.items()}


@pytest.fixture(params=FILE_SCENARIOS.items(), ids=lambda x: x[0])
def test_files_scenario(request, backend_config, tmp_path: Path) -> dict[str, str]:
    """Parameterized multi-file scenarios - returns dict of {name: path}."""
    _, files = request.param
    backend_instance, _ = backend_config
    return {name: _prepare_file(backend_instance, name, content, tmp_path) for name, content in sorted(files.items())}
