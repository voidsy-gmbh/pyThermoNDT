"""Fixtures for backend agnostic tests."""

import os
from collections.abc import Generator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import cast

import boto3
import pytest
from moto import mock_aws

from pythermondt.io import BaseBackend, IOPathWrapper, LocalBackend, S3Backend

from .utils import TestConfig

BACKENDS = [
    TestConfig(backend_cls=LocalBackend, is_remote=False),
    TestConfig(backend_cls=S3Backend, is_remote=True),
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

    mock: AbstractContextManager | None = None
    if config.backend_cls == LocalBackend:
        backend_instance = LocalBackend(pattern=str(tmp_path))
    elif config.backend_cls == S3Backend:
        # Mock the AWS S3 service
        os.environ["AWS_ACCESS_KEY_ID"] = "testing"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
        os.environ["AWS_SECURITY_TOKEN"] = "testing"
        os.environ["AWS_SESSION_TOKEN"] = "testing"
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
        mock = mock_aws()
        mock.start()

        # Setup S3 bucket
        s3_client = boto3.client("s3")
        s3_client.create_bucket(Bucket="test-bucket")

        # Create S3 backend instance
        backend_instance = S3Backend(bucket="test-bucket", prefix="test/")
    else:
        raise NotImplementedError(f"Backend {config.backend_cls} not implemented")

    yield backend_instance, config
    backend_instance.close()
    if mock:
        mock.stop()


def _prepare_file(backend_instance: BaseBackend, name: str, content: bytes, tmp_path: Path) -> str:
    """Prepare file and return path."""
    if isinstance(backend_instance, LocalBackend):
        file_path = tmp_path / name
        file_path.write_bytes(content)
        return file_path.as_uri()
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
