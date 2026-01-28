"""Fixtures for backend agnostic tests."""

import os
from collections.abc import Generator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws

from pythermondt.io import AzureBlobBackend, BaseBackend, IOPathWrapper, LocalBackend, S3Backend

from .utils import MockAzureBlob, TestConfig

BACKENDS = [
    TestConfig(backend_cls=LocalBackend, scheme="file", is_remote=False),
    TestConfig(backend_cls=S3Backend, scheme="s3", is_remote=True),
    TestConfig(backend_cls=AzureBlobBackend, scheme="az", is_remote=True),
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


@pytest.fixture()
def aws_creds():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture()
def s3_client(aws_creds):
    """Create mocked S3 client."""
    with mock_aws():
        yield boto3.client("s3")


@pytest.fixture()
def azure_mock():
    """Create mocked Azure Blob Storage."""
    mock_storage = MockAzureBlob()

    def make_blob_client(container: str, blob: str):
        mock_blob = MagicMock()

        # Mock download_blob - FIX: chunks() must return an iterator
        def download_blob():
            data = mock_storage.download_blob(container, blob)
            mock_stream = MagicMock()
            # Make chunks() return a callable that yields the data
            mock_stream.chunks = lambda: iter([data])
            return mock_stream

        mock_blob.download_blob = download_blob

        # Mock upload_blob
        def upload_blob(data, overwrite=True):
            content = data.read()
            mock_storage.upload_blob(container, blob, content)

        mock_blob.upload_blob = upload_blob

        # Mock get_blob_properties
        def get_blob_properties():
            if not mock_storage.blob_exists(container, blob):
                from azure.core.exceptions import ResourceNotFoundError

                raise ResourceNotFoundError("Blob not found")
            mock_props = MagicMock()
            mock_props.size = mock_storage.get_blob_size(container, blob)
            return mock_props

        mock_blob.get_blob_properties = get_blob_properties

        return mock_blob

    def make_container_client(container: str):
        mock_container = MagicMock()

        def list_blobs(name_starts_with=""):
            blob_names = mock_storage.list_blobs(container, name_starts_with)
            mock_blobs = []
            for name in blob_names:
                mock_blob = MagicMock()
                mock_blob.name = name
                mock_blobs.append(mock_blob)
            return mock_blobs

        mock_container.list_blobs = list_blobs

        return mock_container

    with patch("pythermondt.io.backends.azure_backend.BlobServiceClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.get_blob_client = make_blob_client
        mock_client.get_container_client = make_container_client
        mock_client.close = MagicMock()

        # Both from_connection_string and __init__ return the same mock
        mock_client_class.from_connection_string.return_value = mock_client
        mock_client_class.return_value = mock_client

        yield mock_storage


@pytest.fixture(params=BACKENDS, ids=lambda x: x.backend_cls.__name__)
def backend_config(request, tmp_path: Path, s3_client, azure_mock) -> Generator[tuple[BaseBackend, TestConfig]]:
    """Create backend from configuration."""
    config = cast(TestConfig, request.param)

    mock: AbstractContextManager | None = None
    if config.backend_cls == LocalBackend:
        backend_instance = LocalBackend(pattern=str(tmp_path))
    elif config.backend_cls == S3Backend:
        # Mock the AWS S3 service
        mock = mock_aws()
        mock.start()

        # Setup S3 bucket
        s3_client.create_bucket(Bucket="test-bucket")

        # Create S3 backend instance
        backend_instance = S3Backend(bucket="test-bucket", prefix="")
    elif config.backend_cls == AzureBlobBackend:
        azure_mock.create_container("test-container")
        backend_instance = AzureBlobBackend(
            account_url="https://test.blob.core.windows.net",
            container_name="test-container",
            prefix="",
            connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=fake==;EndpointSuffix=core.windows.net",
        )
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
    elif isinstance(backend_instance, AzureBlobBackend):
        backend_instance.write_file(IOPathWrapper(content), name)
        return f"az://test-container/{name}"
    elif isinstance(backend_instance, S3Backend):
        backend_instance.write_file(IOPathWrapper(content), name)
        return "s3://test-bucket/" + name
    raise NotImplementedError("Unsupported backend for file preparation.")


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
