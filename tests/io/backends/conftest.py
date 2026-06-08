from collections.abc import Generator
from pathlib import Path
from typing import cast

import pytest
from moto import mock_aws
from moto.core.models import MockAWS

from pythermondt.io import AzureBlobBackend, BaseBackend, LocalBackend, S3Backend
from tests.support.storage import (
    BACKENDS,
    FILE_SCENARIOS,
    TEST_FILES,
    TestConfig,
    mocked_azure_blob_storage,
    prepare_file,
)


@pytest.fixture()
def azure_mock():
    """Create mocked Azure Blob Storage."""
    with mocked_azure_blob_storage() as mock_storage:
        yield mock_storage


@pytest.fixture(params=BACKENDS, ids=lambda x: x.backend_cls.__name__)
def backend_config(request, tmp_path: Path, s3_client, azure_mock) -> Generator[tuple[BaseBackend, TestConfig]]:
    """Create backend from configuration."""
    config = cast(TestConfig, request.param)

    mock: MockAWS | None = None
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


@pytest.fixture(params=TEST_FILES.items(), ids=lambda x: x[0])
def test_file(request, backend_config, tmp_path: Path) -> tuple[str, bytes]:
    """Single test file - returns (path, content) tuple."""
    name, content = request.param
    backend_instance, _ = backend_config
    file_path = prepare_file(backend_instance, name, content, tmp_path)
    return file_path, content


@pytest.fixture
def test_files_all(backend_config, tmp_path: Path) -> dict[str, str]:
    """All test files - returns dict of {name: path}."""
    backend_instance, _ = backend_config
    return {name: prepare_file(backend_instance, name, content, tmp_path) for name, content in TEST_FILES.items()}


@pytest.fixture(params=FILE_SCENARIOS.items(), ids=lambda x: x[0])
def test_files_scenario(request, backend_config, tmp_path: Path) -> dict[str, str]:
    """Parameterized multi-file scenarios - returns dict of {name: path}."""
    _, files = request.param
    backend_instance, _ = backend_config
    return {name: prepare_file(backend_instance, name, content, tmp_path) for name, content in sorted(files.items())}
