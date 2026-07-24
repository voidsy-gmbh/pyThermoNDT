from collections.abc import Generator
from pathlib import Path
from typing import cast

import boto3
import numpy as np
import pytest
import torch
from moto import mock_aws

from pythermondt import DataContainer, LocalReader, S3Reader
from pythermondt.io import AzureBlobBackend, BaseBackend, LocalBackend, S3Backend
from pythermondt.transforms import Compose, RandomThermoTransform, ThermoTransform
from tests.support.storage import (
    BACKENDS,
    StorageTestContext,
    TestConfig,
    mocked_azure_blob_storage,
    prepare_file,
)


class AltReader(LocalReader):
    """LocalReader subclass used in tests to simulate a different reader type."""


@pytest.fixture()
def fake_aws_creds(monkeypatch):
    """Mocked AWS Credentials for moto."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture()
def s3_client(fake_aws_creds):
    """Create mocked S3 client."""
    with mock_aws():
        yield boto3.client("s3")


@pytest.fixture(params=BACKENDS, ids=lambda x: x.backend_cls.__name__)
def storage_backend(request, tmp_path: Path, s3_client) -> Generator[StorageTestContext]:
    """Create backend + prepare_file helper, parametrized over all storage backends."""
    config = cast(TestConfig, request.param)

    backend: BaseBackend
    azure_ctx = None
    if config.backend_cls == LocalBackend:
        backend = LocalBackend(pattern=str(tmp_path))
    elif config.backend_cls == S3Backend:
        s3_client.create_bucket(Bucket="test-bucket")
        backend = S3Backend(bucket="test-bucket", prefix="")
    elif config.backend_cls == AzureBlobBackend:
        azure_ctx = mocked_azure_blob_storage()
        azure_storage = azure_ctx.__enter__()
        azure_storage.create_container("test-container")
        backend = AzureBlobBackend(
            account_url="https://test.blob.core.windows.net",
            container_name="test-container",
            prefix="",
            connection_string=(
                "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=fake==;EndpointSuffix=core.windows.net"
            ),
        )
    else:
        raise NotImplementedError(f"Backend {config.backend_cls} not implemented")

    def _prepare_file(name: str, content: bytes) -> str:
        """Prepare one file in the backend's storage and return its canonical URI."""
        return prepare_file(backend, name, content, tmp_path)

    yield StorageTestContext(backend=backend, config=config, prepare_file=_prepare_file)

    backend.close()
    if azure_ctx is not None:
        azure_ctx.__exit__(None, None, None)


@pytest.fixture
def sample_tensor():
    """Basic tensor fixture available to all tests."""
    return torch.tensor([[1, 2], [3, 4]])


@pytest.fixture
def sample_tensor2():
    """Basic tensor fixture available to all tests."""
    return torch.tensor([[5, 6], [7, 8]])


@pytest.fixture
def sample_empty_tensor():
    """Empty tensor fixture available to all tests."""
    return torch.empty(0)


@pytest.fixture
def sample_eye_tensor():
    """Identity tensor fixture available to all tests."""
    return torch.eye(3)


@pytest.fixture
def sample_ndarray():
    """Basic ndarray fixture available to all tests."""
    return np.array([[1, 2], [3, 4]])


@pytest.fixture
def sample_ndarray2():
    """Basic ndarray fixture available to all tests."""
    return np.array([[5, 6], [7, 8]])


@pytest.fixture
def sample_empty_ndarray():
    """Empty ndarray fixture available to all tests."""
    return np.empty(0)


@pytest.fixture
def sample_eye_ndarray():
    """Identity ndarray fixture available to all tests."""
    return np.eye(3)


@pytest.fixture
def localreader_no_files():
    """Fixture for a reader that has no files."""
    return LocalReader(pattern="MadeUpPattern")


@pytest.fixture
def altreader_no_files():
    """Fixture for an AltReader that has no files."""
    return AltReader(pattern="MadeUpPattern")


@pytest.fixture
def localreader_with_file():
    """Fixture for a reader that has a single file."""
    return LocalReader(pattern="./tests/assets/integration/simulation/source1.mat")


@pytest.fixture
def localreader_with_glob():
    """Fixture for a reader that has files."""
    return LocalReader(pattern="./tests/assets/integration/simulation/*.mat")


@pytest.fixture
def localreader_with_directory():
    """Fixture for a reader that has files."""
    return LocalReader(pattern="./tests/assets/integration/simulation/")


@pytest.fixture()
def s3reader_with_file(s3_client):
    """Fixture for an S3 reader that has a single file."""
    # Ensure the bucket exists
    s3_client.create_bucket(Bucket="test-bucket")

    # Upload a test file to the bucket
    s3_client.upload_file(
        Filename="./tests/assets/integration/simulation/source1.mat",
        Bucket="test-bucket",
        Key="source1.mat",
    )

    yield S3Reader(bucket="test-bucket", prefix="")


@pytest.fixture
def sample_transform():
    """Create a simple ThermoTransform that adds an attribute."""

    class SimpleTransform(ThermoTransform):
        """A simple transform that increments a 'transformed' attribute."""

        def __init__(self, value: str):
            super().__init__()
            self.value = value

        def forward(self, container: DataContainer) -> DataContainer:
            if "transformed" in container.get_all_attributes("/MetaData"):
                v = container.get_attribute("/MetaData", "transformed")
                assert isinstance(v, list)
                v = [*v, self.value]
                container.update_attribute("/MetaData", "transformed", v)
            else:
                container.add_attribute("/MetaData", "transformed", [self.value])
            return container

    return SimpleTransform


@pytest.fixture
def sample_random_transform():
    """Create a simple ThermoTransform that adds an attribute."""

    class SimpleRandomTransform(RandomThermoTransform):
        """A simple transform that increments a 'transformed' attribute."""

        def __init__(self):
            super().__init__()

        def forward(self, container: DataContainer) -> DataContainer:
            if "transformed_random" in container.get_all_attributes("/MetaData"):
                v = container.get_attribute("/MetaData", "transformed_random")
                assert isinstance(v, list)
                v = [*v, self.value]
                container.update_attribute("/MetaData", "transformed_random", v)
            else:
                container.add_attribute("/MetaData", "transformed_random", [torch.rand(1).item()])
            return container

    return SimpleRandomTransform


@pytest.fixture
def sample_pipeline(sample_transform: type[ThermoTransform], sample_random_transform: type[RandomThermoTransform]):
    """Create a transform pipeline with multiple levels of transforms."""
    return Compose(
        [
            sample_transform("base_level"),
            sample_transform("first_level"),
            sample_transform("second_level"),
            sample_random_transform(),
            sample_transform("third_level"),
            sample_transform("fourth_level"),
        ]
    )
