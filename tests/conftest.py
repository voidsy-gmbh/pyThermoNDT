from pathlib import Path

import boto3
import numpy as np
import pytest
import torch
from moto import mock_aws

from pythermondt import DataContainer, LocalReader, S3Reader
from pythermondt.transforms import Compose, RandomThermoTransform, ThermoTransform
from tests.support.storage import BACKENDS, StorageTestContext


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


@pytest.fixture(params=BACKENDS, ids=lambda backend: backend.__name__)
def storage_context(request: pytest.FixtureRequest, tmp_path: Path, fake_aws_creds):
    """Create a managed context for each general storage backend.

    Args:
        request: Pytest request whose ``param`` contains the selected backend class. Tests can override the default
            ``BACKENDS`` parametrization with indirect fixture parameters when only a specific backend is relevant.
        tmp_path: Temporary directory used by the local storage backend.
        fake_aws_creds: Fixture providing dummy AWS credentials for the mocked S3 backend.
    """
    with StorageTestContext(request.param, tmp_path) as context:
        yield context


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
