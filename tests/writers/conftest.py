from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from pythermondt.data import DataContainer
from pythermondt.writers import AzureBlobWriter, BaseWriter, LocalWriter, S3Writer
from tests.support import storage
from tests.support.storage.config import StorageTestContext

from ..utils import make_container


@dataclass(frozen=True)
class WriterTestContext:
    """Writer fixture context with per-backend writer and read-back helpers."""

    writer: BaseWriter
    config: storage.TestConfig
    read_path: Callable[[str], str]


@pytest.fixture
def writer_config(storage_backend: StorageTestContext, tmp_path: Path) -> Generator[WriterTestContext]:
    """Create writer + read_path helper, parametrized over all storage backends."""
    config = storage_backend.config

    writer: BaseWriter

    if config.writer_cls == LocalWriter:
        dest = tmp_path / "dest"
        writer = LocalWriter(str(dest))

        def read_path(filename: str) -> str:
            if not filename.endswith(".hdf5"):
                filename += ".hdf5"
            return str(dest / filename)

    elif config.writer_cls == S3Writer:
        writer = S3Writer(bucket="test-bucket", prefix="")

        def read_path(filename: str) -> str:
            if not filename.endswith(".hdf5"):
                filename += ".hdf5"
            return f"s3://test-bucket/{filename}"

    elif config.writer_cls == AzureBlobWriter:
        writer = AzureBlobWriter(
            account_url="https://test.blob.core.windows.net",
            container_name="test-container",
            prefix="",
            connection_string=(
                "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=fake==;EndpointSuffix=core.windows.net"
            ),
        )

        def read_path(filename: str) -> str:
            if not filename.endswith(".hdf5"):
                filename += ".hdf5"
            return f"az://test-container/{filename}"

    else:
        raise NotImplementedError(f"Writer {config.writer_cls} not implemented")

    yield WriterTestContext(writer=writer, config=config, read_path=read_path)

    writer.backend.close()


@pytest.fixture
def test_container() -> DataContainer:
    """Return a small DataContainer for writer round-trip tests."""
    c = make_container(("/Data", "values", torch.tensor([[1.0, 2.0], [3.0, 4.0]])))
    c.add_attribute("/Data", "description", "test data")
    return c
