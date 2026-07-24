from collections.abc import Callable
from dataclasses import dataclass

from pythermondt.io import AzureBlobBackend, BaseBackend, LocalBackend, S3Backend
from pythermondt.readers import AzureBlobReader, BaseReader, LocalReader, S3Reader
from pythermondt.writers import AzureBlobWriter, BaseWriter, LocalWriter, S3Writer


@dataclass(frozen=True)
class TestConfig:
    """Configuration for backend, reader, and writer testing."""

    backend_cls: type[BaseBackend]
    reader_cls: type[BaseReader]
    writer_cls: type[BaseWriter]
    scheme: str
    is_remote: bool


@dataclass(frozen=True)
class StorageTestContext:
    """Shared storage fixture context used by reader and writer conftests."""

    backend: BaseBackend
    config: TestConfig
    prepare_file: Callable[[str, bytes], str]


BACKENDS = [
    TestConfig(LocalBackend, LocalReader, LocalWriter, "file", False),
    TestConfig(S3Backend, S3Reader, S3Writer, "s3", True),
    TestConfig(AzureBlobBackend, AzureBlobReader, AzureBlobWriter, "az", True),
]
