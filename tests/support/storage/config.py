from dataclasses import dataclass

from pythermondt.io import AzureBlobBackend, BaseBackend, LocalBackend, S3Backend
from pythermondt.readers import AzureBlobReader, BaseReader, LocalReader, S3Reader


@dataclass(frozen=True)
class TestConfig:
    """Configuration for backend and reader testing."""

    backend_cls: type[BaseBackend]
    reader_cls: type[BaseReader]
    scheme: str
    is_remote: bool


BACKENDS = [
    TestConfig(backend_cls=LocalBackend, reader_cls=LocalReader, scheme="file", is_remote=False),
    TestConfig(backend_cls=S3Backend, reader_cls=S3Reader, scheme="s3", is_remote=True),
    TestConfig(backend_cls=AzureBlobBackend, reader_cls=AzureBlobReader, scheme="az", is_remote=True),
]
