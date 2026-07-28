from collections.abc import Callable, Mapping
from contextlib import ExitStack
from pathlib import Path

import boto3
from moto import mock_aws

from pythermondt.io import AzureBlobBackend, BaseBackend, FileInfo, IOPathWrapper, LocalBackend, S3Backend
from pythermondt.io.parsers import BaseParser
from pythermondt.readers import AzureBlobReader, BaseReader, LocalReader, S3Reader
from pythermondt.writers import AzureBlobWriter, BaseWriter, LocalWriter, S3Writer

from .azure import mocked_azure_blob_storage
from .parsers import PlainTextParser

AWS_BUCKET = "test-bucket"
AZURE_CONTAINER = "test-container"
AZURE_ACCOUNT_URL = "https://test.blob.core.windows.net"
AZURE_CONNECTION_STRING = (
    "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=fake==;EndpointSuffix=core.windows.net"
)


class StorageTestContext:
    """Own a storage backend, its mocks, and all related test resources."""

    def __init__(self, backend_type: type[BaseBackend], root: Path):
        self._backend_type = backend_type
        self._root = root
        self._destination = root / "dest"
        self._stack = ExitStack()
        self._started = False
        self._backend: BaseBackend | None = None

    def __enter__(self) -> "StorageTestContext":
        self._started = True
        try:
            # Start the backend-specific mock before creating any clients.
            if self._backend_type is S3Backend:
                self._stack.enter_context(mock_aws())
                boto3.client("s3").create_bucket(Bucket=AWS_BUCKET)
            elif self._backend_type is AzureBlobBackend:
                azure = self._stack.enter_context(mocked_azure_blob_storage())
                azure.create_container(AZURE_CONTAINER)

            # Keep one primary backend for file preparation and direct backend tests.
            self._backend = self.make_backend()
            return self
        except BaseException:
            # __exit__ is not called when __enter__ fails, so clean up explicitly.
            self._stack.close()
            self._started = False
            raise

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._stack.close()

    @property
    def backend(self) -> BaseBackend:
        """Return the primary backend."""
        if self._backend is None:
            raise RuntimeError("StorageTestContext must be entered before use.")
        return self._backend

    def make_backend(self) -> BaseBackend:
        """Create and immediately register a backend for cleanup."""
        if not self._started:
            raise RuntimeError("StorageTestContext must be entered before use.")
        if self._backend_type is LocalBackend:
            backend = LocalBackend(pattern=str(self._root))
        elif self._backend_type is S3Backend:
            backend = S3Backend(bucket=AWS_BUCKET, prefix="")
        elif self._backend_type is AzureBlobBackend:
            backend = AzureBlobBackend(
                account_url=AZURE_ACCOUNT_URL,
                container_name=AZURE_CONTAINER,
                prefix="",
                connection_string=AZURE_CONNECTION_STRING,
            )
        else:
            raise NotImplementedError(f"Backend {self._backend_type} not implemented.")

        # ExitStack closes resources in reverse creation order.
        self._stack.callback(backend.close)
        return backend

    def make_reader(
        self,
        parser: type[BaseParser] | None = PlainTextParser,
        num_files: int | None = None,
        cache_files: bool = True,
        file_filter: Callable[[FileInfo], bool] | None = None,
    ) -> BaseReader:
        """Create a reader and immediately register its lazy backend."""
        if self._backend_type is LocalBackend:
            reader = LocalReader(
                pattern=str(self._root),
                parser=parser,
                num_files=num_files,
                cache_files=cache_files,
                file_filter=file_filter,
            )
        elif self._backend_type is S3Backend:
            reader = S3Reader(
                bucket=AWS_BUCKET,
                prefix="",
                parser=parser,
                num_files=num_files,
                cache_files=cache_files,
                file_filter=file_filter,
            )
        elif self._backend_type is AzureBlobBackend:
            reader = AzureBlobReader(
                account_url=AZURE_ACCOUNT_URL,
                container_name=AZURE_CONTAINER,
                prefix="",
                connection_string=AZURE_CONNECTION_STRING,
                parser=parser,
                num_files=num_files,
                cache_files=cache_files,
                file_filter=file_filter,
            )
        else:
            raise NotImplementedError(f"Reader for {self._backend_type} not implemented.")

        # Access the lazy backend now so teardown never creates a new resource.
        self._stack.callback(reader.backend.close)
        return reader

    def make_writer(self) -> BaseWriter:
        """Create a writer and immediately register its lazy backend."""
        if self._backend_type is LocalBackend:
            writer = LocalWriter(str(self._destination))
        elif self._backend_type is S3Backend:
            writer = S3Writer(bucket=AWS_BUCKET, prefix="")
        elif self._backend_type is AzureBlobBackend:
            writer = AzureBlobWriter(
                account_url=AZURE_ACCOUNT_URL,
                container_name=AZURE_CONTAINER,
                prefix="",
                connection_string=AZURE_CONNECTION_STRING,
            )
        else:
            raise NotImplementedError(f"Writer for {self._backend_type} not implemented.")

        # Access the lazy backend now so teardown never creates a new resource.
        self._stack.callback(writer.backend.close)
        return writer

    def prepare_file(self, name: str, content: bytes) -> str:
        """Write one file and return its canonical URI."""
        path = self._source_path(name)
        self.backend.write_file(IOPathWrapper(content), path)
        return path

    def prepare_files(self, files: Mapping[str, bytes]) -> dict[str, str]:
        """Write files and return their canonical URIs by name."""
        return {name: self.prepare_file(name, content) for name, content in files.items()}

    def canonical_path(self, name: str, destination: bool = False) -> str:
        """Return the canonical path used to read or assert a stored file."""
        if self._backend_type is LocalBackend and destination:
            return (self._destination / name).resolve().as_uri()
        return self._source_path(name)

    def _source_path(self, name: str) -> str:
        # Expose one canonical URI format across all storage implementations.
        if self._backend_type is LocalBackend:
            return (self._root / name).resolve().as_uri()
        if self._backend_type is S3Backend:
            return f"s3://{AWS_BUCKET}/{name}"
        return f"az://{AZURE_CONTAINER}/{name}"


BACKENDS = (LocalBackend, S3Backend, AzureBlobBackend)
