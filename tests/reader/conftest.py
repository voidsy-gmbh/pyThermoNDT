from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import pytest

from pythermondt.io import AzureBlobBackend, BaseBackend, FileInfo, LocalBackend, S3Backend
from pythermondt.io.parsers import BaseParser
from pythermondt.readers import AzureBlobReader, BaseReader, LocalReader, S3Reader
from tests.support import storage


class ReaderFactory(Protocol):
    """Factory signature for creating equivalent readers with test-specific options."""

    def __call__(
        self,
        parser: type[BaseParser] | None = ...,
        num_files: int | None = ...,
        cache_files: bool = ...,
        file_filter: Callable[[FileInfo], bool] | None = ...,
    ) -> BaseReader: ...


@dataclass(frozen=True)
class ReaderTestContext:
    """Reader fixture context with backend-specific setup hidden behind callbacks."""

    reader: BaseReader
    config: storage.TestConfig
    make_reader: ReaderFactory
    prepare_file: Callable[[str, bytes], str]


@dataclass(frozen=True)
class ReaderTestData:
    """Prepared reader files and their source contents."""

    context: ReaderTestContext
    files: dict[str, str]
    contents: dict[str, str]

    @property
    def reader(self) -> BaseReader:
        """Return the reader under test."""
        return self.context.reader

    @property
    def expected_files(self) -> list[str]:
        """Return the parser-supported files in deterministic reader order."""
        return [self.files[name] for name in sorted(self.contents) if name.endswith(".test")]


@pytest.fixture()
def azure_mock():
    """Create mocked Azure Blob Storage."""
    with storage.mocked_azure_blob_storage() as mock_storage:
        yield mock_storage


@pytest.fixture(params=storage.BACKENDS, ids=lambda x: x.reader_cls.__name__)
def reader_config(request, tmp_path: Path, s3_client, azure_mock) -> Generator[ReaderTestContext]:
    """Create a reader and a storage setup callback from configuration."""
    config = cast(storage.TestConfig, request.param)

    backend: BaseBackend
    if config.reader_cls == LocalReader:
        backend = LocalBackend(pattern=str(tmp_path))

        def make_reader(
            parser: type[BaseParser] | None = storage.PlainTextParser,
            num_files: int | None = None,
            cache_files: bool = True,
            file_filter: Callable[[FileInfo], bool] | None = None,
        ) -> BaseReader:
            return LocalReader(
                pattern=str(tmp_path),
                num_files=num_files,
                parser=parser,
                cache_files=cache_files,
                file_filter=file_filter,
            )

    elif config.reader_cls == S3Reader:
        s3_client.create_bucket(Bucket="test-bucket")
        backend = S3Backend(bucket="test-bucket", prefix="")

        def make_reader(
            parser: type[BaseParser] | None = storage.PlainTextParser,
            num_files: int | None = None,
            cache_files: bool = True,
            file_filter: Callable[[FileInfo], bool] | None = None,
        ) -> BaseReader:
            return S3Reader(
                bucket="test-bucket",
                prefix="",
                num_files=num_files,
                parser=parser,
                cache_files=cache_files,
                file_filter=file_filter,
            )

    elif config.reader_cls == AzureBlobReader:
        azure_mock.create_container("test-container")
        backend = AzureBlobBackend(
            account_url="https://test.blob.core.windows.net",
            container_name="test-container",
            prefix="",
            connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=fake==;EndpointSuffix=core.windows.net",
        )

        def make_reader(
            parser: type[BaseParser] | None = storage.PlainTextParser,
            num_files: int | None = None,
            cache_files: bool = True,
            file_filter: Callable[[FileInfo], bool] | None = None,
        ) -> BaseReader:
            return AzureBlobReader(
                account_url="https://test.blob.core.windows.net",
                container_name="test-container",
                prefix="",
                connection_string=(
                    "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=fake==;EndpointSuffix=core.windows.net"
                ),
                num_files=num_files,
                parser=parser,
                cache_files=cache_files,
                file_filter=file_filter,
            )

    else:
        raise NotImplementedError(f"Reader {config.reader_cls} not implemented")

    def prepare_reader_file(name: str, content: bytes) -> str:
        """Prepare one file in the reader's storage and return its canonical URI."""
        return storage.prepare_file(backend, name, content, tmp_path)

    reader = make_reader()
    yield ReaderTestContext(
        reader=reader,
        config=config,
        make_reader=make_reader,
        prepare_file=prepare_reader_file,
    )

    backend.close()
    reader.backend.close()


@pytest.fixture()
def reader_test_data(reader_config: ReaderTestContext) -> ReaderTestData:
    """Prepare plain text files for reader tests."""
    asset_dir = Path(__file__).parents[1] / "assets" / "reader"
    names = ("sample1.test", "sample2.test", "ignored.txt")

    # Use real test assets so read/parse assertions exercise the reader path end-to-end.
    contents = {name: (asset_dir / name).read_text() for name in names}
    files = {name: reader_config.prepare_file(name, content.encode()) for name, content in sorted(contents.items())}
    return ReaderTestData(context=reader_config, files=files, contents=contents)
