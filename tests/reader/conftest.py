from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pytest

from pythermondt.io import FileInfo
from pythermondt.io.parsers import BaseParser
from pythermondt.readers import AzureBlobReader, BaseReader, LocalReader, S3Reader
from tests.support import storage
from tests.support.storage.config import StorageTestContext


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


@pytest.fixture
def reader_config(storage_backend: StorageTestContext, tmp_path: Path) -> Generator[ReaderTestContext]:
    """Create a reader from the shared storage backend configuration."""
    config = storage_backend.config

    if config.reader_cls == LocalReader:

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

    reader = make_reader()
    yield ReaderTestContext(
        reader=reader,
        config=config,
        make_reader=make_reader,
        prepare_file=storage_backend.prepare_file,
    )

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
