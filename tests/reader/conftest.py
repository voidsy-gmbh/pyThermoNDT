from dataclasses import dataclass
from pathlib import Path

import pytest

from pythermondt.readers import BaseReader
from tests.support.storage import StorageTestContext


@dataclass(frozen=True)
class ReaderTestData:
    """Prepared reader files and their source contents."""

    context: StorageTestContext
    reader: BaseReader
    files: dict[str, str]
    contents: dict[str, str]

    @property
    def expected_files(self) -> list[str]:
        """Return parser-supported files in deterministic reader order."""
        return [self.files[name] for name in sorted(self.contents) if name.endswith(".test")]


@pytest.fixture()
def reader_test_data(storage_context: StorageTestContext) -> ReaderTestData:
    """Prepare plain text files for reader tests."""
    asset_dir = Path(__file__).parents[1] / "assets" / "reader"
    names = ("sample1.test", "sample2.test", "ignored.txt")
    contents = {name: (asset_dir / name).read_text() for name in names}
    files = storage_context.prepare_files({name: content.encode() for name, content in sorted(contents.items())})
    return ReaderTestData(
        context=storage_context,
        reader=storage_context.make_reader(),
        files=files,
        contents=contents,
    )
