from re import escape

import pytest

from pythermondt.data import DataContainer
from tests.reader.conftest import ReaderTestContext, ReaderTestData
from tests.support.storage import PlainTextParser


def _assert_payload(container: DataContainer) -> str:
    """Extract the text payload written by PlainTextParser."""
    payload = container.get_attribute("/MetaData", "payload")
    assert isinstance(payload, str)
    return payload


def test_remote_source(reader_config: ReaderTestContext):
    """Test that readers expose the backend remote/local source type."""
    assert reader_config.reader.remote_source == reader_config.config.is_remote


def test_parser_class(reader_config: ReaderTestContext):
    """Test that readers expose the configured parser class (in this case always PlainTextParser)."""
    assert reader_config.reader.parser == PlainTextParser


def test_files_use_parser_extensions(reader_test_data: ReaderTestData):
    """Test that reader file discovery respects parser-supported extensions."""
    assert reader_test_data.reader.files == reader_test_data.expected_files
    assert all(
        path.startswith(f"{reader_test_data.context.config.scheme}://") for path in reader_test_data.reader.files
    )
    assert all(path.endswith(".test") for path in reader_test_data.reader.files)
    assert reader_test_data.files["ignored.txt"] not in reader_test_data.reader.files


def test_num_files_limits_reader_files(reader_test_data: ReaderTestData):
    """Test that num_files limits the discovered reader files."""
    reader = reader_test_data.context.make_reader(parser=PlainTextParser, num_files=1)

    assert reader.files == reader_test_data.expected_files[:1]


def test_file_names(reader_test_data: ReaderTestData):
    """Test that file_names strips storage-specific path prefixes."""
    assert reader_test_data.reader.file_names == ["sample1.test", "sample2.test"]


def test_len(reader_test_data: ReaderTestData):
    """Test that len(reader) reflects the discovered files."""
    assert len(reader_test_data.reader) == len(reader_test_data.expected_files)


def test_getitem(reader_test_data: ReaderTestData):
    """Test indexed access reads and parses the selected file."""
    container = reader_test_data.reader[0]

    assert _assert_payload(container) == reader_test_data.contents["sample1.test"]


@pytest.mark.parametrize("index", [-100, -1, None])
def test_getitem_invalid_index(reader_test_data: ReaderTestData, index: int | None):
    """Test indexed access validates bounds before reading."""
    idx = index if index is not None else len(reader_test_data.expected_files) + 5
    with pytest.raises(IndexError, match=escape("Index out of bounds.")):
        reader_test_data.reader[idx]


def test_iter(reader_test_data: ReaderTestData):
    """Test forward iteration reads files in reader order."""
    payloads = [_assert_payload(container) for container in reader_test_data.reader]

    assert payloads == [reader_test_data.contents["sample1.test"], reader_test_data.contents["sample2.test"]]


def test_reversed(reader_test_data: ReaderTestData):
    """Test reverse iteration reads files in reverse reader order."""
    payloads = [_assert_payload(container) for container in reversed(reader_test_data.reader)]

    assert payloads == [reader_test_data.contents["sample2.test"], reader_test_data.contents["sample1.test"]]


def test_read_file_uses_explicit_parser(reader_test_data: ReaderTestData):
    """Test that read_file delegates parsing to the configured parser."""
    container = reader_test_data.reader.read_file(reader_test_data.files["sample1.test"])

    assert _assert_payload(container) == reader_test_data.contents["sample1.test"]


def test_read_file_without_matching_parser_raises(reader_test_data: ReaderTestData):
    """Test that automatic parser lookup rejects unsupported extensions."""
    reader = reader_test_data.context.make_reader(parser=None, num_files=None)

    with pytest.raises(ValueError, match=escape("No parser found for file extension: .test")):
        reader.read_file(reader_test_data.files["sample1.test"])
