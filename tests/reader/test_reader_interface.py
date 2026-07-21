import pickle
from collections.abc import Callable
from re import escape

import pytest

from pythermondt.data import DataContainer
from pythermondt.io import FileInfo
from pythermondt.readers.base_reader import BaseReader, ItemsBy
from tests.reader.conftest import ReaderTestContext, ReaderTestData
from tests.support.storage import PlainTextParser


def _picklable_filter(info: FileInfo) -> bool:
    """Module-level filter for pickle test success case."""
    return "sample1" in info.path


class _PicklableCallable:
    """Callable class filter for pickle test success case."""

    def __init__(self, pattern: str):
        self.pattern = pattern

    def __call__(self, info: FileInfo) -> bool:
        return self.pattern in info.path


def _make_closure(pattern: str) -> Callable[[FileInfo], bool]:
    """Return a non-picklable closure for pickle test failure case."""

    def closure(info: FileInfo) -> bool:
        return pattern in info.path

    return closure


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


def test_str_representation(reader_config: ReaderTestContext):
    """__str__ exposes class name, _get_reader_params output, num_files, download_files, cache_files, and parser."""
    reader = reader_config.reader
    s = str(reader)

    assert reader.__class__.__name__ in s
    assert reader.parser is not None and reader.parser == PlainTextParser
    assert f"num_files={reader.num_files}" in s
    assert f"download_remote_files={reader.download_files}" in s
    assert f"cache_files={reader.cache_files}" in s
    assert f"parser={reader.parser.__name__}" in s


def test_files_use_parser_extensions(reader_test_data: ReaderTestData):
    """Test that reader file discovery respects parser-supported extensions."""
    expected_scheme = reader_test_data.context.config.scheme
    assert reader_test_data.reader.files == reader_test_data.expected_files
    assert all(path.startswith(f"{expected_scheme}://") for path in reader_test_data.reader.files)
    assert all(path.endswith(".test") for path in reader_test_data.reader.files)
    assert reader_test_data.files["ignored.txt"] not in reader_test_data.reader.files


def test_files_and_file_uris_same_count(reader_test_data: ReaderTestData):
    """Test that files and file_uris return the same number of entries for all backends."""
    assert len(reader_test_data.reader.files) == len(reader_test_data.reader.file_uris)


@pytest.mark.parametrize("num_files", [1, 3, 10, 100, None], ids=["1", "3", "10", "100", "None"])
def test_num_files_limits_reader_files(reader_test_data: ReaderTestData, num_files: int | None):
    """Test that num_files limits the discovered reader files."""
    reader = reader_test_data.context.make_reader(parser=PlainTextParser, num_files=num_files)
    expected_files = reader_test_data.expected_files

    assert reader.files == expected_files[:num_files] if num_files else expected_files


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


@pytest.mark.parametrize("by", ["files", "file_names", "file_uris", "file_entries"])
def test_items_keys_match_properties(reader_test_data: ReaderTestData, by: ItemsBy):
    """Test items() keys match the corresponding reader property and payloads stay in order."""
    reader = reader_test_data.reader
    pairs = list(reader.items(by=by))
    expected_keys = getattr(reader, by)

    assert [key for key, _ in pairs] == list(expected_keys)
    assert [_assert_payload(container) for _, container in pairs] == [
        reader_test_data.contents["sample1.test"],
        reader_test_data.contents["sample2.test"],
    ]


def test_items_default_uses_files(reader_test_data: ReaderTestData):
    """Test items() defaults to by='files'."""
    reader = reader_test_data.reader
    assert [key for key, _ in reader.items()] == list(reader.files)


def test_items_reverse(reader_test_data: ReaderTestData):
    """Test items(reverse=True) yields pairs in reverse reader order."""
    reader = reader_test_data.reader
    pairs = list(reader.items(by="file_names", reverse=True))

    assert [key for key, _ in pairs] == list(reversed(reader.file_names))
    assert [_assert_payload(container) for _, container in pairs] == [
        reader_test_data.contents["sample2.test"],
        reader_test_data.contents["sample1.test"],
    ]


def test_items_invalid_by(reader_test_data: ReaderTestData):
    """Test items() raises ValueError for an invalid 'by' value."""
    with pytest.raises(ValueError, match=escape("Invalid 'by' value: 'bogus'")):
        list(reader_test_data.reader.items(by="bogus"))  # type: ignore[arg-type]


def test_read_file_uses_explicit_parser(reader_test_data: ReaderTestData):
    """Test that read_file delegates parsing to the configured parser."""
    container = reader_test_data.reader.read_file(reader_test_data.files["sample1.test"])

    assert _assert_payload(container) == reader_test_data.contents["sample1.test"]


def test_read_file_without_matching_parser_raises(reader_test_data: ReaderTestData):
    """Test that automatic parser lookup rejects unsupported extensions."""
    reader = reader_test_data.context.make_reader(parser=None, num_files=None)

    with pytest.raises(ValueError, match=escape("No parser found for file extension: .test")):
        reader.read_file(reader_test_data.files["sample1.test"])


def test_file_entries_contains_metadata(reader_test_data: ReaderTestData):
    """Each FileInfo entry carries valid path, size, timestamp, and identity."""
    entries = reader_test_data.reader.file_entries

    assert len(entries) == len(reader_test_data.expected_files)
    for entry in entries:
        assert isinstance(entry.path, str)
        assert entry.size > 0
        assert entry.last_modified.tzinfo is not None
        assert isinstance(entry.file_identity, str)


def test_file_uris_and_file_entries_are_consistent(reader_test_data: ReaderTestData):
    """file_uris and file_entries are derived from the same sorted snapshot."""
    reader = reader_test_data.reader

    uris = reader.file_uris
    entries = reader.file_entries

    assert len(uris) == len(entries)
    for uri, entry in zip(uris, entries, strict=True):
        assert uri == entry.path


def test_file_filter_includes_only_matching_files(reader_config: ReaderTestContext):
    """Filter restricts both file_uris and file_entries to matching files."""
    a_uri = reader_config.prepare_file("a.test", b"a")
    b_uri = reader_config.prepare_file("b.test", b"b")
    reader_config.prepare_file("skip1.test", b"s1")
    reader_config.prepare_file("skip2.test", b"s2")

    reader = reader_config.make_reader(file_filter=lambda f: f.path in {a_uri, b_uri})

    assert sorted(reader.file_uris) == sorted([a_uri, b_uri])
    assert len(reader.file_entries) == 2
    assert {e.path for e in reader.file_entries} == {a_uri, b_uri}


def test_file_filter_with_num_files(reader_config: ReaderTestContext):
    """Filter applies before num_files truncation."""
    a_uri = reader_config.prepare_file("a.test", b"a")
    b_uri = reader_config.prepare_file("b.test", b"b")
    reader_config.prepare_file("skip1.test", b"s1")
    reader_config.prepare_file("skip2.test", b"s2")

    reader = reader_config.make_reader(file_filter=lambda f: f.path in {a_uri, b_uri}, num_files=1)

    assert len(reader.file_uris) == 1
    assert len(reader.file_entries) == 1


def test_cache_files_false_reflects_changes(reader_config: ReaderTestContext):
    """Without caching, adding a file is reflected immediately in URIs and entries."""
    reader_config.prepare_file("a.test", b"a")
    reader_config.prepare_file("b.test", b"b")

    reader = reader_config.make_reader(cache_files=False)

    uris_before = reader.file_uris
    entries_before = reader.file_entries

    reader_config.prepare_file("c.test", b"c")

    assert len(reader.file_uris) == len(uris_before) + 1
    assert len(reader.file_entries) == len(entries_before) + 1


def test_cache_files_false_with_filter_excludes_new(reader_config: ReaderTestContext):
    """Without caching, a new file excluded by the filter is not reflected."""
    a_uri = reader_config.prepare_file("a.test", b"a")

    reader = reader_config.make_reader(cache_files=False, file_filter=lambda f: f.path == a_uri)

    assert len(reader.file_uris) == 1
    assert len(reader.file_entries) == 1

    reader_config.prepare_file("b.test", b"b")  # excluded by filter

    assert len(reader.file_uris) == 1
    assert len(reader.file_entries) == 1


@pytest.mark.parametrize("num_files", [1, 3, 10, 100, None], ids=["1", "3", "10", "100", "None"])
@pytest.mark.parametrize("cache_files", [True, False], ids=["cache_files=True", "cache_files=False"])
@pytest.mark.parametrize("parser", [PlainTextParser, None], ids=["parser", "no_parser"])
@pytest.mark.parametrize("file_filter", [None, _picklable_filter], ids=["no_filter", "picklable_filter"])
def test_file_filter_combinations(
    reader_config: ReaderTestContext,
    num_files: int | None,
    cache_files: bool,
    parser: type[PlainTextParser] | None,
    file_filter: Callable[[FileInfo], bool] | None,
):
    """Test that file_filter, num_files, and cache_files interact correctly."""
    # Prepare test files to read
    reader_config.prepare_file("sample1_a.test", b"ma")
    reader_config.prepare_file("sample1_b.test", b"mb")
    reader_config.prepare_file("other_1.test", b"o1")
    reader_config.prepare_file("other_2.test", b"o2")

    # Construct reader with the given parameters
    reader = reader_config.make_reader(
        file_filter=file_filter, num_files=num_files, cache_files=cache_files, parser=parser
    )

    # Construct expected file counts
    total_available = 0 if parser is None else (2 if file_filter is not None else 4)
    expected = min(num_files or total_available, total_available)

    # Assert reader length
    assert len(reader.file_uris) == expected
    assert len(reader.file_entries) == expected
    assert len(reader.files) == expected
    assert len(reader.file_names) == expected

    # Assert file entries and URIs are consistent
    for uri, entry in zip(reader.file_uris, reader.file_entries, strict=True):
        assert uri == entry.path

    # Assert file names
    if file_filter is not None and parser is not None:
        assert all("sample1" in uri for uri in reader.file_uris)


@pytest.mark.parametrize(
    "filter_fn, expect_failure",
    [
        (lambda f: True, True),
        (_make_closure(".test"), True),
        (_picklable_filter, False),
        (_PicklableCallable(".test"), False),
    ],
    ids=["lambda", "closure", "module_fn", "callable_class"],
)
def test_file_filter_pickle(
    reader_config: ReaderTestContext,
    filter_fn: Callable[[FileInfo], bool],
    expect_failure: bool,
):
    """Non-picklable filters raise PicklingError; picklable filters survive a roundtrip."""
    # Setup reader and test files
    reader_config.prepare_file("sample1.test", b"a")
    reader_config.prepare_file("sample2.test", b"b")
    reader = reader_config.make_reader(file_filter=filter_fn)

    # Fail for lambda and closure filters
    if expect_failure:
        with pytest.raises(pickle.PicklingError):
            pickle.dumps(reader)
        return

    # Restore
    original_uris = reader.file_uris
    restored: BaseReader = pickle.loads(pickle.dumps(reader))

    # Assert that the restored reader is correctly configured
    assert restored.backend is not None
    assert restored.file_filter is not None
    assert restored.file_uris == original_uris

    # Assert reader can still read files after being restored
    container = restored.read_file(original_uris[0])
    assert _assert_payload(container) == "a"
