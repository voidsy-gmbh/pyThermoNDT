import logging
import pickle
from collections.abc import Callable
from re import escape
from unittest.mock import patch

import pytest

import pythermondt.readers.base_reader as reader_module
from pythermondt.data import DataContainer
from pythermondt.io import AzureBlobBackend, FileInfo, LocalBackend, S3Backend
from pythermondt.readers import BaseReader, ItemsBy
from tests.reader.conftest import ReaderTestData
from tests.support.storage import PlainTextParser, StorageTestContext


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


@pytest.mark.parametrize(
    "storage_context, expected_remote_source",
    [(LocalBackend, False), (S3Backend, True), (AzureBlobBackend, True)],
    indirect=["storage_context"],
)
def test_remote_source(storage_context: StorageTestContext, expected_remote_source: bool):
    """Test that readers expose the backend remote/local source type."""
    reader = storage_context.make_reader()
    assert reader.remote_source is expected_remote_source


def test_parser_class(storage_context: StorageTestContext):
    """Test that readers expose the configured parser class (in this case always PlainTextParser)."""
    assert storage_context.make_reader().parser == PlainTextParser


def test_str_representation(storage_context: StorageTestContext):
    """__str__ exposes class name, _get_reader_params output, num_files, download_files, listing_ttl, and parser."""
    reader = storage_context.make_reader()
    s = str(reader)

    assert reader.__class__.__name__ in s
    assert reader.parser is not None and reader.parser == PlainTextParser
    assert f"num_files={reader.num_files}" in s
    assert f"download_remote_files={reader.download_files}" in s
    assert f"listing_ttl={reader.listing_ttl}" in s
    assert f"parser={reader.parser.__name__}" in s


def test_files_use_parser_extensions(reader_test_data: ReaderTestData):
    """Test that reader file discovery respects parser-supported extensions."""
    expected_scheme = reader_test_data.context.backend.scheme
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
    assert list(reader.items()) == list(reader.items(by="files"))


def test_items_pairs_key_to_container(reader_test_data: ReaderTestData):
    """Test each items() key maps to the container for that file."""
    reader = reader_test_data.reader
    for name, container in reader.items(by="file_names"):
        assert _assert_payload(container) == reader_test_data.contents[name]


def test_items_reverse(reader_test_data: ReaderTestData):
    """Test items(reverse=True) yields pairs in reverse reader order."""
    reader = reader_test_data.reader
    pairs = list(reader.items(by="file_names", reverse=True))

    assert [key for key, _ in pairs] == list(reversed(reader.file_names))
    for name, container in pairs:
        assert _assert_payload(container) == reader_test_data.contents[name]


def test_items_invalid_by(reader_test_data: ReaderTestData):
    """Test items() raises ValueError for an invalid 'by' value."""
    with pytest.raises(ValueError, match=escape("Invalid 'by' value: 'bogus'")):
        list(reader_test_data.reader.items(by="bogus"))  # type: ignore[arg-type]


@pytest.mark.parametrize("by", ["files", "file_names", "file_uris", "file_entries"])
def test_items_with_listing_ttl_zero(storage_context: StorageTestContext, by: ItemsBy):
    """items() works with listing_ttl=0 and keeps key/container pairs consistent."""
    storage_context.prepare_file("sample1.test", b"payload1")
    storage_context.prepare_file("sample2.test", b"payload2")
    reader = storage_context.make_reader(listing_ttl=0)

    pairs = list(reader.items(by=by))
    expected_keys = getattr(reader, by)

    assert [key for key, _ in pairs] == list(expected_keys)
    assert [_assert_payload(container) for _, container in pairs] == ["payload1", "payload2"]


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


def test_file_filter_includes_only_matching_files(storage_context: StorageTestContext):
    """Filter restricts both file_uris and file_entries to matching files."""
    a_uri = storage_context.prepare_file("a.test", b"a")
    b_uri = storage_context.prepare_file("b.test", b"b")
    storage_context.prepare_file("skip1.test", b"s1")
    storage_context.prepare_file("skip2.test", b"s2")

    reader = storage_context.make_reader(file_filter=lambda f: f.path in {a_uri, b_uri})

    assert sorted(reader.file_uris) == sorted([a_uri, b_uri])
    assert len(reader.file_entries) == 2
    assert {e.path for e in reader.file_entries} == {a_uri, b_uri}


def test_file_filter_with_num_files(storage_context: StorageTestContext):
    """Filter applies before num_files truncation."""
    a_uri = storage_context.prepare_file("a.test", b"a")
    b_uri = storage_context.prepare_file("b.test", b"b")
    storage_context.prepare_file("skip1.test", b"s1")
    storage_context.prepare_file("skip2.test", b"s2")

    reader = storage_context.make_reader(file_filter=lambda f: f.path in {a_uri, b_uri}, num_files=1)

    assert len(reader.file_uris) == 1
    assert len(reader.file_entries) == 1


def test_listing_ttl_zero_reflects_changes(storage_context: StorageTestContext):
    """Without caching, adding a file is reflected immediately in URIs and entries."""
    storage_context.prepare_file("a.test", b"a")
    storage_context.prepare_file("b.test", b"b")

    reader = storage_context.make_reader(listing_ttl=0)

    uris_before = reader.file_uris
    entries_before = reader.file_entries

    storage_context.prepare_file("c.test", b"c")

    assert len(reader.file_uris) == len(uris_before) + 1
    assert len(reader.file_entries) == len(entries_before) + 1


def test_listing_ttl_zero_with_filter_excludes_new(storage_context: StorageTestContext):
    """Without caching, a new file excluded by the filter is not reflected."""
    a_uri = storage_context.prepare_file("a.test", b"a")

    reader = storage_context.make_reader(listing_ttl=0, file_filter=lambda f: f.path == a_uri)

    assert len(reader.file_uris) == 1
    assert len(reader.file_entries) == 1

    storage_context.prepare_file("b.test", b"b")  # excluded by filter

    assert len(reader.file_uris) == 1
    assert len(reader.file_entries) == 1


def test_listing_ttl_zero_uses_fast_path(storage_context: StorageTestContext):
    """URI access without a metadata filter does not fetch file metadata."""
    storage_context.prepare_file("a.test", b"a")
    reader = storage_context.make_reader(listing_ttl=0)

    with patch.object(reader.backend, "get_file_list_with_metadata", side_effect=AssertionError("metadata listing")):
        assert len(reader.file_uris) == 1
        assert len(reader.files) == 1


@pytest.mark.parametrize("listing_ttl", [None, 0, 60])
def test_listing_ttl_refresh(storage_context: StorageTestContext, listing_ttl: float | None):
    """New files appear only after the listing expires, unless every access refreshes it."""
    # Control the reader's clock without waiting for the TTL to pass.
    clock = [100.0]
    storage_context.prepare_file("a.test", b"a")
    reader = storage_context.make_reader(listing_ttl=listing_ttl)

    with (
        patch.object(reader_module, "monotonic", lambda: clock[0]),
        patch.object(
            reader.backend, "get_file_list_with_metadata", wraps=reader.backend.get_file_list_with_metadata
        ) as listing,
    ):
        assert reader.file_names == ["a.test"]
        storage_context.prepare_file("b.test", b"b")
        # Check both sides of the 60-second expiry boundary.
        clock[0] = 159.0
        assert reader.file_names == (["a.test", "b.test"] if listing_ttl == 0 else ["a.test"])
        clock[0] = 160.0
        assert reader.file_names == (["a.test"] if listing_ttl is None else ["a.test", "b.test"])
        assert listing.call_count == (0 if listing_ttl == 0 else 1 if listing_ttl is None else 2)

    assert len(reader.file_uris) == len(reader.file_entries) == len(reader.files) == len(reader.file_names)


@pytest.mark.parametrize("listing_ttl", [None, 60])
def test_clear_listing_cache(storage_context: StorageTestContext, listing_ttl: float | None):
    """Explicit invalidation is lazy and clears all derived file lists."""
    storage_context.prepare_file("a.test", b"a")
    reader = storage_context.make_reader(listing_ttl=listing_ttl)
    assert reader.file_names == ["a.test"]
    storage_context.prepare_file("b.test", b"b")

    with patch.object(
        reader.backend, "get_file_list_with_metadata", wraps=reader.backend.get_file_list_with_metadata
    ) as listing:
        # Clearing only invalidates the listing; the next access calls the backend.
        reader.clear_listing_cache()
        assert listing.call_count == 0
        assert reader.file_names == ["a.test", "b.test"]
        assert listing.call_count == 1
        assert [entry.path for entry in reader.file_entries] == reader.file_uris
        assert listing.call_count == 1


@pytest.mark.parametrize("storage_context", [LocalBackend], indirect=True)
def test_file_list_cache_debug_logs(storage_context: StorageTestContext, caplog: pytest.LogCaptureFixture):
    """Manual clearing and expiry log once each, but clearing an empty cache stays quiet."""
    clock = [100.0]
    storage_context.prepare_file("a.test", b"a")
    reader = storage_context.make_reader(listing_ttl=60)

    with (
        patch.object(reader_module, "monotonic", lambda: clock[0]),
        caplog.at_level(logging.DEBUG, logger=reader_module.__name__),
    ):
        # An empty-cache clear stays quiet; only manual clearing and expiry are logged.
        reader.clear_listing_cache()
        assert reader.file_names == ["a.test"]
        reader.clear_listing_cache()
        assert reader.file_names == ["a.test"]
        clock[0] = 160.0
        assert reader.file_names == ["a.test"]

    messages = [record.message for record in caplog.records if record.name == reader_module.__name__]
    assert messages == [
        "LocalReader - File listing cache cleared manually.",
        "LocalReader - File listing expired after 60 seconds.",
    ]


def test_listing_ttl_refresh_with_filter(storage_context: StorageTestContext):
    """Timed refresh applies the metadata filter and updates both listing views."""
    clock = [100.0]
    storage_context.prepare_file("a.test", b"a")
    reader = storage_context.make_reader(listing_ttl=60, file_filter=_picklable_filter)

    with patch.object(reader_module, "monotonic", lambda: clock[0]):
        assert reader.file_entries == []
        storage_context.prepare_file("sample1.test", b"sample1")
        clock[0] = 160.0

        assert reader.file_names == ["sample1.test"]
        assert [entry.path for entry in reader.file_entries] == reader.file_uris


def test_listing_failure_retries(storage_context: StorageTestContext):
    """A failed refresh raises and the next access retries without serving expired data."""
    clock = [100.0]
    storage_context.prepare_file("a.test", b"a")
    reader = storage_context.make_reader(listing_ttl=60)

    with patch.object(reader_module, "monotonic", lambda: clock[0]):
        assert reader.file_names == ["a.test"]
        storage_context.prepare_file("b.test", b"b")
        clock[0] = 160.0

        with patch.object(reader.backend, "get_file_list_with_metadata", side_effect=OSError("listing failed")):
            with pytest.raises(OSError, match="listing failed"):
                assert reader.file_names

        # A failed refresh does not make the old listing valid again.
        assert reader.file_names == ["a.test", "b.test"]


def test_large_integer_listing_ttl(storage_context: StorageTestContext):
    """Integer TTLs beyond the float range retain the cached listing without overflow."""
    listing_ttl = 10**1000
    storage_context.prepare_file("a.test", b"a")
    reader = storage_context.make_reader(listing_ttl=listing_ttl)

    assert reader.listing_ttl == listing_ttl
    uris = reader.file_uris
    storage_context.prepare_file("b.test", b"b")
    assert reader.file_uris == uris


@pytest.mark.parametrize(
    "listing_ttl",
    [-1, pytest.param(-(10**1000), id="large_negative_integer"), float("nan"), float("inf"), float("-inf")],
)
def test_invalid_listing_ttl_value(storage_context: StorageTestContext, listing_ttl: float):
    with pytest.raises(ValueError, match="listing_ttl must be finite and non-negative"):
        storage_context.make_reader(listing_ttl=listing_ttl)


@pytest.mark.parametrize("listing_ttl", ["60", object()])
def test_invalid_listing_ttl_type(storage_context: StorageTestContext, listing_ttl: object):
    with pytest.raises(TypeError, match="listing_ttl must be a non-negative number or None"):
        storage_context.make_reader(listing_ttl=listing_ttl)  # type: ignore[arg-type]


@pytest.mark.parametrize("listing_ttl", [True, False])
def test_boolean_listing_ttl_requires_migration(storage_context: StorageTestContext, listing_ttl: bool):
    message = (
        f"listing_ttl must be a non-negative number or None, got {listing_ttl!r} (bool). "
        "cache_files is deprecated; use listing_ttl=None instead of cache_files=True "
        "or listing_ttl=0 instead of cache_files=False."
    )
    with pytest.raises(TypeError, match=escape(message)):
        storage_context.make_reader(listing_ttl=listing_ttl)


def test_pickle_clears_listing_timestamp(storage_context: StorageTestContext):
    """Unpickling reloads the listing, even if its previous TTL has not elapsed."""
    storage_context.prepare_file("a.test", b"a")
    reader = storage_context.make_reader(listing_ttl=60)
    assert reader.file_names == ["a.test"]
    restored = pickle.loads(pickle.dumps(reader))
    storage_context.prepare_file("b.test", b"b")

    assert restored.listing_ttl == 60
    assert restored.file_names == ["a.test", "b.test"]


@pytest.mark.parametrize("num_files", [1, 3, 10, 100, None], ids=["1", "3", "10", "100", "None"])
@pytest.mark.parametrize("listing_ttl", [None, 0, 60], ids=["forever", "always", "timed"])
@pytest.mark.parametrize("parser", [PlainTextParser, None], ids=["parser", "no_parser"])
@pytest.mark.parametrize("file_filter", [None, _picklable_filter], ids=["no_filter", "picklable_filter"])
def test_file_filter_combinations(
    storage_context: StorageTestContext,
    num_files: int | None,
    listing_ttl: float | None,
    parser: type[PlainTextParser] | None,
    file_filter: Callable[[FileInfo], bool] | None,
):
    """Test that file_filter, num_files, and listing_ttl interact correctly."""
    # Prepare test files to read
    storage_context.prepare_file("sample1_a.test", b"ma")
    storage_context.prepare_file("sample1_b.test", b"mb")
    storage_context.prepare_file("other_1.test", b"o1")
    storage_context.prepare_file("other_2.test", b"o2")

    # Construct reader with the given parameters
    reader = storage_context.make_reader(
        file_filter=file_filter, num_files=num_files, listing_ttl=listing_ttl, parser=parser
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
    storage_context: StorageTestContext,
    filter_fn: Callable[[FileInfo], bool],
    expect_failure: bool,
):
    """Non-picklable filters raise PicklingError; picklable filters survive a roundtrip."""
    # Setup reader and test files
    storage_context.prepare_file("sample1.test", b"a")
    storage_context.prepare_file("sample2.test", b"b")
    reader = storage_context.make_reader(file_filter=filter_fn)

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
