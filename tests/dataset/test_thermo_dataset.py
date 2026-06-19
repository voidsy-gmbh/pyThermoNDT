import time
from re import escape
from unittest.mock import patch

import pytest
import torch

from pythermondt import LocalReader, S3Reader, ThermoDataset, configure_logging
from pythermondt.transforms import ThermoTransform

from ..utils import containers_equal


def test_basic_initialization(localreader_with_file: LocalReader):
    """Test root dataset parent is None."""
    dataset = ThermoDataset(localreader_with_file)
    assert dataset.parent is None
    assert len(dataset) == len(localreader_with_file)
    assert dataset.files == localreader_with_file.files


def test_empty_readers_list():
    """Test initialization with empty reader list."""
    # Initialize ThermoDataset with an empty list
    with pytest.raises(
        ValueError,
        match=escape("No readers provided. Please provide at least one BaseReader instance or a list of them."),
    ):
        ThermoDataset([])


def test_empty_reader(caplog, localreader_no_files):
    """Test initialization with readers that have no files."""
    with pytest.warns(UserWarning, match="No files found for reader of type LocalReader"):
        ThermoDataset(localreader_no_files)


@pytest.mark.parametrize(
    "paths",
    [
        # glob -> directory
        ("./tests/assets/integration/simulation/*.mat", "./tests/assets/integration/simulation/"),
        ("./tests/assets/integration/simulation/", "./tests/assets/integration/simulation/"),
        # file -> directory (file is contained in directory)
        ("./tests/assets/integration/simulation/source1.mat", "./tests/assets/integration/simulation/"),
        # glob -> glob (same pattern)
        ("./tests/assets/integration/simulation/*.mat", "./tests/assets/integration/simulation/*.mat"),
        # file -> file (same file)
        ("./tests/assets/integration/simulation/source1.mat", "./tests/assets/integration/simulation/source1.mat"),
        # glob -> file (file matches glob pattern)
        ("./tests/assets/integration/simulation/*.mat", "./tests/assets/integration/simulation/source1.mat"),
        # directory -> file (file is contained in directory)
        ("./tests/assets/integration/simulation/", "./tests/assets/integration/simulation/source1.mat"),
        # glob -> glob (overlapping patterns)
        ("./tests/assets/integration/simulation/source*.mat", "./tests/assets/integration/simulation/*.mat"),
        # file -> glob (file matches different glob)
        ("./tests/assets/integration/simulation/source1.mat", "./tests/assets/integration/simulation/source*.mat"),
    ],
)
def test_duplicate_files(caplog, paths: tuple[str, str]):
    """Test initialization with duplicate files."""
    configure_logging()
    localreader1 = LocalReader(pattern=paths[0])
    localreader2 = LocalReader(pattern=paths[1])

    with pytest.warns(UserWarning, match="Duplicate files found for reader of type LocalReader"):
        ThermoDataset([localreader1, localreader2])


@pytest.mark.parametrize(
    "paths",
    [
        # Different directories
        ("./tests/assets/integration/simulation/", "./tests/assets/perf/small/"),
        # Different files
        ("./tests/assets/integration/simulation/source1.mat", "./tests/assets/integration/simulation/source2.mat"),
        # Different glob patterns
        ("./tests/assets/integration/simulation/*.mat", "./tests/assets/integration/simulation/*.hdf5"),
    ],
)
def test_no_false_positive_duplicates(paths: tuple[str, str]):
    """Test that duplicate detection doesn't produce false positives for non-overlapping sources."""
    localreader1 = LocalReader(pattern=paths[0])
    localreader2 = LocalReader(pattern=paths[1])

    # This should NOT raise an exception
    try:
        dataset = ThermoDataset([localreader1, localreader2])
        # Verify the dataset was created successfully
        assert len(dataset) == len(localreader1.files) + len(localreader2.files)
    except ValueError as e:
        if "Duplicate files found for reader of type LocalReader" in str(e):
            pytest.fail(f"False positive duplicate detection for paths {paths}: {e}")
        else:
            # Re-raise if it's a different ValueError (like no files found)
            raise


@pytest.mark.parametrize("mode", ["immediate", "lazy"])
@pytest.mark.parametrize("num_workers", [None, 0, 1, -1])
def test_build_cache_thermodataset(
    local_reader_three_files: LocalReader, sample_pipeline: ThermoTransform, mode: str, num_workers: int | None
):
    """Test building cache for ThermoDataset and verify correctness and speedup."""
    # Create the datasets
    dataset_no_cache = ThermoDataset(local_reader_three_files, transform=sample_pipeline)
    dataset_cache = ThermoDataset(local_reader_three_files, transform=sample_pipeline)

    dataset_cache.build_cache(mode=mode, num_workers=num_workers)  # type: ignore[call-arg]

    # Check correctness
    for idx in range(len(dataset_no_cache)):
        torch.manual_seed(42)
        cache = dataset_cache[idx]
        torch.manual_seed(42)
        no_cache = dataset_no_cache[idx]
        # If mode is lazy ==> datacontainer gets pickled and NaN values may not be equal: see https://bugs.python.org/issue43078
        if mode == "lazy":
            assert containers_equal(cache, no_cache, ignore_nan_inequality=True), f"Cache mismatch at index {idx}"
        else:
            assert containers_equal(cache, no_cache), f"Cache mismatch at index {idx}"

    # Check speedup
    torch.manual_seed(42)
    start_no_cache = time.perf_counter()
    for _ in dataset_no_cache:
        pass
    duration_no_cache = time.perf_counter() - start_no_cache

    torch.manual_seed(42)
    start_cache = time.perf_counter()
    for _ in dataset_cache:
        pass
    duration_cache = time.perf_counter() - start_cache

    # Cached access should be faster (allow some tolerance for small datasets)
    assert duration_cache < duration_no_cache * 0.8 or duration_no_cache - duration_cache > 0.01, (
        f"Caching did not provide a significant speedup: no_cache={duration_no_cache:.4f}s, cache={duration_cache:.4f}s"
    )


def test_cache_files_false_warning():
    """Test that a reader with cache_files=False emits a warning."""
    reader = LocalReader(pattern="./tests/assets/integration/simulation/source1.mat", cache_files=False)
    with pytest.warns(UserWarning, match="cache_files=False"):
        ThermoDataset(reader)


def test_remote_no_download_warning(s3reader_with_file: S3Reader):
    """Test that a remote reader with download_files=False emits a warning."""
    with pytest.warns(UserWarning, match="S3Reader is remote but download_files=False."):
        ThermoDataset(s3reader_with_file)


def test_empty_reader_in_multi_reader_warns(localreader_with_file: LocalReader, localreader_no_files: LocalReader):
    """Test that an empty reader among multiple readers of same type emits a warning."""
    with pytest.warns(UserWarning, match="No files found for reader of type"):
        ThermoDataset([localreader_no_files, localreader_with_file])


def test_empty_reader_different_type_warns(localreader_with_file: LocalReader, altreader_no_files: LocalReader):
    """Test that an empty reader of a different type is detected regardless of its position.

    Regression test for issue #425: the else branch of _validate_readers incorrectly checked
    readers[0] instead of readers_objects[0], so empty readers of a different type than the
    first reader were silently missed.
    """
    expected_match = r"No files found for reader of type .*AltReader"

    # Empty reader NOT first — must still warn (this was the swallowed-warning case)
    with pytest.warns(UserWarning, match=expected_match):
        ThermoDataset([localreader_with_file, altreader_no_files])

    # Empty reader first — must warn and NOT produce a false positive for the other type
    with pytest.warns(UserWarning, match=expected_match) as record:
        ThermoDataset([altreader_no_files, localreader_with_file])
    messages = [str(w.message) for w in record]
    assert not any("LocalReader" in m for m in messages), (
        f"False positive warning for non-empty reader type: {messages}"
    )


def test_download_delegates_to_readers(s3reader_with_file: S3Reader, localreader_with_file: LocalReader):
    """Test that dataset.download() calls download on remote readers."""
    # Expect warning because fixture has download_files=False for the S3 reader
    with pytest.warns(UserWarning, match="S3Reader is remote but download_files=False."):
        dataset = ThermoDataset([s3reader_with_file, localreader_with_file])

    # Patch the download methods to track calls
    with patch.object(localreader_with_file, "download") as mock_local_download:
        with patch.object(s3reader_with_file, "download") as mock_s3_download:
            dataset.download(num_workers=2)
            mock_s3_download.assert_called_once_with(num_workers=2)
            mock_local_download.assert_not_called()


def test_download_skips_local_readers(recwarn, localreader_with_file: LocalReader):
    """Test that dataset.download() skips non-remote readers."""
    dataset = ThermoDataset(localreader_with_file)
    dataset.download()  # No-op for local readers, should not error
    assert len(recwarn) == 0, f"Unexpected warning when downloading with local reader: {recwarn.list}"


def test_load_raw_data_index_validation(sample_dataset_single_file: ThermoDataset):
    """Test that load_raw_data validates index bounds."""
    msg = escape(f"Index -1 out of range. Must be within [0, {len(sample_dataset_single_file) - 1}]")
    with pytest.raises(IndexError, match=msg):
        sample_dataset_single_file.load_raw_data(-1)

    msg = escape(
        f"Index {len(sample_dataset_single_file)} out of range. "
        f"Must be within [0, {len(sample_dataset_single_file) - 1}]"
    )
    with pytest.raises(IndexError, match=msg):
        sample_dataset_single_file.load_raw_data(len(sample_dataset_single_file))


@pytest.mark.parametrize(
    "error,expected_type,match_pattern",
    [
        (FileNotFoundError("gone"), RuntimeError, "Cannot read file"),
        (OSError("disk error"), RuntimeError, "Cannot read file"),
        (PermissionError("denied"), RuntimeError, "Cannot read file"),
        (ValueError("bad format"), ValueError, "Cannot parse file"),
    ],
)
def test_load_raw_data_error_handling(
    localreader_with_file: LocalReader, error: BaseException, expected_type: type, match_pattern: str
):
    """Test that load_raw_data wraps reader exceptions with informative messages."""
    dataset = ThermoDataset(localreader_with_file)
    with patch.object(localreader_with_file, "read_file", side_effect=error):
        with pytest.raises(expected_type, match=match_pattern):
            dataset.load_raw_data(0)
