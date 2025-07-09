import time

import pytest
import torch

from pythermondt import LocalReader, ThermoDataset
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
        ValueError, match="No readers provided. Please provide at least one BaseReader instance or a list of them."
    ):
        ThermoDataset([])


def test_empty_reader(localreader_no_files):
    """Test initialization with readers that have no files."""
    with pytest.raises(ValueError, match="No files found for reader of type LocalReader"):
        ThermoDataset(localreader_no_files)


@pytest.mark.parametrize(
    "paths",
    [
        # glob -> directory
        pytest.param(
            ("./tests/assets/integration/simulation/*.mat", "./tests/assets/integration/simulation/"),
            marks=pytest.mark.xfail(reason="Backend path normalization needed"),
        ),
        ("./tests/assets/integration/simulation/", "./tests/assets/integration/simulation/"),
        # file -> directory (file is contained in directory)
        ("./tests/assets/integration/simulation/source1.mat", "./tests/assets/integration/simulation/"),
        # glob -> glob (same pattern)
        ("./tests/assets/integration/simulation/*.mat", "./tests/assets/integration/simulation/*.mat"),
        # file -> file (same file)
        ("./tests/assets/integration/simulation/source1.mat", "./tests/assets/integration/simulation/source1.mat"),
        # glob -> file (file matches glob pattern)
        pytest.param(
            ("./tests/assets/integration/simulation/*.mat", "./tests/assets/integration/simulation/source1.mat"),
            marks=pytest.mark.xfail(reason="Backend path normalization needed"),
        ),
        # directory -> file (file is contained in directory)
        ("./tests/assets/integration/simulation/", "./tests/assets/integration/simulation/source1.mat"),
        # glob -> glob (overlapping patterns)
        ("./tests/assets/integration/simulation/source*.mat", "./tests/assets/integration/simulation/*.mat"),
        # file -> glob (file matches different glob)
        pytest.param(
            ("./tests/assets/integration/simulation/source1.mat", "./tests/assets/integration/simulation/source*.mat"),
            marks=pytest.mark.xfail(reason="Backend path normalization needed"),
        ),
    ],
)
def test_duplicate_files(paths: tuple[str, str]):
    """Test initialization with duplicate files."""
    localreader1 = LocalReader(pattern=paths[0])
    localreader2 = LocalReader(pattern=paths[1])

    print(f"Files in localreader1: {localreader1.files}")
    print(f"Files in localreader2: {localreader2.files}")

    with pytest.raises(ValueError, match="Duplicate files found for reader of type LocalReader"):
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

    print(f"Files in localreader1: {localreader1.files}")
    print(f"Files in localreader2: {localreader2.files}")

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
def test_build_cache_thermodataset(local_reader_three_files: LocalReader, sample_pipeline: ThermoTransform, mode: str):
    """Test building cache for ThermoDataset and verify correctness and speedup."""
    # Create the datasets
    dataset_no_cache = ThermoDataset(local_reader_three_files, transform=sample_pipeline)
    dataset_cache = ThermoDataset(local_reader_three_files, transform=sample_pipeline)

    dataset_cache.build_cache(mode=mode)  # type: ignore[call-arg]

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
