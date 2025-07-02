import pytest

from pythermondt.dataset import ThermoDataset
from pythermondt.readers import LocalReader


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
