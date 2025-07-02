import pytest

from pythermondt.dataset import IndexedThermoDataset, ThermoDataset


def test_basic_initialization(sample_dataset_single_file: ThermoDataset):
    """Test basic IndexedThermoDataset initialization."""
    indexed = IndexedThermoDataset(sample_dataset_single_file, [0])

    assert indexed.parent == sample_dataset_single_file
    assert len(indexed) == 1
    assert len(indexed.files) == 1


def test_empty_indices(sample_dataset_single_file: ThermoDataset):
    """Test initialization with empty indices list."""
    indexed = IndexedThermoDataset(sample_dataset_single_file, [])

    # Empty dataset should still have a parent and no files ==> can happen e.g. when using random_split()
    assert indexed.parent == sample_dataset_single_file
    assert len(indexed) == 0
    assert len(indexed.files) == 0


@pytest.mark.parametrize(
    "invalid_indices",
    [
        [-1],  # Negative index
        [100],  # Out of bounds
        [0, -1],  # Mixed valid/invalid
        [0, 100],  # Mixed valid/invalid
        [-5, -1],  # Multiple negative
    ],
)
def test_invalid_indices(sample_dataset_single_file: ThermoDataset, invalid_indices):
    """Test initialization with invalid indices raises IndexError."""
    with pytest.raises(IndexError, match="out of range"):
        IndexedThermoDataset(sample_dataset_single_file, invalid_indices)


@pytest.mark.parametrize("idx", [1, 10, -1, -10])
def test_invalid_index_access(sample_dataset_three_files: ThermoDataset, idx: int):
    """Test accessing an invalid index raises IndexError."""
    indexed = IndexedThermoDataset(sample_dataset_three_files, [0])

    with pytest.raises(IndexError, match="Index out of range"):
        indexed[idx]  # Accessing index that does not exist


def test_duplicate_indices_allowed(sample_dataset_three_files: ThermoDataset):
    """Test that duplicate indices in the list are allowed."""
    indices = [0, 1, 0, 1]  # Duplicates (Could be useful e.g. for oversampling)
    indexed = IndexedThermoDataset(sample_dataset_three_files, indices)

    # Should create dataset with 4 items (allowing duplicates)
    assert len(indexed) == 4
