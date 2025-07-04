import pytest

from pythermondt import IndexedThermoDataset, LocalReader, ThermoDataset
from pythermondt.transforms import ThermoTransform

from ..utils import containers_equal


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


def test_transform_chain(local_reader_three_files: LocalReader, simple_transform: type[ThermoTransform]):
    """Test that additional transform is applied after parent's transform."""
    # Create transforms
    base_transform = simple_transform("base_level")
    first_transform = simple_transform("first_level")
    second_transform = simple_transform("second_level")
    third_transform = simple_transform("third_level")

    # Create the datasets
    dataset = ThermoDataset(local_reader_three_files, transform=base_transform)
    indexed = IndexedThermoDataset(dataset, [0, 2], transform=first_transform)
    indexed2 = IndexedThermoDataset(indexed, [1, 0], transform=second_transform)
    indexed3 = IndexedThermoDataset(indexed2, [1], transform=third_transform)

    # Get the first item and check if the transform were applied correctly
    data_parent = dataset[0]
    data_child = indexed[0]
    data_grand_child = indexed2[0]
    data_grand_grand_child = indexed3[0]
    assert data_parent.get_attribute("/MetaData", "transformed") == ["base_level"]
    assert data_child.get_attribute("/MetaData", "transformed") == ["base_level", "first_level"]
    assert data_grand_child.get_attribute("/MetaData", "transformed") == ["base_level", "first_level", "second_level"]
    assert data_grand_grand_child.get_attribute("/MetaData", "transformed") == [
        "base_level",
        "first_level",
        "second_level",
        "third_level",
    ]

    # Check if transform chain is applied correctly in the parent dataset
    chain = dataset.get_transform_chain()
    assert isinstance(chain, ThermoTransform)
    for i, container in enumerate(dataset):
        assert containers_equal(chain(dataset.load_raw_data(i)), container)

    # Check if transform chain is applied correctly in the child dataset
    chain = indexed.get_transform_chain()
    assert isinstance(chain, ThermoTransform)
    for i, container in enumerate(indexed):
        assert chain(indexed.load_raw_data(i)) == container

    # Check if transform chain is applied correctly in the grandchild dataset
    chain = indexed2.get_transform_chain()
    assert isinstance(chain, ThermoTransform)
    for i, container in enumerate(indexed2):
        assert chain(indexed2.load_raw_data(i)) == container

    # Check if transform chain is applied correctly in the grandchild dataset
    chain = indexed3.get_transform_chain()
    assert isinstance(chain, ThermoTransform)
    for i, container in enumerate(indexed3):
        assert chain(indexed3.load_raw_data(i)) == container
