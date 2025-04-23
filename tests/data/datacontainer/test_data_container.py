import pytest
import torch
from torch import Tensor

from pythermondt.data import DataContainer
from pythermondt.utils import IOPathWrapper

from ...utils import containers_equal


def test_initialization(empty_container: DataContainer):
    """Test initialization of DataContainer."""
    assert len(empty_container.get_all_dataset_names()) == 0
    assert len(empty_container.get_all_groups()) == 0


def test_group_operations(empty_container: DataContainer):
    """Test group operations of DataContainer."""
    empty_container.add_group("/", "TestGroup")
    assert "TestGroup" in empty_container.get_all_groups()

    empty_container.remove_group("/TestGroup")
    assert "TestGroup" not in empty_container.get_all_groups()


def test_single_dataset_operations(empty_container: DataContainer, sample_tensor: Tensor):
    """Test single dataset operations of DataContainer."""
    # Test adding a single dataset
    empty_container.add_dataset("/", "TestData", sample_tensor)
    assert "TestData" in empty_container.get_all_dataset_names()

    # Test getting a single dataset
    retrieved_data = empty_container.get_dataset("/TestData")
    assert torch.equal(retrieved_data, sample_tensor)

    # Test updating a single dataset
    new_data = torch.tensor([[5, 6], [7, 8]])
    empty_container.update_dataset("/TestData", new_data)
    updated_data = empty_container.get_dataset("/TestData")
    assert torch.equal(updated_data, new_data)

    # Test removing a single dataset
    empty_container.remove_dataset("/TestData")
    assert "TestData" not in empty_container.get_all_dataset_names()


def test_multiple_dataset_operations(empty_container: DataContainer, sample_tensor: Tensor, sample_eye_tensor: Tensor):
    """Test multiple dataset operations of DataContainer."""
    # Test add_datasets
    empty_container.add_datasets("/", TestData1=sample_tensor, TestData2=sample_eye_tensor)
    assert "TestData1" in empty_container.get_all_dataset_names()
    assert "TestData2" in empty_container.get_all_dataset_names()

    # Test get_datasets
    retrieved_data1, retrieved_data2 = empty_container.get_datasets("/TestData1", "/TestData2")
    assert torch.equal(retrieved_data1, sample_tensor)
    assert torch.equal(retrieved_data2, sample_eye_tensor)

    # Test update_datasets
    empty_container.update_datasets(("/TestData1", sample_eye_tensor), ("/TestData2", sample_tensor))
    updated_data1, updated_data2 = empty_container.get_datasets("/TestData1", "/TestData2")
    assert torch.equal(updated_data1, sample_eye_tensor)
    assert torch.equal(updated_data2, sample_tensor)


def test_attribute_operations(empty_container: DataContainer):
    """Test attribute operations of DataContainer."""
    empty_container.add_group("/", "TestGroup")
    empty_container.add_attribute("/TestGroup", "test_attr", "test_value")
    assert empty_container.get_attribute("/TestGroup", "test_attr") == "test_value"

    # Test multiple attributes
    attrs = empty_container.get_attributes("/TestGroup", "test_attr")
    assert attrs[0] == "test_value"

    empty_container.update_attribute("/TestGroup", "test_attr", "new_value")
    assert empty_container.get_attribute("/TestGroup", "test_attr") == "new_value"

    # Test getting all attributes
    attrs = empty_container.get_all_attributes("/TestGroup")
    assert attrs["test_attr"] == "new_value"

    # Test removing an attribute
    empty_container.remove_attribute("/TestGroup", "test_attr")
    with pytest.raises(KeyError):
        empty_container.get_attribute("/TestGroup", "test_attr")


def test_serialization(empty_container: DataContainer, sample_tensor: Tensor):
    """Test serialization of DataContainer."""
    # Add test data
    empty_container.add_group("/", "TestGroup")
    empty_container.add_dataset("/TestGroup", "TestData", sample_tensor)
    empty_container.add_attribute("/TestGroup/TestData", "test_attr", "test_value")

    # Serialize
    serialized = empty_container.serialize_to_hdf5()
    assert isinstance(serialized, IOPathWrapper)

    # Deserialize
    new_container = DataContainer()
    new_container.deserialize(serialized)

    # Check if data is the same
    assert containers_equal(empty_container, new_container)


def test_error_handling(empty_container: DataContainer):
    """Test error handling of DataContainer."""
    # Test getting non-existent dataset
    with pytest.raises(KeyError):
        empty_container.get_dataset("/NonExistentData")

    # Test getting non-existent attribute
    with pytest.raises(KeyError):
        empty_container.get_attribute("/NonExistentGroup", "test_attr")

    # Try to add the same group twice
    with pytest.raises(KeyError):
        empty_container.add_group("/", "TestGroup")
        empty_container.add_group("/", "TestGroup")


# Only run the tests in this file if it is run directly
if __name__ == "__main__":
    pytest.main(["-v", __file__])
