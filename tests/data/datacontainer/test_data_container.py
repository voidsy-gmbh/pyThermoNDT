from io import BytesIO

import numpy as np
import pytest
import torch
from torch import Tensor

from pythermondt.data import DataContainer

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
    assert isinstance(serialized, BytesIO)

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


def test_set_attribute_new(empty_container: DataContainer):
    """Test set_attribute adds attribute when it does not exist."""
    empty_container.add_group("/", "TestGroup")

    # Attribute does not exist yet - should add it
    empty_container.set_attribute("/TestGroup", "new_attr", "value1")
    assert empty_container.get_attribute("/TestGroup", "new_attr") == "value1"


def test_set_attribute_existing(empty_container: DataContainer):
    """Test set_attribute updates attribute when it already exists."""
    empty_container.add_group("/", "TestGroup")
    empty_container.add_attribute("/TestGroup", "existing_attr", "original")

    # Attribute exists - should update it
    empty_container.set_attribute("/TestGroup", "existing_attr", "updated")
    assert empty_container.get_attribute("/TestGroup", "existing_attr") == "updated"


def test_set_attribute_multiple_times(empty_container: DataContainer):
    """Test set_attribute can be called multiple times without try/except."""
    empty_container.add_group("/", "TestGroup")

    # Multiple calls should work without errors
    empty_container.set_attribute("/TestGroup", "attr", "value1")
    assert empty_container.get_attribute("/TestGroup", "attr") == "value1"

    empty_container.set_attribute("/TestGroup", "attr", "value2")
    assert empty_container.get_attribute("/TestGroup", "attr") == "value2"

    empty_container.set_attribute("/TestGroup", "attr", "value3")
    assert empty_container.get_attribute("/TestGroup", "attr") == "value3"


def test_set_attributes_new(empty_container: DataContainer):
    """Test set_attributes adds multiple attributes when they do not exist."""
    empty_container.add_group("/", "TestGroup")

    empty_container.set_attributes("/TestGroup", attr1="value1", attr2=42, attr3=3.14)
    assert empty_container.get_attribute("/TestGroup", "attr1") == "value1"
    assert empty_container.get_attribute("/TestGroup", "attr2") == 42
    assert empty_container.get_attribute("/TestGroup", "attr3") == 3.14


def test_set_attributes_existing(empty_container: DataContainer):
    """Test set_attributes updates multiple attributes when they already exist."""
    empty_container.add_group("/", "TestGroup")
    empty_container.add_attributes("/TestGroup", attr1="original1", attr2="original2")

    empty_container.set_attributes("/TestGroup", attr1="updated1", attr2="updated2")
    assert empty_container.get_attribute("/TestGroup", "attr1") == "updated1"
    assert empty_container.get_attribute("/TestGroup", "attr2") == "updated2"


def test_set_attributes_mixed(empty_container: DataContainer):
    """Test set_attributes with mix of existing and new attributes."""
    empty_container.add_group("/", "TestGroup")
    empty_container.add_attribute("/TestGroup", "existing", "original")

    # Mix of existing and new attributes
    empty_container.set_attributes("/TestGroup", existing="updated", new_attr="brand_new")
    assert empty_container.get_attribute("/TestGroup", "existing") == "updated"
    assert empty_container.get_attribute("/TestGroup", "new_attr") == "brand_new"


def test_set_dataset_new(empty_container: DataContainer, sample_tensor: Tensor):
    """Test set_dataset adds dataset when it does not exist."""
    # Dataset does not exist yet - should add it
    empty_container.set_dataset("/", "NewData", sample_tensor)
    assert "NewData" in empty_container.get_all_dataset_names()
    assert torch.equal(empty_container.get_dataset("/NewData"), sample_tensor)


def test_set_dataset_existing(empty_container: DataContainer, sample_tensor: Tensor, sample_eye_tensor: Tensor):
    """Test set_dataset updates dataset when it already exists."""
    empty_container.add_dataset("/", "ExistingData", sample_tensor)

    # Dataset exists - should update it
    empty_container.set_dataset("/", "ExistingData", sample_eye_tensor)
    assert torch.equal(empty_container.get_dataset("/ExistingData"), sample_eye_tensor)


def test_set_dataset_multiple_times(empty_container: DataContainer, sample_tensor: Tensor, sample_eye_tensor: Tensor):
    """Test set_dataset can be called multiple times without try/except."""
    # Multiple calls should work without errors
    empty_container.set_dataset("/", "Data", sample_tensor)
    assert torch.equal(empty_container.get_dataset("/Data"), sample_tensor)

    empty_container.set_dataset("/", "Data", sample_eye_tensor)
    assert torch.equal(empty_container.get_dataset("/Data"), sample_eye_tensor)


def test_set_dataset_with_ndarray(empty_container: DataContainer, sample_ndarray: np.ndarray, sample_tensor: Tensor):
    """Test set_dataset accepts numpy arrays (converts to tensor)."""
    # Should accept ndarray and convert to tensor
    empty_container.set_dataset("/", "NdarrayData", sample_ndarray)
    result = empty_container.get_dataset("/NdarrayData")
    assert torch.equal(result, sample_tensor)


def test_set_datasets_new(empty_container: DataContainer, sample_tensor: Tensor, sample_eye_tensor: Tensor):
    """Test set_datasets adds multiple datasets when they do not exist."""
    empty_container.set_datasets("/", Data1=sample_tensor, Data2=sample_eye_tensor)
    assert "Data1" in empty_container.get_all_dataset_names()
    assert "Data2" in empty_container.get_all_dataset_names()
    assert torch.equal(empty_container.get_dataset("/Data1"), sample_tensor)
    assert torch.equal(empty_container.get_dataset("/Data2"), sample_eye_tensor)


def test_set_datasets_existing(empty_container: DataContainer, sample_tensor: Tensor, sample_eye_tensor: Tensor):
    """Test set_datasets updates multiple datasets when they already exist."""
    empty_container.add_datasets("/", Data1=sample_tensor, Data2=sample_eye_tensor)

    # Update both datasets
    empty_container.set_datasets("/", Data1=sample_eye_tensor, Data2=sample_tensor)
    assert torch.equal(empty_container.get_dataset("/Data1"), sample_eye_tensor)
    assert torch.equal(empty_container.get_dataset("/Data2"), sample_tensor)


def test_set_datasets_mixed(empty_container: DataContainer, sample_tensor: Tensor, sample_eye_tensor: Tensor):
    """Test set_datasets with mix of existing and new datasets."""
    empty_container.add_dataset("/", "ExistingData", sample_tensor)

    # Mix of existing and new datasets
    empty_container.set_datasets("/", ExistingData=sample_eye_tensor, NewData=sample_tensor)
    assert torch.equal(empty_container.get_dataset("/ExistingData"), sample_eye_tensor)
    assert torch.equal(empty_container.get_dataset("/NewData"), sample_tensor)


def test_set_attribute_parent_group_not_exist(empty_container: DataContainer):
    """Test set_attribute raises KeyError when parent group does not exist."""
    with pytest.raises(KeyError):
        empty_container.set_attribute("/NonExistentGroup", "attr", "value")


def test_set_attributes_parent_group_not_exist(empty_container: DataContainer):
    """Test set_attributes raises KeyError when parent group does not exist."""
    with pytest.raises(KeyError):
        empty_container.set_attributes("/NonExistentGroup", attr1="value1", attr2="value2")


def test_set_dataset_parent_group_not_exist(empty_container: DataContainer, sample_tensor: Tensor):
    """Test set_dataset raises KeyError when parent group does not exist."""
    with pytest.raises(KeyError):
        empty_container.set_dataset("/NonExistentGroup", "Data", sample_tensor)


def test_set_datasets_parent_group_not_exist(empty_container: DataContainer, sample_tensor: Tensor):
    """Test set_datasets raises KeyError when parent group does not exist."""
    with pytest.raises(KeyError):
        empty_container.set_datasets("/NonExistentGroup", Data=sample_tensor)


def test_set_dataset_new_with_none(empty_container: DataContainer):
    """Test set_dataset creates empty dataset when data is None and dataset does not exist."""
    empty_container.set_dataset("/", "EmptyData", None)
    assert "EmptyData" in empty_container.get_all_dataset_names()
    result = empty_container.get_dataset("/EmptyData")
    assert result.shape == torch.empty(0).shape


def test_set_dataset_existing_with_none(empty_container: DataContainer, sample_tensor: Tensor):
    """Test set_dataset sets data to empty when data is None and dataset exists."""
    empty_container.add_dataset("/", "ExistingData", sample_tensor)
    assert torch.equal(empty_container.get_dataset("/ExistingData"), sample_tensor)

    empty_container.set_dataset("/", "ExistingData", None)
    result = empty_container.get_dataset("/ExistingData")
    assert result.shape == torch.empty(0).shape


def test_set_attribute_existing_type_change_raises(empty_container: DataContainer):
    """Test set_attribute raises TypeError when type changes with check_type=True (default)."""
    empty_container.add_group("/", "TestGroup")
    empty_container.add_attribute("/TestGroup", "my_attr", "string_value")

    with pytest.raises(TypeError):
        empty_container.set_attribute("/TestGroup", "my_attr", 123)


def test_set_attribute_existing_type_change_allowed(empty_container: DataContainer):
    """Test set_attribute allows type change when check_type=False."""
    empty_container.add_group("/", "TestGroup")
    empty_container.add_attribute("/TestGroup", "my_attr", "string_value")

    empty_container.set_attribute("/TestGroup", "my_attr", 123, check_type=False)
    assert empty_container.get_attribute("/TestGroup", "my_attr") == 123


def test_set_attributes_type_change_raises(empty_container: DataContainer):
    """Test set_attributes raises TypeError when type changes with check_type=True (default)."""
    empty_container.add_group("/", "TestGroup")
    empty_container.add_attributes("/TestGroup", attr1="string_value", attr2=42)

    with pytest.raises(TypeError):
        empty_container.set_attributes("/TestGroup", check_type=True, attr1=999)


def test_set_attributes_type_change_allowed(empty_container: DataContainer):
    """Test set_attributes allows type changes when check_type=False."""
    empty_container.add_group("/", "TestGroup")
    empty_container.add_attributes("/TestGroup", attr1="string_value", attr2=42)

    empty_container.set_attributes("/TestGroup", check_type=False, attr1=999, attr2="new_string")
    assert empty_container.get_attribute("/TestGroup", "attr1") == 999
    assert empty_container.get_attribute("/TestGroup", "attr2") == "new_string"


# Only run the tests in this file if it is run directly
if __name__ == "__main__":
    pytest.main(["-v", __file__])
