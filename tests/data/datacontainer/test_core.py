
import pytest
from torch import Tensor
from pythermondt.data import DataContainer

@pytest.fixture
def base_container(sample_tensor: Tensor, sample_eye_tensor: Tensor) -> DataContainer:  
    """Container with basic structure for testing BaseOps"""
    container = DataContainer()
    # Add a testgroup
    container.add_group("/", "TestGroup")

    # Add a nested group 
    container.add_group("/TestGroup", "NestedGroup")

    # Add datasets
    container.add_dataset("/TestGroup", "TestDataset", sample_tensor)
    container.add_dataset("/TestGroup/NestedGroup", "TestDataset2", sample_eye_tensor)

    return container

def test_basic_container_equality(empty_container: DataContainer, base_container: DataContainer):
    """Test basic container equality cases."""
    # Self-equality for empty and filled containers
    assert empty_container == empty_container
    assert base_container == base_container
    # Different containers should not be equal
    assert base_container != empty_container

def test_identical_structure_equality(base_container: DataContainer, sample_tensor: Tensor, sample_eye_tensor: Tensor):
    """Test equality of containers with identical structure but different objects."""
    identical_container = DataContainer()
    identical_container.add_group("/", "TestGroup")
    identical_container.add_group("/TestGroup", "NestedGroup")
    identical_container.add_dataset("/TestGroup", "TestDataset", sample_tensor)
    identical_container.add_dataset("/TestGroup/NestedGroup", "TestDataset2", sample_eye_tensor)
    
    assert base_container == identical_container

def test_different_data_inequality(base_container: DataContainer, sample_tensor: Tensor, sample_eye_tensor: Tensor):
    """Test inequality of containers with same structure but different data."""
    different_data = DataContainer()
    different_data.add_group("/", "TestGroup")
    different_data.add_group("/TestGroup", "NestedGroup")
    different_data.add_dataset("/TestGroup", "TestDataset", sample_eye_tensor)  # Swapped tensors
    different_data.add_dataset("/TestGroup/NestedGroup", "TestDataset2", sample_tensor)
    
    assert base_container != different_data

def test_different_structure_inequality(base_container: DataContainer, sample_tensor: Tensor):
    """Test inequality of containers with different structure."""
    different_structure = DataContainer()
    different_structure.add_group("/", "DifferentGroup")
    different_structure.add_dataset("/DifferentGroup", "TestDataset", sample_tensor)
    
    assert base_container != different_structure

def test_different_names_inequality(base_container: DataContainer, sample_tensor: Tensor, sample_eye_tensor: Tensor):
    """Test inequality of containers with different node names."""
    different_names = DataContainer()
    different_names.add_group("/", "TestGroup")
    different_names.add_group("/TestGroup", "NestedGroup")
    different_names.add_dataset("/TestGroup", "DifferentName", sample_tensor)
    different_names.add_dataset("/TestGroup/NestedGroup", "TestDataset2", sample_eye_tensor)
    
    assert base_container != different_names