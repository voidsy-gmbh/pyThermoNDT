import pytest
from torch import Tensor
from pythermondt.data import DataContainer, Units
from pythermondt.data.datacontainer.node import RootNode, GroupNode, DataNode

@pytest.fixture
def root_node():
    """Fixture for RootNode."""
    return RootNode()

@pytest.fixture
def group_node():
    """Fixture for GroupNode."""
    return GroupNode("test_group")

@pytest.fixture
def data_node(sample_tensor: Tensor):
    """Fixture for DataNode."""
    return DataNode("test_data", sample_tensor)

@pytest.fixture
def empty_container():
    """Fixture for DataContainer."""
    return DataContainer()

@pytest.fixture
def filled_container(empty_container: DataContainer, sample_tensor: Tensor, sample_eye_tensor: Tensor):  
    """Container with basic structure for testing BaseOps"""
    # Add a testgroup
    empty_container.add_group("/", "TestGroup")

    # Add a nested group 
    empty_container.add_group("/TestGroup", "NestedGroup")

    # Add datasets
    empty_container.add_dataset("/", "TestDataset", sample_tensor)
    empty_container.add_dataset("/TestGroup", "TestDataset1", sample_tensor)
    empty_container.add_dataset("/TestGroup/NestedGroup", "TestDataset2", sample_eye_tensor)

    return empty_container

@pytest.fixture
def complex_container(filled_container: DataContainer):
    """Fixture for DataContainer with complex structure. Based on filled_container, with additional atttributes added"""
    # Define attributes to be added
    attrs = {
        "str_attr": "test_string",
        "int_attr": 42,
        "float_attr": 3.14,
        "list_attr": [1, 2, 3],
        "dict_attr": {"key": "value"},
        "unit_attr": Units.kelvin
    }

    # Add various types of attributes to the TestGroup
    filled_container.add_attributes("/TestGroup", **attrs)

    # Add various types of attributes to the NestedGroup
    filled_container.add_attributes("/TestGroup/NestedGroup", **attrs)

    # Add various types of attributes to the TestDataset
    filled_container.add_attributes("/TestGroup/TestDataset", **attrs)

    return filled_container