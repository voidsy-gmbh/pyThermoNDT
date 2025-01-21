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
def filled_container(sample_tensor: Tensor, sample_eye_tensor: Tensor):  
    """Container with basic structure for testing BaseOps"""
    # Initialize an empty container
    container = DataContainer()

    # Add a testgroup
    container.add_group("/", "TestGroup")

    # Add a nested group 
    container.add_group("/TestGroup", "NestedGroup")

    # Add datasets
    container.add_dataset("/", "TestDataset", sample_tensor)
    container.add_dataset("/TestGroup", "TestDataset1", sample_tensor)
    container.add_dataset("/TestGroup/NestedGroup", "TestDataset2", sample_eye_tensor)

    return container

@pytest.fixture
def complex_container():
    """Fixture for DataContainer with complex structure. Based on filled_container, with additional atttributes added"""
    # Initialize an empty container
    container = DataContainer()

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
    container.add_attributes("/TestGroup", **attrs)

    # Add various types of attributes to the NestedGroup
    container.add_attributes("/TestGroup/NestedGroup", **attrs)

    # Add various types of attributes to the TestDataset
    container.add_attributes("/TestGroup/TestDataset", **attrs)

    return container