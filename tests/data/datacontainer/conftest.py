import copy

import pytest
from torch import Tensor

from pythermondt.data import DataContainer, Units
from pythermondt.data.datacontainer.node import DataNode, GroupNode, RootNode


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
    """Container with basic structure for testing BaseOps."""
    # Initialize an empty container using deepcopy to avoid modifying the previous fixture
    container = copy.deepcopy(empty_container)

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
def complex_container(filled_container: DataContainer):
    """Fixture for DataContainer with complex structure.

    Fixture is based on filled_container, with additional attributes added.
    """
    # Initialize an empty container using deepcopy to avoid modifying the previous fixture
    container = copy.deepcopy(filled_container)

    # Define attributes to be added
    attrs = {
        "str_attr": "test_string",
        "int_attr": 42,
        "float_attr": 3.14,
        "list_attr": [1, 2, 3],
        "dict_attr": {"key": "value"},
        "unit_attr": Units.kelvin,
    }

    # Add various types of attributes to the TestGroup
    container.add_attributes("/TestGroup", **attrs)

    # Add various types of attributes to the NestedGroup
    container.add_attributes("/TestGroup/NestedGroup", **attrs)

    # Add various types of attributes to the TestDatasets
    container.add_attributes("/TestDataset", **attrs)
    container.add_attributes("/TestGroup/TestDataset1", **attrs)
    container.add_attributes("/TestGroup/NestedGroup/TestDataset2", **attrs)

    return container
