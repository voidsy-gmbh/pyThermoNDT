import copy

import pytest
from torch import Tensor

from pythermondt.data import DataContainer, ThermoContainer


@pytest.fixture
def thermo_container():
    """Fixture for ThermoContainer."""
    return ThermoContainer()


@pytest.fixture
def empty_container():
    """Fixture for empty DataContainer."""
    return DataContainer()


@pytest.fixture
def filled_container(empty_container: DataContainer, sample_tensor: Tensor, sample_eye_tensor: Tensor):
    """Container with basic structure for testing."""
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
