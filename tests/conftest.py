import numpy as np
import pytest
import torch

from pythermondt import DataContainer, LocalReader
from pythermondt.transforms import ThermoTransform


@pytest.fixture
def sample_tensor():
    """Basic tensor fixture available to all tests."""
    return torch.tensor([[1, 2], [3, 4]])


@pytest.fixture
def sample_tensor2():
    """Basic tensor fixture available to all tests."""
    return torch.tensor([[5, 6], [7, 8]])


@pytest.fixture
def sample_empty_tensor():
    """Empty tensor fixture available to all tests."""
    return torch.empty(0)


@pytest.fixture
def sample_eye_tensor():
    """Identity tensor fixture available to all tests."""
    return torch.eye(3)


@pytest.fixture
def sample_ndarray():
    """Basic ndarray fixture available to all tests."""
    return np.array([[1, 2], [3, 4]])


@pytest.fixture
def sample_ndarray2():
    """Basic ndarray fixture available to all tests."""
    return np.array([[5, 6], [7, 8]])


@pytest.fixture
def sample_empty_ndarray():
    """Empty ndarray fixture available to all tests."""
    return np.empty(0)


@pytest.fixture
def sample_eye_ndarray():
    """Identity ndarray fixture available to all tests."""
    return np.eye(3)


@pytest.fixture
def localreader_no_files():
    """Fixture for a reader that has no files."""
    return LocalReader(pattern="MadeUpPattern")


@pytest.fixture
def localreader_with_file():
    """Fixture for a reader that has files."""
    return LocalReader(pattern="./tests/assets/integration/simulation/source1.mat")


@pytest.fixture
def localreader_with_glob():
    """Fixture for a reader that has files."""
    return LocalReader(pattern="./tests/assets/integration/simulation/*.mat")


@pytest.fixture
def localreader_with_directory():
    """Fixture for a reader that has files."""
    return LocalReader(pattern="./tests/assets/integration/simulation/")


@pytest.fixture
def simple_transform():
    """Create a simple ThermoTransform that adds an attribute."""

    class SimpleTransform(ThermoTransform):
        """A simple transform that increments a 'transformed' attribute."""

        def __init__(self, value: str):
            super().__init__()
            self.value = value

        def forward(self, container: DataContainer) -> DataContainer:
            if "transformed" in container.get_all_attributes("/MetaData"):
                container.update_attribute("/MetaData", "transformed", self.value)
            else:
                container.add_attribute("/MetaData", "transformed", self.value)
            return container

    return SimpleTransform
