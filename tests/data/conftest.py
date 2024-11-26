import pytest
from pythermondt.data import DataContainer

@pytest.fixture
def empty_data_container():
    """Fixture for DataContainer."""
    return DataContainer()