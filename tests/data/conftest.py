import pytest
from pythermondt.data import DataContainer, ThermoContainer

@pytest.fixture
def empty_data_container():
    """Fixture for DataContainer."""
    return DataContainer()

@pytest.fixture
def thermo_container():
    """Fixture for ThermoContainer."""
    return ThermoContainer()