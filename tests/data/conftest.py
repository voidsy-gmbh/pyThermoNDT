import pytest
from pythermondt.data import ThermoContainer

@pytest.fixture
def thermo_container():
    """Fixture for ThermoContainer."""
    return ThermoContainer()