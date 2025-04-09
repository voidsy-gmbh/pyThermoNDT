import io
from unittest.mock import patch

import numpy as np
import pytest

from pythermondt.data import ThermoContainer
from pythermondt.io.parsers import SimulationParser


def test_simulation_parser_empty_bytes():
    """Test that SimulationParser raises appropriate error with empty BytesIO."""
    empty_bytes = io.BytesIO()
    with pytest.raises(ValueError, match="The given BytesIO object is empty"):
        SimulationParser.parse(empty_bytes)


def test_simulation_parser_invalid_bytes():
    """Test that SimulationParser raises appropriate error with invalid MAT file."""
    invalid_bytes = io.BytesIO(b"not a mat file")
    with pytest.raises(ValueError, match="The given BytesIO object does not contain a valid .mat file"):
        SimulationParser.parse(invalid_bytes)


@patch("pythermondt.io.parsers.simulation_parser.mat73.loadmat")
def test_simulation_parser_basic_parsing(mock_loadmat):
    """Test that SimulationParser correctly parses a simple MAT file."""
    # Setup mock data structure - use dict instead of MagicMock for keys
    sim_result = {
        "Tdata": np.zeros((10, 10, 5)),
        "Time": np.linspace(0, 1, 5),
        "GroundTruth": np.zeros((10, 10), dtype=bool),
        "LookUpTable": np.linspace(0, 100, 100),
        "ExcitationSignal": np.ones(5),
    }

    # Create a proper return structure for loadmat
    mock_loadmat.return_value = {"SimResult": sim_result}

    # Create a ThermoContainer to check against
    expected = ThermoContainer()
    expected.add_attributes(path="/MetaData", Source="Simulation")
    expected.update_dataset(path="/Data/Tdata", data=sim_result["Tdata"])
    expected.update_dataset(path="/MetaData/DomainValues", data=sim_result["Time"])
    expected.update_dataset(path="/GroundTruth/DefectMask", data=sim_result["GroundTruth"])
    expected.update_dataset(path="/MetaData/LookUpTable", data=sim_result["LookUpTable"])
    expected.update_dataset(path="/MetaData/ExcitationSignal", data=sim_result["ExcitationSignal"])

    # Parse the mock data
    parsed = SimulationParser.parse(io.BytesIO(b"dummy"))

    # Check containers match
    assert parsed == expected
