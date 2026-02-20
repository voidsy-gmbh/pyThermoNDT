import pytest
import torch

from pythermondt.data import ThermoContainer


@pytest.fixture
def viewer_container() -> ThermoContainer:
    """Container fixture for visualization tests."""
    container = ThermoContainer()
    container.update_dataset("/Data/Tdata", torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))
    container.update_dataset("/MetaData/DomainValues", torch.linspace(0.0, 0.3, 4, dtype=torch.float32))
    container.add_attribute("/Data/Tdata", "Description", "Thermal data")
    container.add_group("/MetaData", "Acquisition")
    container.add_dataset("/MetaData/Acquisition", "FrameRate", torch.tensor([60], dtype=torch.int64))
    return container
