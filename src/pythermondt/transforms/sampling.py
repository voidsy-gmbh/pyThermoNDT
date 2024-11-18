from .utils import ThermoTransform
from ..data import DataContainer
from typing import Union, List

class IndexFrameSelection(ThermoTransform):
    """Select a subset of frames from the data container based on index or slice notation."""
    def __init__(self, frame_indices: Union[int, slice, List[int]]):
        """Select a subset of frames from the data container based on index or slice notation.

        Parameters:
            frame_indices (Union[int, slice, List[int]]): The indices of the frames to select.
        """
        super().__init__()
        self.frame_indices = frame_indices

    def forward(self, container: DataContainer) -> DataContainer:
        tdata, domain_values, excitation_signal = container.get_datasets("/Data/Tdata", "/MetaData/DomainValues", "/MetaData/ExcitationSignal")

        # Handle different index types
        if isinstance(self.frame_indices, int):
            tdata = tdata[..., self.frame_indices:self.frame_indices+1]  # Keep dimension
            domain_values = domain_values[self.frame_indices:self.frame_indices+1]
            excitation_signal = excitation_signal[self.frame_indices:self.frame_indices+1]
        else:
            tdata = tdata[..., self.frame_indices]
            domain_values = domain_values[self.frame_indices]
            excitation_signal = excitation_signal[self.frame_indices]

        # Update Container and return
        container.update_datasets(("/Data/Tdata", tdata), ("/MetaData/DomainValues", domain_values), ("/MetaData/ExcitationSignal", excitation_signal))
        return container
