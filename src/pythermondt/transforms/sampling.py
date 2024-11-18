from .utils import ThermoTransform
from ..data import DataContainer
from typing import Sequence

class SelectFrames(ThermoTransform):
    """Select a subset of frames from the data container specified by a single index or a list of indices."""
    def __init__(self, frame_indices: int | Sequence[int]):
        """Select a subset of frames from the data container specified by a single index or a list of indices.

        Parameters:
            frame_indices (int | Sequence[int]): Single index or list of indices to select frames.
        """
        super().__init__()
        self.frame_indices = frame_indices if isinstance(frame_indices, Sequence) else [frame_indices]

    def forward(self, container: DataContainer) -> DataContainer:
        # Extract Datasets
        tdata, domain_values, excitation_signal = container.get_datasets("/Data/Tdata", "/MetaData/DomainValues", "/MetaData/ExcitationSignal")

        # Check if frame_indices are valid
        if any(idx < 0 or idx >= tdata.shape[-1] for idx in self.frame_indices):
            raise ValueError(f"Invalid frame index. Frame indices must be in the range [0, {tdata.shape[-1] - 1}].")

        # Select Frames
        tdata = tdata[..., self.frame_indices]
        domain_values = domain_values[..., self.frame_indices]
        excitation_signal = excitation_signal[..., self.frame_indices]

        # Update Container and return
        container.update_datasets(("/Data/Tdata", tdata), ("/MetaData/DomainValues", domain_values), ("/MetaData/ExcitationSignal", excitation_signal))
        return container
