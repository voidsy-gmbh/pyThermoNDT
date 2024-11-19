from .utils import ThermoTransform
from ..data import DataContainer
from typing import Sequence, Optional

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
        
        # Check if we are in time domain
        if container.get_unit("/MetaData/DomainValues")["quantity"] != "time":
            raise ValueError("SelectFrames transform can only be applied to time domain data.")

        # Select Frames
        tdata = tdata[..., self.frame_indices]
        domain_values = domain_values[..., self.frame_indices]
        excitation_signal = excitation_signal[..., self.frame_indices]

        # Update Container and return
        container.update_datasets(("/Data/Tdata", tdata), ("/MetaData/DomainValues", domain_values), ("/MetaData/ExcitationSignal", excitation_signal))
        return container

class SelectFrameRange(ThermoTransform):
    """Select a range of frames from the data container, by specifying their start and end index."""
    def __init__(self, start: Optional[int] = None, end: Optional[int] = None):
        """Select a range of frames from the data container, by specifying their start and end index.

        Parameters:
            start (Optional[int]): Start index of the frame range. Default is None, which means the start index is 0.
            end (Optional[int]): End index of the frame range, which is inclusiv. Default is None, which means the end index is the last frame.
        """
        super().__init__()
        self.start = start
        self.end = end

    def forward(self, container: DataContainer) -> DataContainer:
        # Extract Datasets
        tdata, domain_values, excitation_signal = container.get_datasets("/Data/Tdata", "/MetaData/DomainValues", "/MetaData/ExcitationSignal")

        # Check if frame range is valid
        if self.start is not None and (self.start < 0 or self.start >= tdata.shape[-1]):
            raise ValueError(f"Invalid start index. Start index must be in the range [0, {tdata.shape[-1] - 1}].")
        
        if self.end is not None and (self.end < 0 or self.end >= tdata.shape[-1]):
            raise ValueError(f"Invalid end index. End index must be in the range [0, {tdata.shape[-1] - 1}].")
        
        # Check if we are in time domain
        if container.get_unit("/MetaData/DomainValues")["quantity"] != "time":
            raise ValueError("SelectFrameRange transform can only be applied to time domain data.")

        # Select Frames (end index is inclusive)
        start = self.start if self.start is not None else 0
        end = self.end + 1 if self.end is not None else tdata.shape[-1]
        tdata = tdata[..., start:end]
        domain_values = domain_values[..., start:end]
        excitation_signal = excitation_signal[..., start:end]

        # Update Container and return
        container.update_datasets(("/Data/Tdata", tdata), ("/MetaData/DomainValues", domain_values), ("/MetaData/ExcitationSignal", excitation_signal))
        return container