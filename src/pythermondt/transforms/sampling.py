from collections.abc import Sequence

import torch

from ..data import DataContainer
from .utils import ThermoTransform


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
        tdata, domain_values, excitation_signal = container.get_datasets(
            "/Data/Tdata", "/MetaData/DomainValues", "/MetaData/ExcitationSignal"
        )

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

        # Fix time shift in domain values by substracting the first time step
        domain_values = domain_values - domain_values[0]

        # Update Container and return
        container.update_datasets(
            ("/Data/Tdata", tdata),
            ("/MetaData/DomainValues", domain_values),
            ("/MetaData/ExcitationSignal", excitation_signal),
        )
        return container


class SelectFrameRange(ThermoTransform):
    """Select a range of frames from the data container, by specifying their start and end index."""

    def __init__(self, start: int | None = None, end: int | None = None):
        """Select a range of frames from the data container, by specifying their start and end index.

        Parameters:
            start (int, optional): Start index of the frame range. Default is None, which means the start index is 0.
            end (int, optional): End index of the frame range, which is inclusiv.
                Default is None, which means the end index is the last frame.
        """
        super().__init__()
        self.start = start
        self.end = end

    def forward(self, container: DataContainer) -> DataContainer:
        # Extract Datasets
        tdata, domain_values, excitation_signal = container.get_datasets(
            "/Data/Tdata", "/MetaData/DomainValues", "/MetaData/ExcitationSignal"
        )

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

        # Fix time shift in domain values by substracting the first time step
        domain_values = domain_values - domain_values[0]

        # Update Container and return
        container.update_datasets(
            ("/Data/Tdata", tdata),
            ("/MetaData/DomainValues", domain_values),
            ("/MetaData/ExcitationSignal", excitation_signal),
        )
        return container


class NonUniformSampling(ThermoTransform):
    """Implement a non-uniform sampling strategy for the data container according to this paper:

    Efficient defect reconstruction from temporal non-uniform pulsed
    thermography data using the virtual wave concept: https://doi.org/10.1016/j.ndteint.2024.103200
    """

    def __init__(self, n_samples: int, tau: float | None = None):
        """Implement a non-uniform sampling strategy for the data container according to this paper:

        Efficient defect reconstruction from temporal non-uniform pulsed
        thermography data using the virtual wave concept: https://doi.org/10.1016/j.ndteint.2024.103200

        Parameters:
            n_samples (int): Number of samples to select from the original data.
            tau (float, optional): Time shift parameter that controls the non-uniform sampling distribution.
                          If None, will be approxmated automatically using binary search to satisfy
                          the minimum time step constraint from Equation (25) of the paper. Default is None.
        """
        super().__init__()
        self.n_samples = n_samples
        self.tau = tau

    def _calculate_tau(self, t_end: float, dt_min: float, n_samples_original: int) -> float:
        """Calculate minimum tau according to equation (25) using binary search."""
        low = dt_min  # use dt_min as lower bound
        high = t_end  # use t_end as a upper bond because tau >= t_end makes no sense
        precision = 1e-2

        # 1.) Binary search
        while high - low > precision:
            tau = (low + high) / 2
            t_diff = tau * ((t_end / tau + 1) ** (1 / (n_samples_original - 1)) - 1)

            # Update bounds
            if t_diff > dt_min:
                high = tau  # Narrow down to lower half
            else:
                low = tau  # Narrow down to upper half

        # return the calculated tau
        return (low + high) / 2

    def forward(self, container: DataContainer) -> DataContainer:
        # Extract Datasets
        tdata, domain_values, excitation_signal = container.get_datasets(
            "/Data/Tdata", "/MetaData/DomainValues", "/MetaData/ExcitationSignal"
        )

        # Check if we are in time domain
        if container.get_unit("/MetaData/DomainValues")["quantity"] != "time":
            raise ValueError("NonUniformSampling transform can only be applied to time domain data.")

        # Check if number of samples is valid
        if self.n_samples <= 0 or self.n_samples > len(domain_values):
            raise ValueError(
                f"Invalid number of samples. Number of samples must be in the range [1, {len(domain_values)}]."
            )

        # Calculate tau using binary search if not provided
        n_samples_original = len(domain_values)
        t_end = domain_values[-1]
        if not self.tau:
            tau = self._calculate_tau(
                t_end.item(), domain_values[1].item() - domain_values[0].item(), n_samples_original
            )
        else:
            tau = torch.tensor(self.tau)

        # Calculate time steps according to equation (6) in the paper
        k = torch.arange(self.n_samples)
        t_k = tau * ((t_end / tau + 1) ** (k / (self.n_samples - 1)) - 1)

        # Find the indices of the closest time steps in the domain values
        indices = torch.searchsorted(domain_values, t_k)

        # Clamp indices to the valid range
        indices = torch.clamp(indices, 0, n_samples_original - 1)

        # Select the frames according to the indices
        tdata = tdata[..., indices]
        domain_values = domain_values[indices]
        excitation_signal = excitation_signal[indices]

        # Update Container and return
        container.update_datasets(
            ("/Data/Tdata", tdata),
            ("/MetaData/DomainValues", domain_values),
            ("/MetaData/ExcitationSignal", excitation_signal),
        )
        return container
