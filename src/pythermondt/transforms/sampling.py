from collections.abc import Sequence
from typing import Literal

import torch

from ..data import DataContainer
from .base import ThermoTransform


class SelectFrames(ThermoTransform):
    """Select a subset of frames from the data container specified by a single index or a list of indices."""

    def __init__(self, frame_indices: int | Sequence[int]):
        """Select a subset of frames from the data container specified by a single index or a list of indices.

        Args:
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

        # Check if tdata has a "frame" dimension (last axis)
        if tdata.ndim < 3:
            raise ValueError("SelectFrames transform requires tdata to have at least 3 dimensions (H, W, Frames).")

        # Select Frames
        tdata = tdata[..., self.frame_indices]
        domain_values = domain_values[..., self.frame_indices]
        excitation_signal = excitation_signal[..., self.frame_indices]

        # Fix time shift in domain values by subtracting the first time step
        domain_values = domain_values - domain_values[0]

        # Update Container and return
        # pylint: disable=duplicate-code
        container.update_datasets(
            ("/Data/Tdata", tdata),
            ("/MetaData/DomainValues", domain_values),
            ("/MetaData/ExcitationSignal", excitation_signal),
        )
        # pylint: enable=duplicate-code
        return container


class SelectFrameRange(ThermoTransform):
    """Select a range of frames from the data container, by specifying their start and end index."""

    def __init__(self, start: int | None = None, end: int | None = None):
        """Select a range of frames from the data container, by specifying their start and end index.

        Args:
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

        # Check if tdata has a "frame" dimension (last axis)
        if tdata.ndim < 3:
            raise ValueError("SelectFrameRange transform requires tdata to have at least 3 dimensions (H, W, Frames).")

        # Select Frames (end index is inclusive)
        start = self.start if self.start is not None else 0
        end = self.end + 1 if self.end is not None else tdata.shape[-1]
        tdata = tdata[..., start:end]
        domain_values = domain_values[..., start:end]
        excitation_signal = excitation_signal[..., start:end]

        # Fix time shift in domain values by subtracting the first time step
        domain_values = domain_values - domain_values[0]

        # Update Container and return
        container.update_datasets(
            ("/Data/Tdata", tdata),
            ("/MetaData/DomainValues", domain_values),
            ("/MetaData/ExcitationSignal", excitation_signal),
        )
        return container


class NonUniformSampling(ThermoTransform):
    """Implement a non-uniform sampling strategy for the data container.

    The implementation is based on the following paper:
    Efficient defect reconstruction from temporal non-uniform pulsed
    thermography data using the virtual wave concept: https://doi.org/10.1016/j.ndteint.2024.103200
    """

    def __init__(
        self,
        n_samples: int,
        tau: float | None = None,
        interpolate: Literal["nearest", "linear", "averaging"] = "linear",
        precision: float = 1e-2,
    ):
        """Implement a non-uniform sampling strategy for the data container.

        The implementation is based on the following paper:
        Efficient defect reconstruction from temporal non-uniform pulsed
        thermography data using the virtual wave concept: https://doi.org/10.1016/j.ndteint.2024.103200

        Args:
            n_samples (int): Number of samples to select from the original data.
            tau (float, optional): Time shift parameter that controls the non-uniform sampling distribution.
            If None, will be automatically calculated using binary search to satisfy the minimum time step
            constraint from Equation (25) of the paper. Default is None.
            interpolate (str, optional): Interpolation method to use after non-uniform sampling. Options are:
            - "nearest": Find the closest time steps in the original data using torch.searchsorted.
            - "linear": Apply linear interpolation to match the exact time steps calculated according to
              Equation (6) of the paper.
            - "averaging": Bin the original data into exponentially spaced intervals and average within
              each bin to reduce aliasing effects.
            Default is "linear".
            precision (float, optional): Precision used for the binary search when automatically calculating tau.
            Default is 1e-2, which is sufficient for most applications.
        """
        super().__init__()
        self.n_samples = n_samples
        self.tau = tau
        self.interpolate = interpolate
        self.precision = precision

    def _calculate_tau(self, t_end: float, dt_min: float, n_t: int) -> float:
        """Find the minimum tau that satisfies equation (25) using binary search.

        Args:
            t_end (float): End time of the original data.
            dt_min (float): Minimum time step required.
            n_t (int): Number of desired time steps after downsampling.

        Returns:
            float: The minimum tau value that satisfies the constraint.
        """
        low = dt_min  # use dt_min as lower bound
        high = t_end  # use t_end as a upper bond because tau >= t_end makes no sense

        # 1.) Binary search
        while high - low > self.precision:
            tau = (low + high) / 2
            t_diff = tau * ((t_end / tau + 1) ** (1 / (n_t - 1)) - 1)

            # Update bounds
            if t_diff > dt_min:
                high = tau  # Equation satisfied: tau works, try smaller tau ==> narrow down to lower half
            else:
                low = tau  # Equation violated: tau too small, need larger tau ==> narrow down to upper half

        # Verify the result actually works
        tau = high  # take the upper bound instead of mean as final tau because it is guaranteed to satisfy the equation
        t_diff_final = tau * ((t_end / tau + 1) ** (1 / (n_t - 1)) - 1)

        if t_diff_final < dt_min:
            raise ValueError(
                f"Binary search failed to find valid tau. Got t_diff={t_diff_final:.6f} < "
                f"required dt_min={dt_min:.6f}. Please provide tau manually."
            )

        return tau

    def _interp_vectorized(self, x_new, x_old, y_old_batch):
        """Vectorized 1D interpolation for batched data.

        Args:
            x_new: Target interpolation points, shape (n_new,)
            x_old: Original sample points, shape (n_old,)
            y_old_batch: Original values, shape (batch_size, n_old)

        Returns:
            Interpolated values, shape (batch_size, n_new)
        """
        # Find indices (same for all batch elements)
        indices = torch.searchsorted(x_old, x_new, right=False)
        indices = torch.clamp(indices, 1, len(x_old) - 1)

        # Get surrounding points
        x0, x1 = x_old[indices - 1], x_old[indices]
        y0 = y_old_batch[:, indices - 1]  # Shape: (batch_size, n_new)
        y1 = y_old_batch[:, indices]  # Shape: (batch_size, n_new)

        # Vectorized interpolation
        t = ((x_new - x0) / (x1 - x0)).to(y_old_batch.dtype)  # Shape: (n_new,)
        return y0 + t * (y1 - y0)  # Broadcasting handles batch dimension

    def _average_binned(self, t_k, domain_values, data):
        """Vectorized binning and averaging using torch.bucketize."""
        # Create bin edges as midpoints between target time steps ==> optimal time steps are centered in each bin
        bin_edges = torch.zeros(len(t_k) + 1, dtype=domain_values.dtype)
        bin_edges[0] = domain_values[0] - 1e-10  # Ensure first sample is included
        bin_edges[-1] = domain_values[-1] + 1e-10  # Ensure last sample is included
        bin_edges[1:-1] = (t_k[:-1] + t_k[1:]) / 2

        # Assign each time point to a bin
        bin_indices = torch.bucketize(domain_values, bin_edges, right=False) - 1
        bin_indices = torch.clamp(bin_indices, 0, len(t_k) - 1)

        # Flatten the sequence for vectorized processing
        original_shape = data.shape
        data_flat = data.view(-1, data.shape[-1])  # Shape: (num_locations, time)

        # One hot encode bins and average
        bin_oh = torch.nn.functional.one_hot(bin_indices, num_classes=len(t_k)).to(data.dtype)  # Shape: (time, n_bins)
        summed = data_flat @ bin_oh  # Shape: (num_locations, n_bins)
        counts = bin_oh.sum(dim=0).clamp(min=1)  # Shape: (n_bins,)
        result = summed / counts  # Shape: (num_locations, n_bins)

        return result.view(original_shape[:-1] + (len(t_k),))

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
        t_end = domain_values[-1]
        dt_min = domain_values[1] - domain_values[0]  # Assuming the input data is uniformly sampled
        if not self.tau:
            tau = torch.tensor(self._calculate_tau(t_end.item(), dt_min.item(), self.n_samples))
        else:
            tau = torch.tensor(self.tau)

        # Calculate time steps according to equation (6) in the paper
        k = torch.arange(self.n_samples, dtype=domain_values.dtype)
        t_k = tau * ((t_end / tau + 1) ** (k / (self.n_samples - 1)) - 1)

        match self.interpolate:
            case "nearest":
                # Find the indices of the closest time steps in the domain values
                indices = torch.searchsorted(domain_values, t_k)

                # Clamp indices to the valid range
                indices = torch.clamp(indices, 0, len(domain_values) - 1)

                # Select the frames according to the indices
                tdata = tdata[..., indices]
                domain_values = domain_values[indices]
                excitation_signal = excitation_signal[indices]
            case "linear":
                # Interpolate signals
                # Excitation signal
                excitation_signal = self._interp_vectorized(t_k, domain_values, excitation_signal.unsqueeze(0)).squeeze(
                    0
                )

                # Interpolate tdata (vectorized across all spatial locations)
                h, w, _ = tdata.shape
                tdata_flat = tdata.view(h * w, -1)  # Shape: (h*w, time)
                tdata = self._interp_vectorized(t_k, domain_values, tdata_flat).view(h, w, self.n_samples)

                domain_values = t_k  # Use exact exponential times as new domain values
            case "averaging":
                # Bin and average signals
                # Excitation signal
                excitation_signal = self._average_binned(t_k, domain_values, excitation_signal.unsqueeze(0)).squeeze(0)
                # Average tdata (vectorized across all spatial locations)
                h, w, _ = tdata.shape
                tdata_flat = tdata.view(h * w, -1)  # Shape: (h*w, time)
                tdata = self._average_binned(t_k, domain_values, tdata_flat).view(h, w, self.n_samples)
                domain_values = t_k  # Use exact exponential times as new domain values

        # Update Container and return
        container.update_datasets(
            ("/Data/Tdata", tdata),
            ("/MetaData/DomainValues", domain_values),
            ("/MetaData/ExcitationSignal", excitation_signal),
        )
        return container
