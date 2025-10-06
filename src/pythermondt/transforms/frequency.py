from collections.abc import Sequence

import numpy as np
import torch

from ..data import DataContainer
from ..data.units import Units
from .base import ThermoTransform


class PulsePhaseThermography(ThermoTransform):
    """Transform thermal data from time domain to frequency domain using FFT.

    Applies FFT to temperature data and stores complex-valued frequency components
    at specified frequencies. The complex representation preserves both amplitude
    and phase information, which can be extracted as needed for analysis.
    """

    def __init__(self, freq_indices: Sequence[int] | None = None):
        """Initialize PPT transform.

        Args:
            freq_indices: Specific frequency bin indices to keep after FFT. If None, all frequencies are retained.
                Index 0 = DC component, 1 = first harmonic, etc.
        """
        super().__init__()
        self.freq_indices = list(freq_indices) if freq_indices else None

    def forward(self, container: DataContainer) -> DataContainer:
        # Extract data
        tdata, domain_values = container.get_datasets("/Data/Tdata", "/MetaData/DomainValues")

        # Verify time domain
        if container.get_unit("/MetaData/DomainValues")["quantity"] != "time":
            raise ValueError("PulsePhaseThermography requires time-domain data")

        # Compute FFT along time axis (dim=-1)
        # Use rfft to only get positive frequencies below the Nyquist limit
        fft_result = torch.fft.rfft(tdata, dim=-1)

        # Calculate frequency bins
        dt = domain_values[1] - domain_values[0]  # Time step
        freqs = torch.fft.rfftfreq(len(domain_values), d=dt.item())

        # Select specific frequencies if requested
        if self.freq_indices is not None:
            # Validate indices
            max_idx = fft_result.shape[-1]
            if any(idx < 0 or idx >= max_idx for idx in self.freq_indices):
                raise ValueError(f"freq_indices must be in range [0, {max_idx - 1}]")

            fft_result = fft_result[..., self.freq_indices]
            freqs = freqs[self.freq_indices]

        # Update container with complex FFT results
        container.update_dataset("/Data/Tdata", fft_result)
        container.update_dataset("/MetaData/DomainValues", freqs)
        container.update_unit("/Data/Tdata", Units.arbitrary)
        container.update_unit("/MetaData/DomainValues", Units.hertz)

        return container


class ExtractPhase(ThermoTransform):
    """Extract phase images from complex thermal data."""

    def __init__(self, unwrap: bool = True) -> None:
        super().__init__()
        self.unwrap = unwrap

    def forward(self, container: DataContainer) -> DataContainer:
        tdata = container.get_dataset("/Data/Tdata")

        if not torch.is_complex(tdata):
            raise ValueError("Data must be complex (FFT result)")

        # Extract phase and unwrap
        phase = torch.angle(tdata)
        if self.unwrap:
            # phase = np.unwrap(phase.numpy(), axis=-1)
            y = phase % (2 * np.pi)
            phase = torch.where(y > np.pi, 2 * np.pi - y, y)
        container.update_dataset("/Data/Tdata", phase)
        container.update_unit("/Data/Tdata", Units.dimensionless)
        return container
