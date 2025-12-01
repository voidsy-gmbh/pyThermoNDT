import math
from collections.abc import Sequence

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
        dt = domain_values[1] - domain_values[0]
        # Validate uniform sampling (required for FFT frequency calculation)
        diffs = torch.diff(domain_values)
        if not torch.allclose(diffs, dt, rtol=1e-5):
            raise ValueError("Domain values must be uniformly spaced for FFT")
        freqs = torch.fft.rfftfreq(len(domain_values), d=dt.item())

        # Select specific frequencies if requested
        if self.freq_indices is not None:
            # Validate indices
            max_idx = fft_result.shape[-1]
            if any(idx < 0 or idx >= max_idx for idx in self.freq_indices):
                raise ValueError(f"freq_indices must be in range [0, {max_idx - 1}], got {self.freq_indices}")

            fft_result = fft_result[..., self.freq_indices]
            freqs = freqs[self.freq_indices]

        # Update container with complex FFT results
        container.update_dataset("/Data/Tdata", fft_result)
        container.update_dataset("/MetaData/DomainValues", freqs)
        container.update_unit("/Data/Tdata", Units.arbitrary)
        container.update_unit("/MetaData/DomainValues", Units.hertz)
        return container


class ExtractAmplitude(ThermoTransform):
    """Extract amplitude images from complex thermal data."""

    def forward(self, container: DataContainer) -> DataContainer:
        tdata = container.get_dataset("/Data/Tdata")

        if not torch.is_complex(tdata):
            raise ValueError("Data must be complex (FFT result). Please apply PulsePhaseThermography first.")

        # Extract amplitude
        amplitude = torch.abs(tdata)
        container.update_dataset("/Data/Tdata", amplitude)
        container.update_unit("/Data/Tdata", Units.dimensionless)
        return container


class ExtractPhase(ThermoTransform):
    """Extract phase images from complex thermal data."""

    def __init__(self, unwrap: bool = True) -> None:
        """Initialize phase extraction transform.

        Args:
            unwrap: If True, the phase information is unwrapped to avoid discontinuities. Defaults to True.
        """
        super().__init__()
        self.unwrap = unwrap

    def _unwrap(self, phase: torch.Tensor, discont: float = math.pi, dim: int = -1) -> torch.Tensor:
        """Unwrap phase information along the specified dimension. Matches numpy.unwrap behavior.

        Args:
            phase: Input tensor containing phase values in radians.
            discont: Discontinuity threshold, typically set to pi.
            dim: Dimension along which to unwrap the phase. Default is the last dimension.
        """
        # Normalize dim to positive index
        dim = dim if dim >= 0 else phase.ndim + dim

        # Compute differences along specified dimension
        d = torch.diff(phase, dim=dim)
        d_mod = (d + discont) % (2 * discont) - discont

        # Handle edge case where difference is exactly -discont
        mask = (d_mod == -discont) & (d > 0)
        d_mod = torch.where(mask, torch.full_like(d_mod, discont), d_mod)

        # Compute correction and accumulate
        corr = d_mod - d
        correction = torch.cumsum(corr, dim=dim)

        # Pad on the correct dimension to account for reduced size after diff
        pad_shape = list(correction.shape)
        pad_shape[dim] = 1
        zeros = torch.zeros(pad_shape, dtype=phase.dtype, device=phase.device)
        correction = torch.cat([zeros, correction], dim=dim)

        return phase + correction

    def forward(self, container: DataContainer) -> DataContainer:
        tdata = container.get_dataset("/Data/Tdata")

        if not torch.is_complex(tdata):
            raise ValueError("Data must be complex (FFT result). Please apply PulsePhaseThermography first.")

        # Extract phase and unwrap
        phase = torch.angle(tdata)
        if self.unwrap:
            phase = self._unwrap(phase, dim=-1)
        container.update_dataset("/Data/Tdata", phase)
        container.update_unit("/Data/Tdata", Units.dimensionless)
        return container
