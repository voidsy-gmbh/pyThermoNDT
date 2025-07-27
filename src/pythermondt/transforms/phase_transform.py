import math
from collections.abc import Sequence

import torch
from numpy import ndarray
from torch import Tensor

from pythermondt.data.datacontainer.core import DataContainer

from ..data.units import Units
from .utils import ThermoTransform


class PulsePhaseTransform(ThermoTransform):
    def __init__(self, freq_bins: Sequence[float] | Tensor | ndarray, frame_rate: int | None = None):
        """Applies DFT to the DataContainer and calculates the phase images for the specified frequencies, using the Goertzel algorithm.

        It uses the Temperature Data and the DomainValues to apply the DFT to the container and calculate the phase images for the specified frequencies.
        The phase images are stored as new Temperature Data under "/Data/Tdata" and the frequency bins are stored as new DomainValues.
        The Temperature data is deleted from the container.

        Note: This DFT differs from a conventional fourier transform because the signal length that is used to calculate the phase image is always reduced to match a single period at the respective frequency!

        URLS:
        https://stackoverflow.com/questions/13499852/scipy-fourier-transform-of-a-few-selected-frequencies
        https://gist.github.com/sebpiq/4128537
        https://de.wikipedia.org/wiki/Goertzel-Algorithmus

        Parameters:
            frame_rate (int): The frame rate of the data. This is required to determine the actual evaluation frequencies
            freq_bins (List[int]): The number specifies the respective frequency bin (Note: bin "0" contains just the DC component and is a uniform image!)
        """
        super().__init__()
        self.frame_rate = frame_rate

        # Convert freq_bins to a tensor
        if isinstance(freq_bins, list):
            self.freq_bins = torch.tensor(freq_bins, dtype=torch.float32)

        elif isinstance(freq_bins, ndarray):
            self.freq_bins = torch.from_numpy(freq_bins).float()

        elif isinstance(freq_bins, Tensor):
            self.freq_bins = freq_bins

    def forward(self, container: DataContainer) -> DataContainer:
        # 1. Data preparation
        data, domain_values = container.get_datasets("/Data/Tdata", "/MetaData/DomainValues")
        frame_rate = self.frame_rate or (data.shape[2] / domain_values[-1].item())

        # 2. Setup dimensions
        height, width, n_samples = data.shape
        n_pixels = height * width
        n_freqs = len(self.freq_bins)

        # 3. Reshape for processing ==> HxWxT -> (H*W)xT
        data = data.reshape(n_pixels, n_samples)  # (H*W)xT

        # 4 Create the container for the phase images and actual frequencies ==> for results
        phase_imgs = torch.zeros(height, width, n_freqs, dtype=torch.float32)
        actual_freqs = torch.zeros(n_freqs, dtype=torch.float32)  # Store actual frequencies

        # 4. Process each frequency
        for idx, freq in enumerate(self.freq_bins):
            # Calculate normalized frequency
            samples_per_period = int(frame_rate / freq)  # Direct calculation
            f = freq / frame_rate  # Normalized frequency

            # Goertzel coefficients
            w_real = 2.0 * math.cos(2.0 * math.pi * f)
            w_imag = math.sin(2.0 * math.pi * f)

            # Process all pixels for this frequency
            d1 = torch.zeros(n_pixels, 1)
            d2 = torch.zeros(n_pixels, 1)

            # Use minimum between period samples and available samples
            n_process = min(samples_per_period, n_samples)
            for j in range(n_process):
                y = data[:, j : j + 1] + w_real * d1 - d2
                d2 = d1.clone()
                d1 = y.clone()

            # Extract phase and actual frequency
            phase = torch.arctan2(w_imag * d1, 0.5 * w_real * d1 - d2)
            phase_imgs[:, :, idx] = phase.reshape(height, width)
            actual_freqs[idx] = freq

        # Update the container
        container.update_dataset("/Data/Tdata", phase_imgs)
        container.update_dataset("/MetaData/DomainValues", actual_freqs)
        container.update_unit("/MetaData/DomainValues", Units.hertz)
        return container
