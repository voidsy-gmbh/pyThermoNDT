import torch
import math
from torch import Tensor
from numpy import ndarray
from typing import List, Optional
from pythermondt.data.datacontainer.core import DataContainer
from .utils import ThermoTransform
from ..data.units import Units

class VdsyGoertzel(ThermoTransform):
    def __init__(self, freq_bins: List[int] | Tensor | ndarray, frame_rate: Optional[int] = None):
        """ Applies DFT to the DataContainer and calculates the phase images for the specified frequencies, using the Goertzel algorithm.

        It uses the Temperature Data and the DomainValues to apply the DFT to the container and calculate the phase images for the specified frequencies. 
        The phase images are stored as new Temperature Data under "/Data/Tdata" and the frequency bins are stored as new DomainValues.
        The Temperatue data is deleted from the container.

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
        # Get the data from the container
        data, domain_values = container.get_datasets("/Data/Tdata", "/MetaData/DomainValues")

        # Check if frame_rate is given ==> else calculate it from the datacontainer domain values
        if self.frame_rate:
            frame_rate = self.frame_rate
        else:
            frame_rate = data.shape[2]/domain_values[-1].item()

        # Get shape of the data
        input_shape = data.shape
        nbr_of_pixels = input_shape[0]*input_shape[1]
        nbr_of_bins = len(self.freq_bins)

        # Initialize the phase images and the frequency bins
        phase_imgs = torch.zeros(input_shape[0], input_shape[1], nbr_of_bins, dtype=torch.float32)
        freqs = torch.zeros_like(self.freq_bins, dtype=torch.float32)

        # Reshape the data into a 2D array
        data = torch.reshape(data, shape=(nbr_of_pixels, input_shape[2]))
        
        # Calculate all the DFT bins we have to compute to include the specified frequencies
        for idx in range (0, nbr_of_bins):
            freq = (self.freq_bins[idx] * (1.0/input_shape[2])) * frame_rate # in Hz
            nbr_of_samples = int(torch.floor(1.0/freq * frame_rate)) #Number of samples for a single period of the required frequency
            f = 1.0/nbr_of_samples # Normalized frequency
            freqs[idx] = f*frame_rate

            w_real = 2.0 * math.cos(2.0*math.pi*f)
            w_imag = 1.0 * math.sin(2.0*math.pi*f)

            d1 = torch.zeros(size=(nbr_of_pixels,1))
            d2 = torch.zeros(size=(nbr_of_pixels,1))

            for jdx in range(0,nbr_of_samples):
                y = torch.unsqueeze(data[:,jdx],1) + w_real*d1-d2
                d2 = d1[:]
                d1 = y[:]
            
            phase_im = torch.arctan2(w_imag*d1, 0.5*w_real*d1-d2)
            phase_imgs[:,:,idx] = torch.reshape(phase_im,shape=(input_shape[0], input_shape[1]))

        # Update the container
        container.update_dataset("/Data/Tdata", phase_imgs)
        container.update_dataset("/MetaData/DomainValues", freqs)
        container.update_unit("/MetaData/DomainValues", Units.hertz)
        return container
