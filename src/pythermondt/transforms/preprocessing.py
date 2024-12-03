import torch
from ..data import DataContainer
from .utils import ThermoTransform
from ..data.units import Units

class ApplyLUT(ThermoTransform):
    ''' 
    Applies the LookUpTable of the container to the Temperature data (Tdata) in the container. Therefore Tdata gets converted from uint16 to float64.
    '''
    def __init__(self):
        ''' 
        Applies the LookUpTable of the container to the Temperature data (Tdata) in the container. Therefore Tdata gets converted from uint16 to float64.
        '''
        super().__init__()
    
    def forward(self, container: DataContainer) -> DataContainer:
        # Extract the data
        lut = container.get_dataset("/MetaData/LookUpTable")
        tdata = container.get_dataset("/Data/Tdata")

        # Check if LUT is available
        if lut is None or tdata is None:
            raise ValueError("LookUpTable or Tdata is not available in the container.")
        
        # Check if LUT has already been applied
        if container.get_unit("/Data/Tdata") != Units.arbitrary and torch.is_floating_point(tdata):
            raise ValueError("LookUpTable has already been applied to the Temperature data.")

        # Check if the data is of the correct type
        if not isinstance(lut, torch.Tensor):
            raise ValueError("LookUpTable is not a torch.Tensor")
        if not isinstance(tdata, torch.Tensor):
            raise ValueError("Tdata is not a torch.Tensor")
        
        # Convert Tdata from uin16 to int32, because indexing in pytorch does not work with unsigned integers
        tdata = tdata.to(torch.int32)

        # Check for index out of bounds
        if tdata.min() < 0 or tdata.max() >= lut.shape[0]:
            raise IndexError("Index out of bounds. Tdata contains indices that are not available in the LookUpTable.")

        # Apply the LUT to Tdata
        tdata = lut[tdata]

        # Update the container and return it
        container.update_dataset("/Data/Tdata", tdata)
        container.update_unit("/Data/Tdata", Units.kelvin)
        return container

class SubstractFrame(ThermoTransform):
    ''' 
    Substracts 1 frame from all other frames in the Temperature data (Tdata) of the container.
    '''
    def __init__(self, frame: int = 0):
        ''' 
        Substracts 1 frame from all other frames in the Temperature data (Tdata) of the container.
        
        Parameters:
            frame (int): Frame number that should be substracted from the Temperature data. Default is the initial frame (frame 0).
        '''
        super().__init__()

        # Check if frame is a positive integer
        if frame < 0 or not isinstance(frame, int):
            raise ValueError("Frame must be a positive integer.")
               
        self.frame = frame
    
    def forward(self, container: DataContainer) -> DataContainer:
        # Extract the data
        tdata = container.get_dataset("/Data/Tdata")

        # Check if data is available
        if tdata is None:
            raise ValueError("Tdata is not available in the container.")
        
        # Check if the data is of the correct type
        if not isinstance(tdata, torch.Tensor):
            raise ValueError("Tdata is not a torch.Tensor")
        
        # Check for index out of bounds
        if self.frame >= tdata.shape[2]:
            raise IndexError("Index out of bounds. Frame number is bigger than the number of frames in the data.")
    
        # Substract the frame from Tdata
        tdata = tdata - tdata[:, :, self.frame].unsqueeze(2)

        # Update the container and return it
        container.update_dataset("/Data/Tdata", tdata)
        return container
    
class RemoveFlash(ThermoTransform):
    """ Automatically detect the flash and removes the frames before.
    """
    def __init__(self, offset: int = 0):
        super().__init__()
        self.offset = offset
    
    def forward(self, container: DataContainer) -> DataContainer:
        # Extract the data
        tdata, excitation_signal, domain_values = container.get_datasets("/Data/Tdata", "/MetaData/ExcitationSignal", "/MetaData/DomainValues")

        # Detect the flash ==> find frame where excitation signal goes from 1 back to 0 (flash end)
        flash_end_idx = None
        for i in range(len(excitation_signal)):
            if excitation_signal[i - 1] == 1 and excitation_signal[i] == 0:
                flash_end_idx = i + self.offset
                break
        
        if flash_end_idx is None:
            raise ValueError("Flash could not be detected in the excitation signal.")
        
        # Keep only the frames after the flash
        tdata = tdata[..., flash_end_idx:]
        domain_values = domain_values[flash_end_idx:]
        excitation_signal = excitation_signal[flash_end_idx:]

        # Fix time shift in domain values by substracting the first time step
        domain_values = domain_values - domain_values[0]

        # Update the container and return it
        container.update_datasets(("/Data/Tdata", tdata), ("/MetaData/DomainValues", domain_values), ("/MetaData/ExcitationSignal", excitation_signal))
        return container