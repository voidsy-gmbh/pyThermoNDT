import torch
from ..data import DataContainer
from .utils import ThermoTransform

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
        lut = container.get_dataset("MetaData/LookUpTable")
        tdata = container.get_dataset("Data/Tdata")

        # Check if data is available
        if lut is None or tdata is None:
            raise ValueError("LookUpTable or Tdata is not available in the container.")
        
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
        container.fill_dataset("Data/Tdata", tdata)
        return container

class SubstractFrame(ThermoTransform):
    ''' 
    Substracts 1 frame from all other frames in the Temperature data (Tdata) of the container.
    '''
    def __init__(self, frame: int = 0):
        ''' 
        Substracts 1 frame from all other frames in the Temperature data (Tdata) of the container.
        
        Parameters:
        - frame (int): Frame number that should be substracted from the Temperature data. Default is the initial frame (frame 0).
        '''
        super().__init__()

        # Check if frame is a positive integer
        if frame < 0 or not isinstance(frame, int):
            raise ValueError("Frame must be a positive integer.")
               
        self.frame = frame
    
    def forward(self, container: DataContainer) -> DataContainer:
        # Extract the data
        tdata = container.get_dataset("Data/Tdata")

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
        container.fill_dataset("Data/Tdata", tdata)
        return container