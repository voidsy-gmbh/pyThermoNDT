import torch
from ..data import DataContainer
from .utils import ThermoTransform

class MinMaxNormalize(ThermoTransform):
    ''' 
    Normalizes the Temperature data (Tdata) in the container to range [0, 1] by using the min and max values of the data.
    '''
    def __init__(self, eps: float = 1e-12):
        ''' Normalizes the Temperature data (Tdata) in the container to range [0, 1], by using the min and max values of the data.

        Parameters:
        - eps (float): Small value added to the denominator to avoid division by zero. Default is 1e-12.
        '''
        super().__init__()
        if eps > 1e-3:
            print("Warning: eps is bigger than 1e-3. This might lead to unexpected results.")
        self.eps = eps
    
    def forward(self, container: DataContainer) -> DataContainer:
        # Extract the data
        tdata = container.get_dataset("Data/Tdata")

        # Get min and max values of the tensor and normalize the data
        min_val = tdata.min()
        max_val = tdata.max()
        tdata = (tdata - min_val) / (max_val - min_val + self.eps)

        # Update the container and return it
        container.fill_dataset("Data/Tdata", tdata)
        return container