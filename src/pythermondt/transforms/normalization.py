import torch
from ..data import DataContainer
from .utils import ThermoTransform

class MinMaxNormalize(ThermoTransform):
    ''' 
    Normalizes the Temperature data (Tdata) in the container to range [0, 1] by using the min and max values of the data.
    This is done by subtracting the min value from the data and dividing by the difference between the max and min values.
    '''
    def __init__(self, eps: float = 1e-12):
        ''' 
        Normalizes the Temperature data (Tdata) in the container to range [0, 1], by using the min and max values of the data.

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
    
class MaxNormalize(ThermoTransform):
    ''' 
    Normalizes the Temperature data (Tdata) in the container to range [0, 1] by using the max value of the data.
    This is done by dividing the data by the max value of the data.
    '''
    def __init__(self, eps: float = 1e-12):
        ''' Normalizes the Temperature data (Tdata) in the container to range [0, 1], by using the max value of the data.

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

        # Get max value of the tensor and normalize the data
        max_val = tdata.max()
        tdata = tdata / (max_val + self.eps)

        # Update the container and return it
        container.fill_dataset("Data/Tdata", tdata)
        return container
    
class ZScoreNormalize(ThermoTransform):
    ''' 
    Normalizes the Temperature data (Tdata) in the container to have mean 0 and standard deviation 1.
    This is done by subtracting the mean value from the data and dividing by the standard deviation of the data.
    '''
    def __init__(self, eps: float = 1e-12):
        ''' Normalizes the Temperature data (Tdata) in the container to have mean 0 and standard deviation 1.

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

        # Get mean and standard deviation of the tensor and normalize the data
        mean_val = tdata.mean()
        std_val = tdata.std()
        tdata = (tdata - mean_val) / (std_val + self.eps)

        # Update the container and return it
        container.fill_dataset("Data/Tdata", tdata)
        return container