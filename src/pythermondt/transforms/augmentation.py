import torch
from ..data import DataContainer
from .utils import ThermoTransform

class RandomFlip(ThermoTransform):
    ''' 
    Randomly flips all frames in the Temperature data (Tdata) of the container along the x-axis, y-axis, or both, based on specified probabilities.
    '''
    def __init__(self, p_x: float = 0.5, p_y: float = 0.5):
        ''' 
        Initializes the RandomFlip transformation with specified probabilities for flipping along the x-axis and y-axis.
        
        Parameters:
            p_x (float): Probability of flipping along the x-axis, in the range [0, 1]. Default is 0.5.
            p_y (float): Probability of flipping along the y-axis, in the range [0, 1]. Default is 0.5.
        '''
        super().__init__()

        # Check if p_x and p_y are probabilities
        if not 0 <= p_x <= 1:
            raise ValueError("p_x must be in the range [0, 1]")
        
        if not 0 <= p_y <= 1:
            raise ValueError("p_y must be in the range [0, 1]")

        # Store the probabilities
        self.p_x = p_x
        self.p_y = p_y

    def forward(self, container: DataContainer) -> DataContainer:
        # Extract the data
        tdata, mask = container.get_datasets("/Data/Tdata", "/GroundTruth/DefectMask")
        
        # Flip the data along the x-axis if the random number is less than p_x
        if torch.rand(1).item() < self.p_x:
            tdata = torch.flip(tdata, [1])
            mask = torch.flip(mask, [1])
        
        # Flip the data along the y-axis if the random number is less than p_y
        if torch.rand(1).item() < self.p_y:
            tdata = torch.flip(tdata, [0])
            mask = torch.flip(mask, [0])

        # Update the container and return it
        container.update_dataset("/Data/Tdata", tdata)
        return container