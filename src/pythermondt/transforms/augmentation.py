import torch
from ..data import DataContainer
from .utils import ThermoTransform

class RandomFlip(ThermoTransform):
    ''' 
    Randomly flips all frames in the Temperature data (Tdata) of the container along the height (vertical flip) or width (horizontal flip), or both, based on specified probabilities.
    '''
    def __init__(self, p_height: float = 0.5, p_width: float = 0.5):
        ''' 
        Initializes the RandomFlip transformation with specified probabilities for flipping along the height and width directions.
        
        Parameters:
            p_height (float): Probability of flipping along the height (vertical flip), in the range [0, 1]. Default is 0.5.
            p_width (float): Probability of flipping along the width (horizontal flip), in the range [0, 1]. Default is 0.5.
        '''
        super().__init__()

        # Check if p_height and p_width are probabilities
        if not 0 <= p_height <= 1:
            raise ValueError("p_height must be in the range [0, 1]")
        
        if not 0 <= p_width <= 1:
            raise ValueError("p_width must be in the range [0, 1]")

        # Store the probabilities
        self.p_height = p_height
        self.p_width = p_width

    def forward(self, container: DataContainer) -> DataContainer:
        # Extract the data
        tdata, mask = container.get_datasets("/Data/Tdata", "/GroundTruth/DefectMask")
        
        # Flip the data along the height if the random number is less than p_height
        if torch.rand(1).item() < self.p_height:
            tdata = torch.flip(tdata, [1])
            mask = torch.flip(mask, [1])
        
        # Flip the data along the width if the random number is less than p_width
        if torch.rand(1).item() < self.p_width:
            tdata = torch.flip(tdata, [0])
            mask = torch.flip(mask, [0])

        # Update the container and return it
        container.update_datasets(("/Data/Tdata", tdata), ("/GroundTruth/DefectMask", mask))
        return container