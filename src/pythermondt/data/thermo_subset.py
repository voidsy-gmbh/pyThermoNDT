from typing import Optional
from collections.abc import Sequence
from torch.utils.data import Subset
from .thermo_dataset import ThermoDataset
from ..transforms import ThermoTransform
from .datacontainer import DataContainer

class ThermoSubset(Subset):
    """ Custom subset class used for splitting a ThermoDataset and applying a transform to the data.

    This is needed, so that different transforms can be applied to the a dataset, after it has been split (e.g. train and validation set)
    """
    def __init__(self, dataset: ThermoDataset, indices: Sequence[int], transform: Optional[ThermoTransform] = None):
        """ Initialize a custom PyTorch subset from a ThermoDataset.

        Parameters:
            dataset (ThermoDataset): The dataset to be split
            indices (Sequence[int]): Indices to be used for the subset
            transform (ThermoTransform, optional): Optional transform to be directly applied to the data when it is read
        """
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx) -> DataContainer:
        return super().__getitem__(idx) if not self.transform else self.transform(super().__getitem__(idx))