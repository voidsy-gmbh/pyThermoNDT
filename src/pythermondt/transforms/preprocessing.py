import torch
import torch.nn as nn
from ..data import DataContainer
from typing import Callable, TypeVar, cast

C = TypeVar('C', bound=Callable)

def proxy(f: C) -> C:
    return cast(C, lambda self, *args, **kwargs: f(self, *args, **kwargs))

class Compose(nn.Module):
    def __init__(self, transforms: list):
        super(Compose, self).__init__()
        self.transforms = transforms

    def forward(self, container: DataContainer) -> DataContainer:
        for t in self.transforms:
            container = t(container)
        return container

    __call__: Callable[..., DataContainer] = proxy(forward)

class ApplyLUT(nn.Module):
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
        tdata = lut[tdata.int()]

        # Update the container and return it
        container.fill_dataset("Data/Tdata", tdata)
        return container # type: DataContainer
    
    __call__: Callable[..., DataContainer] = proxy(forward)