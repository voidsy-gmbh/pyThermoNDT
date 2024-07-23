import torch.nn as nn
from typing import Callable, List
from abc import ABC, abstractmethod
from ..data import DataContainer

class ThermoTransform(nn.Module, ABC):
    def __init__(self):
        super(ThermoTransform, self).__init__()

    @abstractmethod
    def forward(self, container: DataContainer) -> DataContainer:
        raise NotImplementedError("Forward method must be implemented in the sub-class.")

    __call__: Callable[..., DataContainer]

class Compose(ThermoTransform):
    def __init__(self, transforms: List[ThermoTransform]):
        super(Compose, self).__init__()
        self.transforms = transforms
    
    def forward(self, container: DataContainer) -> DataContainer:
        for t in self.transforms:
            container = t(container)
        return container