from abc import ABC, abstractmethod
from collections.abc import Callable

import torch.nn as nn

from ..data import DataContainer


class ThermoTransform(nn.Module, ABC):
    """Abstract base class that all transforms of PyThermoNDT must inherit from.
    Initializes the module and sets up necessary configurations.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, container: DataContainer) -> DataContainer:
        """Abstract method that must be implemented in the sub-class. This method should contain the transformation logic.
        """
        raise NotImplementedError("Forward method must be implemented in the sub-class.")

    # Add type annotations to __call__ method, so that the type checker can infer the correct return type.
    # Otherwise, the return type will be inferred as 'Any'.
    # __call__ does not have to be overriden, this is already implemented in the nn.Module class from PyTorch.
    # See: https://github.com/microsoft/pyright/issues/3249
    __call__: Callable[[DataContainer], DataContainer]


class Compose(ThermoTransform):
    """Composes several transforms together. This transform sequentially applies a list of transforms
    to the input container.
    """

    def __init__(self, transforms: list[ThermoTransform]):
        """Composes several transforms together. This transform sequentially applies a list of transforms
        to the input container.
        """
        super().__init__()

        # Check if all the provided transforms are valid
        if not all(isinstance(t, ThermoTransform) for t in transforms):
            raise TypeError("Not all transforms inherit from ThermoTransform.")
        if not all(isinstance(t, Callable) for t in transforms):
            raise TypeError("Not all transforms are callable.")
        self.transforms = transforms

    def forward(self, container: DataContainer) -> DataContainer:
        for t in self.transforms:
            container = t(container)
        return container
