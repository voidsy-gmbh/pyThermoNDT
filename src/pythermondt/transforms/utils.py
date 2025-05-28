from abc import ABC, abstractmethod

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
        """Abstract method that must be implemented in the sub-class.

        This method should contain the actual transformation logic.
        """
        raise NotImplementedError("Forward method must be implemented in the sub-class.")

    # Add type annotations to __call__ method, so that the type checker can infer the correct return type.
    # Otherwise, the return type will be inferred as 'Any'.
    # __call__ does not have to be overridden, this is already implemented in the nn.Module class from PyTorch.
    # See: https://github.com/microsoft/pyright/issues/3249
    def __call__(self, container: DataContainer) -> DataContainer:
        """Override to provide proper type hints."""
        return super().__call__(container)


class Compose(ThermoTransform):
    """Compose a sequence of transforms together into a single transform.

    This transform sequentially applies a list of transforms to the input container.
    """

    def __init__(self, transforms: list[ThermoTransform]):
        """Compose a sequence of transforms together into a single transform.

        This transform sequentially applies a list of transforms to the input container.
        """
        super().__init__()

        # Check if all the provided transforms are valid (Thermotransforms are already callable)
        if not all(isinstance(t, ThermoTransform) for t in transforms):
            raise TypeError("Not all transforms inherit from ThermoTransform.")
        self.transforms = transforms

    def forward(self, container: DataContainer) -> DataContainer:
        for t in self.transforms:
            container = t(container)
        return container
