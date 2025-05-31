from abc import ABC, abstractmethod

from torch import nn

from ..data import DataContainer


class ThermoTransform(nn.Module, ABC):
    """Abstract base class that all deterministic transforms of PyThermoNDT must inherit from.

    Initializes the module and sets up necessary configurations. These transforms are expected to produce deterministic
    outputs given the same input, thus they can safely be cached for improved performance.
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
