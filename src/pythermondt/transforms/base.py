import inspect
from abc import ABC, abstractmethod

from torch import nn

from ..data import DataContainer


class _BaseTransform(nn.Module, ABC):
    """Abstract base class for all transforms in PyThermoNDT.

    This class is not intended to be used directly, but rather serves as a foundation for both deterministic and
    random transforms. It initializes the module and sets up necessary configurations.
    """

    @abstractmethod
    def forward(self, container: DataContainer) -> DataContainer:
        """Abstract method that must be implemented in the sub-class.

        This method should contain the actual transformation logic.
        """
        raise NotImplementedError("Forward method must be implemented in the sub-class.")

    @property
    @abstractmethod
    def is_random(self) -> bool:
        """Indicates whether the transform is random or deterministic."""
        raise NotImplementedError("is_random property must be implemented in the sub-class.")

    def extra_repr(self) -> str:
        """Return extra representation of transform parameters.

        This follows PyTorch's pattern but makes it automatic.
        Override this method if you need custom parameter formatting.
        """
        # Get init parameters by inspecting the constructor
        try:
            sig = inspect.signature(self.__class__.__init__)
            init_params = set(sig.parameters.keys()) - {"self"}
        except (ValueError, TypeError):
            init_params = set()

        # Get attributes that match init parameters
        params = []
        for attr_name in sorted(init_params):
            if hasattr(self, attr_name):
                value = getattr(self, attr_name)
                # Only include simple types that are likely parameters
                if isinstance(value, (int, float, str, bool, list, tuple)):
                    params.append(f"{attr_name}={value}")

        return ", ".join(params)

    def __str__(self) -> str:
        """Get a string representation of the transform."""
        class_name = self.__class__.__name__
        transform_type = "Random" if isinstance(self, RandomThermoTransform) else "Deterministic"

        extra_repr = self.extra_repr()
        if extra_repr:
            return f"{class_name}({extra_repr}) [{transform_type}]"
        return f"{class_name}() [{transform_type}]"

    # Add type annotations to __call__ method, so that the type checker can infer the correct return type.
    # Otherwise, the return type will be inferred as 'Any'.
    # __call__ does not have to be overridden, this is already implemented in the nn.Module class from PyTorch.
    # See: https://github.com/microsoft/pyright/issues/3249
    def __call__(self, container: DataContainer) -> DataContainer:
        """Override to provide proper type hints."""
        return super().__call__(container)


class ThermoTransform(_BaseTransform):
    """Abstract base class that all deterministic transforms of PyThermoNDT must inherit from.

    Initializes the module and sets up necessary configurations. These transforms are expected to produce deterministic
    outputs given the same input, thus they can safely be cached for improved performance.
    """

    @property
    def is_random(self) -> bool:
        return False


class RandomThermoTransform(_BaseTransform):
    """Abstract base class that all random/stochastic transforms of PyThermoNDT must inherit from.

    Initializes the module and sets up necessary configurations. These transforms are expected to produce random outputs
    given the same input, thus they should NOT be cached for improved performance.
    """

    @property
    def is_random(self) -> bool:
        return True
