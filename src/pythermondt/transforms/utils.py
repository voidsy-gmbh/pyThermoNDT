from collections.abc import Sequence

from ..data import DataContainer
from .base import ThermoTransform, _BaseTransform


class Compose(ThermoTransform):
    """Compose a sequence of transforms together into a single transform.

    This transform sequentially applies a list of transforms to the input container.
    """

    def __init__(self, transforms: Sequence[_BaseTransform]):
        """Compose a sequence of transforms together into a single transform.

        This transform sequentially applies a list of transforms to the input container.
        """
        super().__init__()

        # Check if all the provided transforms are valid (Thermotransforms are already callable)
        if not all(isinstance(t, _BaseTransform) for t in transforms):
            raise TypeError("Not all transforms inherit from ThermoTransform.")
        self.transforms = transforms

    def forward(self, container: DataContainer) -> DataContainer:
        for t in self.transforms:
            container = t(container)
        return container
