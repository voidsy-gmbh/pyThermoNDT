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
            raise TypeError("Not all transforms inherit from _BaseTransform.")
        self.transforms = transforms

    def __str__(self) -> str:
        """Custom repr for Compose - no type label, cleaner format."""
        if not self.transforms:
            return "Compose([])"

        # Show transforms in a clean format
        transform_strs = [str(t) for t in self.transforms]
        if len(transform_strs) == 1:
            return f"Compose([{transform_strs[0]}])"

        # Multi-line format for multiple transforms
        transforms_str = ",\n    ".join(transform_strs)
        return f"Compose([\n    {transforms_str}\n])"

    def forward(self, container: DataContainer) -> DataContainer:
        for t in self.transforms:
            container = t(container)
        return container
