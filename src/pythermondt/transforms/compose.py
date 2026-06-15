from collections.abc import Sequence

import torch

from ..data import DataContainer
from .base import RandomThermoTransform, ThermoTransform, _BaseTransform


def _indent_str(s: str, indent: str = "    ") -> str:
    """Indent all lines after the first of a multi-line string."""
    if "\n" not in s:
        return s
    lines = s.split("\n")
    return "\n".join([lines[0]] + [indent + line for line in lines[1:]])


class Compose(ThermoTransform):
    # fmt: off
    """Compose a sequence of transforms together into a single transform by applying them sequentially.

    Each transform is applied in the order provided to the constructor.

    Example:
        >>> train_pipeline = T.Compose([
        ...     T.ApplyLUT(),
        ...     T.SubtractFrame(0),
        ...     T.RemoveFlash(method='excitation_signal'),
        ...     T.NonUniformSampling(64),
        ...     T.RandomFlip(p_height=0.3, p_width=0.3),
        ...     T.GaussianNoise(std=25e-3),
        ...     T.MinMaxNormalize(),
        ... ])
    """
    # fmt: on
    def __init__(self, transforms: Sequence[_BaseTransform]):
        """Initialize the Compose transform.

        Args:
            transforms (Sequence[_BaseTransform]): List of transforms to apply sequentially.

        Raises:
            TypeError: If not all transforms inherit from _BaseTransform.
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

        # Show transforms in a clean format, indent nested containers
        transform_strs = [_indent_str(str(t)) for t in self.transforms]
        if len(transform_strs) == 1:
            return f"Compose([{transform_strs[0]}])"

        # Multi-line format for multiple transforms
        transforms_str = ",\n    ".join(transform_strs)
        return f"Compose([\n    {transforms_str}\n])"

    def forward(self, container: DataContainer) -> DataContainer:
        for t in self.transforms:
            container = t(container)
        return container


class RandomCompose(RandomThermoTransform):
    # fmt: off
    """Apply each transform independently with probability p.

    Unlike Compose (which applies all transforms sequentially),
    RandomCompose applies each transform with probability p.

    Example:
        >>> # Each augmentation applied independently with 30% chance
        >>> augmentations = T.RandomCompose([
        ...     T.AdaptiveGaussianNoise(),
        ...     T.RandomFlip(),
        ... ], p=0.3)

        >>> # Different probabilities per transform
        >>> augmentations = T.RandomCompose([
        ...     T.AdaptiveGaussianNoise(),
        ...     T.RandomFlip(),
        ... ], p=[0.5, 0.2])
    """
    # fmt: on
    def __init__(self, transforms: Sequence[_BaseTransform], p: float | Sequence[float] = 0.5):
        """Initialize the RandomCompose transform.

        Args:
            transforms (Sequence[_BaseTransform]): List of transforms to apply randomly.
            p (float | Sequence[float]): Probability for each transform. Can either be a scalar (same for all) or
                sequence of probabilities (one per transform). Default: 0.5

        Raises:
            TypeError: If not all transforms inherit from _BaseTransform.
            ValueError: If p is a sequence and its length doesn't match transforms.
        """
        super().__init__()

        if not all(isinstance(t, _BaseTransform) for t in transforms):
            raise TypeError("Not all transforms inherit from _BaseTransform.")
        self.transforms = transforms

        # Handle scalar or list of probabilities
        if isinstance(p, (int, float)):
            self.p = [float(p)] * len(transforms)
        else:
            if len(p) != len(transforms):
                raise ValueError(f"Length of p ({len(p)}) must match transforms ({len(transforms)}).")
            self.p = [float(pi) for pi in p]

        # Validate range
        if not all(0 <= prob <= 1 for prob in self.p):
            raise ValueError(f"All probabilities must be in [0, 1], got {self.p}.")

    def __str__(self) -> str:
        """Custom repr for RandomCompose - no type label, cleaner format."""
        if not self.transforms:
            return "RandomCompose([])"

        # Show transforms in a clean format with probabilities, indent nested containers
        transform_strs = [_indent_str(f"{t} (p={p:.2f})") for t, p in zip(self.transforms, self.p, strict=True)]
        if len(transform_strs) == 1:
            return f"RandomCompose([{transform_strs[0]}])"

        # Multi-line format for multiple transforms
        transforms_str = ",\n    ".join(transform_strs)
        return f"RandomCompose([\n    {transforms_str}\n])"

    def forward(self, container: DataContainer) -> DataContainer:
        for transform, prob in zip(self.transforms, self.p, strict=True):
            if torch.rand(1).item() < prob:
                container = transform(container)
        return container
