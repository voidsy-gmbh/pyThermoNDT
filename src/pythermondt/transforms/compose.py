from collections.abc import Sequence

import torch

from ..data import DataContainer
from .base import RandomThermoTransform, ThermoTransform, _BaseTransform


class Compose(ThermoTransform):
    """Compose a sequence of transforms together into a single transform by applying them sequentially.

    Each transform is applied in the order provided to the constructor.

    Example:
        >>> train_pipeline = T.Compose(
        ...     [
        ...         T.ApplyLUT(),
        ...         T.RemoveFlash(),
        ...         T.SubtractFrame(0),
        ...         T.MinMaxNormalize(),
        ...     ]
        ... )
    """

    def __init__(self, transforms: Sequence[_BaseTransform]):
        """Compose a sequence of transforms together into a single transform by applying them sequentially.

        Args:
            transforms (Sequence[_BaseTransform]): List of transforms to apply sequentially.
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


class RandomCompose(RandomThermoTransform):
    """Apply each transform independently with probability p.

    Unlike Compose (which applies all transforms sequentially),
    RandomCompose applies each transform with probability p.

    Example:
        >>> # Each augmentation applied independently with 30% chance
        >>> augmentation_pipeline = T.RandomCompose(
        ...     [
        ...         T.AdaptiveGaussianNoise(),
        ...         T.RandomFlip(),
        ...     ],
        ...     p=0.3,
        ... )

        >>> # Different probabilities per transform
        >>> augmentation_pipeline = T.RandomCompose(
        ...     [
        ...         T.AdaptiveGaussianNoise(),
        ...         T.RandomFlip(),
        ...     ],
        ...     p=0.3,
        ... )
    """

    def __init__(self, transforms: Sequence[_BaseTransform], p: float | Sequence[float] = 0.5):
        """Compose a sequence of transforms together into a single transform, applying each with probability p.

        Args:
            transforms (Sequence[_BaseTransform]): List of transforms to apply randomly.
            p (float | Sequence[float]): Probability for each transform. Can either be a scalar (same for all) or
                sequence of probabilities (one per transform). Default: 0.5
        """
        super().__init__()
        self.transforms = transforms

        # Handle scalar or list of probabilities
        if isinstance(p, (int, float)):
            self.probs = [float(p)] * len(transforms)
        else:
            if len(p) != len(transforms):
                raise ValueError(f"Length of p ({len(p)}) must match transforms ({len(transforms)})")
            self.probs = [float(pi) for pi in p]

    def __str__(self) -> str:
        """Custom repr for RandomCompose - no type label, cleaner format."""
        if not self.transforms:
            return "RandomCompose([])"

        # Show transforms in a clean format with probabilities
        transform_strs = [f"{t} (p={p:.2f})" for t, p in zip(self.transforms, self.probs, strict=True)]
        if len(transform_strs) == 1:
            return f"RandomCompose([{transform_strs[0]}])"

        # Multi-line format for multiple transforms
        transforms_str = ",\n    ".join(transform_strs)
        return f"RandomCompose([\n    {transforms_str}\n])"

    def forward(self, container: DataContainer) -> DataContainer:
        for transform, prob in zip(self.transforms, self.probs, strict=True):
            if torch.rand(1).item() < prob:
                container = transform(container)
        return container
