from collections.abc import Callable, Sequence

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


class CallbackTransform(_BaseTransform):
    """A transform that constructs another transform using a callback function at runtime.

    This is useful for setting up custom transforms, where the input arguments depend on runtime information.

    Per default, randomness is inferred from the callback by executing it with a dummy container. If this fails, the
    transform is assumed to be non-random. To override this behavior, set the `is_random` argument explicitly.
    """

    def __init__(self, callback: Callable[[DataContainer], _BaseTransform], is_random: bool | None = None):
        """A transform that constructs another transform using a callback function at runtime.

        This is useful for setting up custom transforms, where the input arguments depend on runtime information.

        Per default, randomness is inferred from the callback by executing it with a dummy container. If this fails, the
        transform is assumed to be non-random. To override this behavior, set the `is_random` argument explicitly.

        Args:
            callback (Callable[[DataContainer], _BaseTransform]): A function that takes a DataContainer and returns a
                transform instance. This function will be called each time the transform is applied.
            is_random (bool | None, optional): Whether the transform is random. If None, randomness is inferred from
                the callback function, with fall to deterministic if inference fails. Defaults to None.
        """
        super().__init__()
        self.callback = callback
        self._is_random = is_random

    @property
    def is_random(self) -> bool:
        if self._is_random is not None:
            return self._is_random

        # Try to infer randomness from callback
        try:
            dummy_container = DataContainer()
            transform = self.callback(dummy_container)
            return transform.is_random
        except Exception:
            print("Warning: Could not infer randomness from callback function. Assuming non-random transform.")
            return False

    def forward(self, container: DataContainer) -> DataContainer:
        # Instantiate the transform using the callback and apply it
        transform = self.callback(container)
        return transform(container)


def split_transforms_for_caching(
    transforms: _BaseTransform | Sequence[_BaseTransform],
) -> tuple[Compose, Compose]:
    """Split any composed transforms into deterministic and random transforms.

    This function takes a sequence of transforms or a Compose object and splits them into two Compose objects:
    - Deterministic transforms
    - Random transforms and any transforms that follow them

    Nested Compose transforms are flattened before splitting, so that the
    split is applied to the individual transforms rather than the Compose container itself.

    Args:
        transforms (_BaseTransform | Sequence[_BaseTransform]): Transforms to split.

    Returns:
        tuple[Compose, Compose]:
            A tuple containing two Compose objects:
            - The first contains deterministic transforms.
            - The second contains random transforms and any transforms that follow them.
    """
    # Flatten nested Compose transforms first
    if isinstance(transforms, Compose):
        transforms = transforms.transforms
    elif isinstance(transforms, _BaseTransform):
        transforms = [transforms]
    flat_transforms = _flatten_transforms(transforms)

    # Simple split on flattened list
    deterministic = []
    random_and_after = []
    found_random = False

    for transform in flat_transforms:
        if transform.is_random:
            found_random = True

        if found_random:
            random_and_after.append(transform)
        else:
            deterministic.append(transform)

    return Compose(deterministic), Compose(random_and_after)


def _flatten_transforms(transforms: Sequence[_BaseTransform]) -> Sequence[_BaseTransform]:
    """Flatten nested Compose transforms into a single list."""
    flattened: list[_BaseTransform] = []
    for transform in transforms:
        if isinstance(transform, Compose):
            # Recursively flatten nested Compose
            flattened.extend(_flatten_transforms(transform.transforms))
        else:
            flattened.append(transform)
    return flattened
