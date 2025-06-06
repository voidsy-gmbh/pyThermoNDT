from collections.abc import Sequence

from ..data import DataContainer
from .base import RandomThermoTransform, ThermoTransform, _BaseTransform


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

    def forward(self, container: DataContainer) -> DataContainer:
        for t in self.transforms:
            container = t(container)
        return container


def split_transforms_for_caching(
    transforms: Sequence[_BaseTransform],
) -> tuple[Sequence[_BaseTransform], Sequence[_BaseTransform]]:
    """Split any composed transforms into deterministic and random transforms.

    This function takes a sequence of transforms and splits them into two lists:
    - Deterministic transforms
    - Random transforms and any transforms that follow them

    Nested Compose transforms are flattened before splitting, so that the
    split is applied to the individual transforms rather than the Compose container itself.

    Parameters:
        transforms (Sequence[_BaseTransform]): A sequence of transforms to split.

    Returns:
        tuple[Sequence[_BaseTransform], Sequence[_BaseTransform]]:
            A tuple containing two lists:
            - The first list contains deterministic transforms.
            - The second list contains random transforms and any transforms that follow them.
    """
    # Flatten nested Compose transforms first
    flat_transforms = _flatten_transforms(transforms)

    # Simple split on flattened list
    deterministic = []
    random_and_after = []
    found_random = False

    for transform in flat_transforms:
        if isinstance(transform, RandomThermoTransform):
            found_random = True

        if found_random:
            random_and_after.append(transform)
        else:
            deterministic.append(transform)

    return deterministic, random_and_after


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
