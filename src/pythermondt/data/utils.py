import itertools
import math
from collections.abc import Sequence

import torch
from torch import Generator, default_generator

from ..transforms import ThermoTransform
from .thermo_dataset import IndexedThermoDataset, ThermoDataset


def random_split(dataset:ThermoDataset, lengths: Sequence, transforms: Sequence[ThermoTransform | None] | None = None, generator: Generator = default_generator) -> list[IndexedThermoDataset]:
    """ Split a dataset into random non-overlapping subsets of given lengths with optional transforms beeing applied.

    If a list of fractions that sum up to 1 is given, the lengths will be computed automatically as floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be distributed in round-robin fashion to the lengths until there are no remainders left.

    Parameters:
        dataset (Dataset): Dataset to be split
        lengths (Sequence[float]): Fractions for each split that sum up to 1.0. 
        transforms (Sequence[ThermoTransform], optional): Optional sequence of transforms for each split.
        generator (Generator, optional): Generator used for reproducible splits. Per default the default generator is used.

    Returns:
        List[ThermoSubset]: List of subsets with the specified lengths and transforms
        
    Raises:
        ValueError: If lengths don't sum up to 1.0
        ValueError: If number of transforms doesn't match lengths
    """
    # Validate transforms if provided
    if transforms is not None and len(transforms) != len(lengths):
        raise ValueError(f"Number of transforms: {len(transforms)} must match number of splits: {len(lengths)}")

    # Validate lenghts
    if not all(value >= 0 for value in lengths):
        raise ValueError("All values in lengths must be greater or equal to 0")

    # If lengths are provided as fractions and not as absolute numbers
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        # Compute the lenghts of the subsets
        subset_lengths: list[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(math.floor(len(dataset) * frac))
            subset_lengths.append(n_items_in_split)

        # Compute remainder
        remainder = len(dataset) - sum(subset_lengths)

        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths

    # Print a warning if any of the splits have a length of 0
    for i, length in enumerate(lengths):
        if length == 0:
            print(
                f"Length of split at index {i} is 0. ",
                "This might result in an empty dataset."
            )

    # Raise an error if the computed lengths don't match the length of the original dataset
    if sum(lengths) != len(dataset):
        raise ValueError(f"The sum of the computed subset lengths: {lengths} does not match the length of the original dataset: {len(dataset)}")

    # Generate random indices
    indices = torch.randperm(sum(lengths), generator=generator).tolist()

    # Create the subsets and return
    transforms = transforms if transforms else [None] * len(lengths)
    return [
        IndexedThermoDataset(dataset, indices[offset - length:offset], transform)
        for transform, length, offset in zip(transforms, lengths, itertools.accumulate(lengths), strict=False)
    ]
