import itertools
import math
from collections.abc import Callable, Sequence
from functools import partial

import torch
from torch import Generator, default_generator

from ..data import DataContainer
from ..transforms.base import _BaseTransform
from .indexed_thermo_dataset import IndexedThermoDataset
from .thermo_dataset import ThermoDataset


def random_split(
    dataset: ThermoDataset,
    lengths: Sequence,
    transforms: Sequence[_BaseTransform | None] | None = None,
    generator: Generator = default_generator,
) -> list[IndexedThermoDataset]:
    """Split a dataset into random non-overlapping subsets of given lengths with optional transforms being applied.

    If a list of fractions that sum up to 1 is given, the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be distributed in round-robin fashion to the
    lengths until there are no remainders left.

    Args:
        dataset (Dataset): Dataset to be split
        lengths (Sequence[float]): Fractions for each split that sum up to 1.0.
        transforms (Sequence[_BaseTransform | None], optional): Optional sequence of transforms for each split.
        generator (Generator, optional): Generator used for reproducible splits.
            Per default the default generator is used.

    Returns:
        List[ThermoSubset]: List of subsets with the specified lengths and transforms

    Raises:
        ValueError: If lengths don't sum up to 1.0
        ValueError: If number of transforms doesn't match lengths

    Example:
        >>> from pythermondt import transforms as T
        >>> # Create train/val/test splits with different transforms
        >>> train_transform = T.Compose([T.ApplyLUT(), T.RandomCrop(96, 96)])
        >>> val_transform = T.Compose([T.ApplyLUT(), T.CropFrames(96, 96)])
        >>> train, val, test = random_split(dataset, [0.7, 0.2, 0.1], transforms=[train_transform, val_transform, None])
    """
    # Validate transforms if provided
    if transforms is not None and len(transforms) != len(lengths):
        raise ValueError(f"Number of transforms: {len(transforms)} must match number of splits: {len(lengths)}")

    # Validate lengths
    if not all(value >= 0 for value in lengths):
        raise ValueError("All values in lengths must be greater or equal to 0")

    # If lengths are provided as fractions and not as absolute numbers
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        # Compute the lengths of the subsets
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
            print(f"Length of split at index {i} is 0. ", "This might result in an empty dataset.")

    # Raise an error if the computed lengths don't match the length of the original dataset
    if sum(lengths) != len(dataset):
        raise ValueError(
            f"The sum of the computed subset lengths: {lengths} does not match "
            f"the length of the original dataset: {len(dataset)}"
        )

    # Generate random indices
    indices = torch.randperm(sum(lengths), generator=generator).tolist()

    # Create the subsets and return
    transforms = transforms if transforms else [None] * len(lengths)
    return [
        IndexedThermoDataset(dataset, indices[offset - length : offset], transform)
        for transform, length, offset in zip(transforms, lengths, itertools.accumulate(lengths), strict=False)
    ]


def container_collate(*paths: str) -> Callable[[Sequence[DataContainer]], tuple[torch.Tensor, ...]]:
    """Factory function for creating a collate function for DataContainer objects.

    Returns a function that extracts specified dataset paths and stacks them along batch dimension.

    Args:
        *paths (str): Variable number of dataset paths to extract (e.g. '/Data/Tdata', '/GroundTruth/DefectMask')

    Returns:
        Callable[[Sequence[DataContainer]], tuple[torch.Tensor, ...]]: A collate function that takes a batch of
            DataContainer objects and returns a tuple of tensors. The number of tensors in the tuple corresponds to the
            number of dataset paths provided.

    Raises:
        KeyError: If a specified dataset path doesn't exist in any container
        RuntimeError: If tensors have incompatible shapes for stacking
        ValueError: If empty batch is provided

    Example:
        >>> from torch.utils.data import DataLoader
        >>> collate_fn = container_collate("/Data/Tdata", "/GroundTruth/DefectMask")
        >>> dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    """
    if not paths:
        raise ValueError("At least one path must be specified")

    return partial(_container_collate_impl, paths=paths)


def _container_collate_impl(batch: Sequence[DataContainer], paths: tuple[str, ...]) -> tuple[torch.Tensor, ...]:
    """Implementation function that processes a batch of DataContainer objects for collation.

    Args:
        batch (Sequence[DataContainer]): The batch of DataContainer objects to collate.
        paths (tuple[str, ...]): The dataset paths to extract and collate from each container.

    Returns:
        tuple[torch.Tensor, ...]: A tuple of tensors, each stacked along the batch dimension for the corresponding path.

    Raises:
        KeyError: If a specified dataset path doesn't exist in any container.
        RuntimeError: If tensors have incompatible shapes for stacking.
        ValueError: If empty batch is provided.
    """
    if not batch:
        raise ValueError("Empty batch provided - cannot collate empty sequence")

    # Use get_datasets method to efficiently extract all datasets from each container
    all_tensors = []
    for container in batch:
        try:
            # Get all datasets from this container in one call
            tensors = container.get_datasets(*paths)
            all_tensors.append(tensors)
        except KeyError as exc:
            raise KeyError(f"One or more dataset paths not found in container: {exc}") from exc

    # Stack tensors along batch dimension for each path
    result = []
    for i, path in enumerate(paths):
        try:
            # Extract tensors for this path from all containers and stack them
            result.append(torch.stack([tensors[i] for tensors in all_tensors], dim=0))
        except RuntimeError as e:
            raise RuntimeError(f"Cannot stack tensors for path '{path}': {e}") from e

    return tuple(result)
