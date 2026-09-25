import itertools
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from operator import methodcaller

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
            # Defensive check to ensure fractions are between 0 and 1
            if frac < 0 or frac > 1:  # pragma: no cover
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = math.floor(len(dataset) * frac)
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
            warnings.warn(f"Length of split at index {i} is 0. This might result in an empty dataset.", stacklevel=2)

    # Raise an error if the computed lengths don't match the length of the original dataset
    if sum(lengths) != len(dataset):
        raise ValueError(
            f"The sum of the computed subset lengths: {lengths} does not match "
            f"the length of the original dataset: {len(dataset)}"
        )

    # Generate random indices
    indices = torch.randperm(sum(lengths), generator=generator).tolist()

    # Create the subsets and return
    transforms = transforms or [None] * len(lengths)
    return [
        IndexedThermoDataset(dataset, indices[offset - length : offset], transform)
        for transform, length, offset in zip(transforms, lengths, itertools.accumulate(lengths), strict=False)
    ]


@dataclass(frozen=True)
class DeriveField:
    """Specification for a derived collate field.

    The callable is evaluated per sample and its return value is stacked across the batch.
    """

    name: str
    fn: Callable[[DataContainer], "torch.Tensor | bool | float"]


def derive(name: str, fn: Callable[[DataContainer], "torch.Tensor | bool | float"]) -> DeriveField:
    """Create a derived field for use with :func:`container_collate`.

    Args:
        name: Field name used in error messages when stacking fails.
        fn: Callable evaluated per sample. Must return a stackable value (tensor, bool, or float).

    Returns:
        A DeriveField specification.

    Example:
        >>> collate_fn = container_collate(
        ...     "/Data/Tdata",
        ...     derive("tdata_cnn", lambda c: c.get_dataset("/Data/Tdata").permute(2, 0, 1)),
        ...     derive("has_defect", lambda c: "/GroundTruth/DefectMask" in c.nodes),
        ... )
    """
    return DeriveField(name=name, fn=fn)


def container_collate(*fields: str | DeriveField) -> Callable[[Sequence[DataContainer]], tuple[torch.Tensor, ...]]:
    """Factory function for creating a collate function for DataContainer objects.

    Returns a function that evaluates each field per sample and stacks the results along the batch dimension.

    Args:
        *fields (str | DeriveField): Dataset paths to extract (e.g. '/Data/Tdata') and/or derived
            field specifications created with :func:`derive`.

    Returns:
        Callable[[Sequence[DataContainer]], tuple[torch.Tensor, ...]]: A collate function that takes a batch of
            DataContainer objects and returns a tuple of tensors. The number of tensors in the tuple corresponds to the
            number of fields provided.

    Raises:
        ValueError: If no fields are provided, or if the collate function receives an empty batch.
        TypeError: If a field is neither str nor DeriveField.
        KeyError: If the collate function receives a container without a requested field path.
        RuntimeError: If field evaluation or stacking fails.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> collate_fn = container_collate("/Data/Tdata", "/GroundTruth/DefectMask")
        >>> dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    """
    if not fields:
        raise ValueError("At least one path must be specified")

    return partial(_container_collate_impl, specs=tuple(_normalize_field(f) for f in fields))


def _normalize_field(field: str | DeriveField) -> DeriveField:
    """Normalize a field specification to a DeriveField object."""
    if isinstance(field, DeriveField):
        return field
    elif isinstance(field, str):
        return DeriveField(name=field, fn=methodcaller("get_dataset", field))
    raise TypeError(f"Invalid field type: {type(field)}. Must be str or DeriveField.")


def _container_collate_impl(batch: Sequence[DataContainer], specs: tuple[DeriveField, ...]) -> tuple[torch.Tensor, ...]:
    """Implementation function that processes a batch of DataContainer objects for collation.

    Args:
        batch (Sequence[DataContainer]): The batch of DataContainer objects to collate.
        specs (tuple[DeriveField, ...]): Field specifications to extract and collate from each container.

    Returns:
        tuple[torch.Tensor, ...]: Tensors stacked along the batch dimension for each field.

    Raises:
        KeyError: If a field path does not exist in a container.
        RuntimeError: If field evaluation or stacking fails.
        ValueError: If empty batch is provided.
    """
    if not batch:
        raise ValueError("Empty batch provided - cannot collate empty sequence")

    # Evaluate each field per sample
    all_values = []
    for container in batch:
        values = []
        for spec in specs:
            try:
                values.append(spec.fn(container))
            except KeyError as exc:
                detail = exc.args[0] if exc.args else exc
                raise KeyError(f"Field '{spec.name}': {detail}") from exc
            except Exception as exc:
                raise RuntimeError(f"Error evaluating field '{spec.name}': {exc}") from exc
        all_values.append(tuple(values))

    # Stack values along batch dimension for each field
    result = []
    for i, spec in enumerate(specs):
        try:
            result.append(torch.stack([torch.as_tensor(values[i]) for values in all_values], dim=0))
        except Exception as exc:
            raise RuntimeError(f"Cannot stack tensors for field '{spec.name}': {exc}") from exc

    return tuple(result)
