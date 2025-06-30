from collections.abc import Iterator, Sequence

from ..data import DataContainer
from ..transforms import ThermoTransform
from .base import BaseDataset


class IndexedThermoDataset(BaseDataset):
    """Extension of ThermoDataset that supports indexing with optional additional transforms.

    The IndexedThermoDataset maintains a subset of the parent dataset and allows for an additional transform to be
    applied to the data. This can be useful when a subset of the data needs to be selected and a different transform
    needs to be applied to the subset, e.g. for random splits of train, validation and test data. The
    IndexedThermoDataset maintains the transform chain of the parent dataset and appends the additional transform to it.
    """

    def __init__(self, dataset: BaseDataset, indices: Sequence[int], transform: ThermoTransform | None = None):
        """Initialize an indexed dataset with optional additional transform.

        Parameters:
            dataset (ThermoDataset): Parent dataset to index into
            indices (Sequence[int]): Sequence of indices to select from parent
            transform (ThermoTransform, optional): Optional transform to apply after parent's transform

        Raises:
            IndexError: If any of the provided indices are out of range
        """
        # Validate the indices
        if not all(0 <= i < len(dataset) for i in indices):
            raise IndexError(f"Provided indices are out of range. Must be within [0, {len(dataset) - 1}]")

        # Initialize base with dataset as parent and given transform
        super().__init__(parent=dataset, transform=transform)

        # Store parent dataset and indices
        self.__parent_dataset = dataset
        self.__indices = indices  # Indices for subset

    def __len__(self) -> int:
        """Return length of indexed dataset."""
        return len(self.__indices)

    def __iter__(self) -> Iterator[DataContainer]:
        """Return iterator over indexed subset."""
        return (self[i] for i in range(len(self.__indices)))

    def __getitem__(self, idx: int) -> DataContainer:
        """Get an item with proper transform chain.

        Args:
            idx (int): Index into the subset

        Returns:
            DataContainer: Transformed data container
        """
        # Load raw data
        data = self._load_raw_data(idx)

        # Apply additional transform if specified
        if self.transform:
            data = self.transform(data)

        return data

    def _load_raw_data(self, idx: int) -> DataContainer:
        """Load raw data from the parent dataset.

        Args:
            idx (int): Index into the subset

        Returns:
            DataContainer: Raw data container from the parent dataset
        """
        # Validate index
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        # Load raw data using parent's logic
        return self.__parent_dataset._load_raw_data(self.__indices[idx])

    @property
    def files(self) -> list[str]:
        """Return list of files corresponding to indexed subset."""
        return [self.__parent_dataset.files[i] for i in self.__indices]
