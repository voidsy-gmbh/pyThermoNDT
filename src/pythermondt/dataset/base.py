import collections
import copy
from abc import ABC, abstractmethod

from torch.utils.data import Dataset

from ..data import DataContainer
from ..transforms.utils import Compose, _BaseTransform, _flatten_transforms, split_transforms_for_caching


class BaseDataset(Dataset, ABC):
    """Base class for all ThermoDatasets."""

    def __init__(self, parent: "BaseDataset | None" = None, transform: _BaseTransform | None = None):
        self.__parent = parent
        self.__transform = transform

        # Internal state for cache
        self.__cache_built = False
        self.__cache = []
        self.__det_transforms = None
        self.__runtime_transforms = None

    @abstractmethod
    def load_raw_data(self, idx: int) -> DataContainer:
        """Load raw data at index - implemented by concrete classes."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return length of dataset - implemented by concrete classes."""
        pass

    @property
    @abstractmethod
    def files(self) -> list[str]:
        """Get the list of files associated with this dataset."""
        pass

    def __getitem__(self, idx: int) -> DataContainer:
        """Get an item while also applying the proper transform chain.

        Args:
            idx (int): Index of the data to retrieve

        Returns:
            DataContainer: Transformed data container
        """
        # Validate index
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        if self.cache_built:
            cpy = copy.deepcopy(self.__cache[idx])
            return self.__runtime_transforms(cpy) if self.__runtime_transforms else cpy

        # Get the data
        data = self.load_raw_data(idx) if self.parent is None else self.parent[self._map_index(idx)]

        # Apply additional transform if specified
        if self.transform:
            data = self.transform(data)

        return data

    @property
    def parent(self) -> "BaseDataset | None":
        """Get parent dataset if available."""
        return self.__parent

    @property
    def transform(self) -> _BaseTransform | None:
        """Get the transform for this dataset."""
        return self.__transform

    @property
    def cache_built(self) -> bool:
        """Check if the cache has been built."""
        return self.__cache_built

    def _map_index(self, idx: int) -> int:
        """Hook to map the index to the parent's index. Override in subclasses if needed."""
        return idx

    def get_transform_chain(self) -> _BaseTransform:
        """Walk up graph to build the complete sequence transforms for this dataset and compose it in a single one."""
        transforms: collections.deque[_BaseTransform] = collections.deque()
        current: BaseDataset | None = self

        while current is not None:
            if current.transform:
                # Flatten the transforms
                flattened = _flatten_transforms(
                    current.transform.transforms if isinstance(current.transform, Compose) else [current.transform]
                )
                transforms.extendleft(reversed(flattened))
            current = current.parent

        return Compose(list(transforms))

    def build_cache(self):
        """Build the cache for this dataset."""
        self.__det_transforms, self.__runtime_transforms = split_transforms_for_caching(self.get_transform_chain())
        self.__cache = [self.__det_transforms(self.load_raw_data(i)) for i in range(len(self))]
        self.__cache_built = True
