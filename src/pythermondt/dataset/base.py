import collections
import copy
import sys
from abc import ABC, abstractmethod

from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..data import DataContainer
from ..data.datacontainer.utils import format_bytes
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

    def _load_cache_item(self, idx: int) -> DataContainer:
        """Load a single item and apply deterministic transforms."""
        return self.__det_transforms(self.load_raw_data(idx)) if self.__det_transforms else self.load_raw_data(idx)

    def memory_bytes(self) -> int:
        """Calculate the memory usage of this dataset.

        **Note:** If the cache has not been built, this will be small because the data is not loaded yet.

        Returns:
            int: Memory usage in bytes
        """
        return sum(c.memory_bytes() for c in self.__cache) + sys.getsizeof(self) + sys.getsizeof(self.__cache)

    def print_memory_usage(self):
        """Print the memory usage of this dataset."""
        print(f"{self.__class__.__name__} Overview:")
        print("-" * len(f"{self.__class__.__name__} Overview:"))
        print(f"Currently there are {len(self.__cache)} items in the cache")
        print(f"Total memory usage of the cache: {format_bytes(self.memory_bytes())}")
        print("\n")

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
        # fmt: off
        """Build an in-memory cache of preprocessed data for faster training.

        Automatically splits the transform pipeline at the first random transform:
        - Deterministic transforms are applied once and cached
        - Random transforms + subsequent operations run at runtime

        Example with a common preprocessing pipeline:
            >>> train_pipeline = T.Compose([
            ...     T.ApplyLUT(),                           # Cached
            ...     T.SubtractFrame(0),                     # Cached
            ...     T.RemoveFlash(method='excitation_signal'), # Cached
            ...     T.NonUniformSampling(64),               # Cached
            ...     T.RandomFlip(p_height=0.3, p_width=0.3), # Runtime (random)
            ...     T.GaussianNoise(std=25e-3),             # Runtime (random)
            ...     T.MinMaxNormalize(),                    # Runtime (after random)
            ... ])
            >>> dataset.build_cache()  # Caches: ApplyLUT → SubtractFrame → RemoveFlash → NonUniformSampling

        Benefits:
            - 3-5x faster data loading during training
            - Preserves randomness for data augmentation
            - Reduces repeated computation of expensive operations
        """
        # fmt: on
        self.__det_transforms, self.__runtime_transforms = split_transforms_for_caching(self.get_transform_chain())
        self.__cache = [self._load_cache_item(i) for i in tqdm(range(len(self)), desc="Building cache", unit="files")]
        self.__cache_built = True
