import collections
import copy
import gc
import sys
from abc import ABC, abstractmethod
from multiprocessing import Manager
from multiprocessing.managers import ListProxy, SyncManager
from multiprocessing.pool import ThreadPool
from typing import Literal

from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..config import settings
from ..data import DataContainer
from ..data.datacontainer.utils import format_bytes
from ..transforms.utils import Compose, _BaseTransform, _flatten_transforms, split_transforms_for_caching

CacheMode = Literal["immediate", "lazy"]


class BaseDataset(Dataset, ABC):
    """Base class for all ThermoDatasets."""

    def __init__(self, parent: "BaseDataset | None" = None, transform: _BaseTransform | None = None):
        self.__parent = parent
        self.__transform = transform

        # Internal state for cache
        self.__manager: SyncManager | None = None
        self.__cache_built = False
        self.__cache: list | ListProxy = []
        self.__det_transforms: _BaseTransform | None = None
        self.__runtime_transforms: _BaseTransform | None = None

    @abstractmethod
    def load_raw_data(self, idx: int) -> DataContainer:
        """Load raw data at index - implemented by concrete classes."""

    @abstractmethod
    def __len__(self) -> int:
        """Return length of dataset - implemented by concrete classes."""

    @property
    @abstractmethod
    def files(self) -> list[str]:
        """Get the list of files associated with this dataset."""

    def __del__(self):
        """Ensure that the cache is released when the dataset is deleted."""
        if self.cache_built:
            self.release_cache(gc_collect=False)

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
            if self.__cache[idx] is None:
                # Load the item if it was not cached
                self.__cache[idx] = self._load_cache_item(idx)
            data = copy.deepcopy(self.__cache[idx])
            return self.__runtime_transforms(data) if self.__runtime_transforms else data

        # Get the data
        data = self.load_raw_data(idx) if self.parent is None else self.parent[self._map_index(idx)]

        # Apply additional transform if specified
        if self.transform:
            data = self.transform(data)  # pylint: disable=not-callable

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
        container_size = sum(c.memory_bytes() for c in self.__cache if isinstance(c, DataContainer))
        return container_size + sys.getsizeof(self) + sys.getsizeof(self.__cache)

    def print_memory_usage(self):
        """Print the memory usage of this dataset."""
        print(f"{self.__class__.__name__} Overview:")
        print("-" * len(f"{self.__class__.__name__} Overview:"))
        print(f"Currently there are {sum(1 for item in self.__cache if item)} items in the cache")
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

    def build_cache(self, mode: CacheMode = "lazy", num_workers: int | None = None):
        # fmt: off
        """Build an in-memory cache of preprocessed data for faster training.

        Automatically splits the transform pipeline at the first random transform:
        - Deterministic transforms are applied once and cached
        - Random transforms + subsequent operations run at runtime

        Platform Considerations:
            **Windows**: Use lazy mode to avoid memory issues due to cache duplication in forked processes
            **Linux**: Both modes work efficiently - choose based on workflow preference

        Args:
            mode (CacheMode): Cache building strategy
                - "lazy": Create shared cache, workers fill on-demand for faster startup (default)
                - "immediate": Build all items upfront using a ThreadPool
            num_workers (int, optional): Number of workers used for cache building. This setting only applies if `mode`
                is "immediate". If num_workers is None, the global configuration of PyThermoNDT will be used.
                If less than 1, it defaults to 1 worker. Default is None.

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

            >>> # Development: fast startup (default)
            >>> dataset.build_cache(mode="lazy")
            >>> loader = DataLoader(dataset, num_workers=4, persistent_workers=True)

            >>> # Production: parallel cache building (Only recommended on linux)
            >>> dataset.build_cache(mode="immediate", num_workers=8)
            >>> loader = DataLoader(dataset, num_workers=8, persistent_workers=True)
        """
        # fmt: on
        # Skip if cache already built
        if self.__cache_built:
            return

        # Get the complete transform chain and split it into deterministic and runtime transforms
        self.__det_transforms, self.__runtime_transforms = split_transforms_for_caching(self.get_transform_chain())

        # Initialize the cache based on the mode
        if mode == "immediate":
            unit = "files"
            desc = f"{self.__class__.__name__} - Building cache"
            num = len(self)
            workers = max(num_workers, 1) if num_workers is not None else settings.num_workers
            worker_fn = self._load_cache_item
            if workers > 1:
                # Use ThreadPool for immediate cache building in parallel
                with ThreadPool(processes=workers) as pool:
                    self.__cache = list(tqdm(pool.imap(worker_fn, range(num)), desc=desc, unit=unit, total=num))
            else:
                self.__cache = [self._load_cache_item(i) for i in tqdm(range(num), desc=desc, unit=unit)]
        elif mode == "lazy":
            # Store the manager so it can be released in case
            self.__manager = Manager()
            # Create a shared list for lazy loading using a list proxy
            self.__cache = self.__manager.list([None] * len(self))
        else:
            raise ValueError(f"Invalid cache mode: {mode}. Use one of: {list(CacheMode.__args__)}.")

        self.__cache_built = True

    def release_cache(self, gc_collect: bool = True):
        """Release the in-memory cache to free up memory and release any background manager processes.

        Args:
            gc_collect (bool): Whether to run garbage collection after releasing the cache. Default is True.
        """
        # For regular lists, clear the items to free memory ==> ListProxies are handled by manager shutdown
        if not isinstance(self.__cache, ListProxy):
            self.__cache = []
        self.__det_transforms = None
        self.__runtime_transforms = None
        self.__cache_built = False

        # Ensure that the manager process is terminated
        if self.__manager:
            try:
                self.__manager.shutdown()
            except Exception:
                pass
            finally:
                self.__manager = None

        # Garbage collect to free memory if requested
        if gc_collect:
            gc.collect()
