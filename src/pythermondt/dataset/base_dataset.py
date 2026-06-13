import collections
import copy
import gc
import multiprocessing as mp
import sys
from abc import ABC, abstractmethod
from multiprocessing.managers import ListProxy, SyncManager
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Literal

from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..config import settings
from ..data import DataContainer
from ..data.datacontainer.utils import format_bytes
from ..transforms.base import _BaseTransform
from ..transforms.compose import Compose
from ..transforms.utils import _flatten_transforms, split_transforms_for_caching

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
        self.__cache_dir: Path | None = None
        self.__cache_storage: Literal["memory", "disk"] = "memory"
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

    def __getstate__(self):
        """Exclude the lazy-cache Manager so the dataset can be pickled for DataLoader workers."""
        state = self.__dict__.copy()
        # ListProxy is picklable; SyncManager is not (parent keeps ownership / shutdown).
        state["_BaseDataset__manager"] = None
        return state

    def __setstate__(self, state: dict):
        """Restore object from pickled state."""
        vars(self).update(state)

    def __getitem__(self, idx: int) -> DataContainer:
        """Get an item while also applying the proper transform chain.

        Args:
            idx (int): Index of the data to retrieve

        Returns:
            DataContainer: Transformed data container
        """
        # Validate index
        if idx < 0 or idx >= len(self):
            msg = f"Index {idx} out of range."
            raise IndexError(msg + (f" Must be within [0, {len(self) - 1}]" if len(self) > 0 else " Empty dataset"))

        if self.cache_built:
            if self.__cache_storage == "disk":
                data = DataContainer()
                data.load_from_hdf5(str(self.__cache[idx]))
            else:
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

    def _save_cache_item_to_disk(self, idx: int) -> Path:
        """Load a single item, apply deterministic transforms, and save it to disk."""
        assert self.__cache_dir is not None
        container = self._load_cache_item(idx)
        path = self.__cache_dir / f"{idx:06d}.hdf5"
        container.save_to_hdf5(str(path))
        return path

    def _load_cache_item_from_disk(self, idx: int) -> DataContainer:
        """Load a single preprocessed item from disk."""
        assert self.__cache_dir is not None
        container = DataContainer()
        container.load_from_hdf5(str(self.__cache_dir / f"{idx:06d}.hdf5"))
        return container

    def memory_bytes(self) -> int:
        """Calculate the memory usage of this dataset.

        **Note:** If the cache has not been built, this will be small because the data is not loaded yet.

        Returns:
            int: Memory usage in bytes
        """
        if self.__cache_storage == "disk":
            return sys.getsizeof(self) + sys.getsizeof(self.__cache)

        container_size = sum(c.memory_bytes() for c in self.__cache if isinstance(c, DataContainer))
        return container_size + sys.getsizeof(self) + sys.getsizeof(self.__cache)

    def print_memory_usage(self):
        """Print the memory usage of this dataset."""
        print(f"{self.__class__.__name__} Overview:")
        print("-" * len(f"{self.__class__.__name__} Overview:"))
        if self.__cache_storage == "disk":
            print(f"Disk cache active at: {self.__cache_dir}")
            print(f"Currently there are {sum(1 for item in self.__cache if item)} cached files")
        else:
            print(f"Currently there are {sum(1 for item in self.__cache if item)} items in the cache")
            if self.__cache_dir is not None:
                print(f"Disk backup at: {self.__cache_dir}")
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

    def build_cache(
        self,
        mode: CacheMode = "lazy",
        num_workers: int | None = None,
        cache_dir: str | Path | None = None,
        keep_in_memory: bool = False,
    ):
        # fmt: off
        """Build a cache of preprocessed data for faster training.

        Automatically splits the transform pipeline at the first random transform:
        - Deterministic transforms are applied once and cached
        - Random transforms + subsequent operations run at runtime

        When ``cache_dir`` is provided, the deterministic outputs are written to disk as one HDF5 file per sample.
        If ``keep_in_memory`` is also True, the files are additionally loaded into the in-memory cache so that
        runtime access is memory-backed while the disk copy remains available for reuse. The directory is managed
        manually by the caller.

        Platform Considerations:
            **Windows / spawn**: Prefer lazy mode so workers share one cache via a manager list instead of each
            holding a full copy after process start.
            **Linux**: Both modes work efficiently - choose based on workflow preference.

            With lazy mode the parent process owns the manager; DataLoader workers receive only the picklable
            list proxy. Keep the parent dataset (and its cache) alive for the lifetime of the workers.

        Args:
            mode (CacheMode): Cache building strategy
                - "lazy": Create shared cache, workers fill on-demand for faster startup (default)
                - "immediate": Build all items upfront using a ThreadPool
            num_workers (int, optional): Number of workers used for cache building. This setting only applies if `mode`
                is "immediate". If num_workers is None, the global configuration of PyThermoNDT will be used.
                If less than 1, it defaults to 1 worker. Default is None.
            cache_dir (str | Path | None, optional): Directory where the cache is stored on disk. If provided, the cache
                is written to disk instead of memory. Existing files in the directory are reused if their count matches
                the dataset length. Default is None.
            keep_in_memory (bool, optional): If True and ``cache_dir`` is provided, load the saved files back into the
                in-memory cache so runtime access is memory-backed. Default is False.

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

            >>> # Disk cache for hyperparameter search
            >>> dataset.build_cache(mode="immediate", num_workers=8, cache_dir="./cache/preprocessed")

            >>> # Disk-backed memory cache: persisted to disk, held in memory at runtime
            >>> dataset.build_cache(
            ...     mode="immediate", num_workers=8, cache_dir="./cache/preprocessed", keep_in_memory=True
            ... )
        """
        # fmt: on
        # Skip if cache already built
        if self.__cache_built:
            return

        # Get the complete transform chain and split it into deterministic and runtime transforms
        self.__det_transforms, self.__runtime_transforms = split_transforms_for_caching(self.get_transform_chain())

        # Configure disk cache
        self.__cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.__cache_storage = "disk" if self.__cache_dir is not None else "memory"

        if keep_in_memory and self.__cache_dir is None:
            raise ValueError("keep_in_memory=True requires cache_dir to be set.")

        if self.__cache_storage == "disk":
            if mode != "immediate":
                raise ValueError("Disk cache currently only supports mode='immediate'.")

            assert self.__cache_dir is not None
            self.__cache_dir.mkdir(parents=True, exist_ok=True)
            num = len(self)
            expected_files = [self.__cache_dir / f"{i:06d}.hdf5" for i in range(num)]

            # Reuse existing files if the count matches; otherwise build them
            existing_files = sorted(self.__cache_dir.glob("*.hdf5"))
            if len(existing_files) != num:
                unit = "files"
                desc = f"{self.__class__.__name__} - Building disk cache"
                workers = max(num_workers, 1) if num_workers is not None else settings.num_workers
                disk_worker_fn = self._save_cache_item_to_disk
                if workers > 1:
                    with ThreadPool(processes=workers) as pool:
                        list(tqdm(pool.imap(disk_worker_fn, range(num)), desc=desc, unit=unit, total=num))
                else:
                    for i in tqdm(range(num), desc=desc, unit=unit):
                        disk_worker_fn(i)

            if keep_in_memory:
                # Load saved files into memory so runtime access is memory-backed
                unit = "files"
                desc = f"{self.__class__.__name__} - Loading disk cache into memory"
                workers = max(num_workers, 1) if num_workers is not None else settings.num_workers
                load_fn = self._load_cache_item_from_disk
                if workers > 1:
                    with ThreadPool(processes=workers) as pool:
                        self.__cache = list(tqdm(pool.imap(load_fn, range(num)), desc=desc, unit=unit, total=num))
                else:
                    self.__cache = [load_fn(i) for i in tqdm(range(num), desc=desc, unit=unit)]
                self.__cache_storage = "memory"
            else:
                self.__cache = expected_files

            self.__cache_built = True
            return

        # Initialize the in-memory cache based on the mode
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
            # Choose context based on platform to avoid deprecated 'fork' in multithreaded processes (Python 3.12+):
            ctx = mp.get_context("forkserver" if sys.platform == "linux" else "spawn")
            self.__manager = ctx.Manager()
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
        self.__cache_dir = None
        self.__cache_storage = "memory"

        # Ensure that the manager process is terminated
        if self.__manager:
            try:
                self.__manager.shutdown()
            except Exception:  # pylint: disable=broad-except
                pass
            finally:
                self.__manager = None

        # Garbage collect to free memory if requested
        if gc_collect:
            gc.collect()
