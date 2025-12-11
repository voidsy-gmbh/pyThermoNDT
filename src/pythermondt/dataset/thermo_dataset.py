import warnings
from collections.abc import Iterator, Sequence

import torch

from ..data import DataContainer
from ..readers.base_reader import BaseReader
from ..transforms.utils import _BaseTransform
from .base_dataset import BaseDataset


class ThermoDataset(BaseDataset):
    """PyTorch dataset that combines multiple readers into a single dataset.

    Automatically handles file indexing across different data sources (local, S3, etc.)
    and applies transforms consistently. Compatible with PyTorch DataLoaders.
    """

    def __init__(self, data_source: BaseReader | Sequence[BaseReader], transform: _BaseTransform | None = None):
        """Create dataset from one or more readers.

        Files are discovered and downloaded (if remote) during initialization for
        predictable performance during training.

        Args:
            data_source: Single reader or list of readers to combine
            transform: Optional transform applied to each container when loaded

        Raises:
            ValueError: If no readers are provided
        """
        # Initialize base with no parent (this is root) and given transform
        super().__init__(parent=None, transform=transform)

        # Convert single reader to list
        data_source = [data_source] if isinstance(data_source, BaseReader) else list(data_source)

        # Check if data_source is empty
        if not data_source:
            raise ValueError("No readers provided. Please provide at least one BaseReader instance or a list of them.")

        # Validate the readers
        self._validate_readers(data_source)

        # Write the readers to the private attributes
        self.__readers = data_source

        # Build the index map
        self._build_index()

    def _validate_readers(self, readers: list[BaseReader]):
        """Validate readers and check for duplicates."""
        # Check if the readers have found any files and if there are any duplicates
        # Check if all readers have enabled file caching
        for reader in readers:
            # Check for stable file list during training
            if not reader.cache_files:
                warnings.warn(
                    f"{reader.__class__.__name__} has cache_files=False. File list may change during training.",
                    stacklevel=2,
                )

            # Check for efficient remote access
            if reader.remote_source and not reader.download_files:
                warnings.warn(
                    f"{reader.__class__.__name__} is remote but download_files=False. "
                    f"This will be slower for repeated access.",
                    stacklevel=2,
                )

        # Group all the readers by type
        readers_by_type: dict[type[BaseReader], list[BaseReader]] = {}
        for reader in readers:
            readers_by_type[type(reader)] = readers_by_type.get(type(reader), []) + [reader]

        # Check if any of the readers that are of the same type find duplicate or invalid data
        for reader_type, readers_objects in readers_by_type.items():
            # When there a multiple readers ==> check for duplicate files
            if len(readers_objects) > 1:
                all_files: set[str] = set()
                duplicate_files: set[str] = set()

                for reader in readers_objects:
                    # Check if the reader has found any files
                    if not reader.files:
                        warnings.warn(f"No files found for reader of type {reader_type.__qualname__}", stacklevel=2)

                    # Check for duplicate files
                    reader_files = set(reader.files)
                    new_duplicates = reader_files.intersection(all_files)
                    if new_duplicates:
                        duplicate_files.update(new_duplicates)

                    all_files.update(reader_files)

                if duplicate_files:
                    warnings.warn(
                        f"Duplicate files found for reader of type {reader_type.__qualname__}: \n {duplicate_files}",
                        stacklevel=2,
                    )

            # Else duplicates are not possible ==> Check if the reader has found any files
            else:
                if len(readers[0].files) == 0:
                    warnings.warn(f"No files found for reader of type {reader_type.__qualname__}", stacklevel=2)

    def _build_index(self):
        """Build an index map using 2 torch Tensors.

        This is done to achieve a fast and memory efficient mapping of reader and file index to the global index
        of the dataset. 1 Tensor is used to store the reader index and the other Tensor is used to store the file index.
        """
        reader_indices: list[int] = []
        file_indices: list[int] = []
        for reader_idx, reader in enumerate(self.__readers):
            reader_indices.extend([reader_idx] * len(reader.files))
            file_indices.extend(range(len(reader.files)))

        self.__reader_index = torch.tensor(reader_indices, dtype=torch.uint8, requires_grad=False)
        self.__file_index = torch.tensor(file_indices, dtype=torch.int32, requires_grad=False)

    def download(self, num_workers: int | None = None) -> None:
        """Download all files from all readers that support downloading.

        This will call download() on each reader that has a remote source.

        Args:
            num_workers (int, optional): Number of workers for parallel downloads.
                Passed to each reader's download() method. If None, uses the
                global configuration from settings.
        """
        for reader in self.__readers:
            if reader.remote_source:
                reader.download(num_workers=num_workers)

    def load_raw_data(self, idx: int) -> DataContainer:
        """Load raw data from readers - required by BaseDataset."""
        # Validate index first
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        # Extract reader and file index from the index map
        r_idx = int(self.__reader_index[idx].item())
        f_idx = int(self.__file_index[idx].item())

        try:
            return self.__readers[r_idx][f_idx]
        except (FileNotFoundError, OSError, PermissionError) as e:
            # File system errors
            msg = f"{self.__readers[r_idx].__class__.__name__}: Cannot read file '{self.files[idx]}' at index {f_idx}"
            raise RuntimeError(msg) from e
        except ValueError as e:
            # Parser/extension errors from BaseReader
            msg = f"{self.__readers[r_idx].__class__.__name__}: Cannot parse file '{self.files[idx]}' at index {f_idx}"
            raise ValueError(msg) from e

    @property
    def files(self) -> list[str]:
        return [file for reader in self.__readers for file in reader.files]

    def __len__(self) -> int:
        return sum(len(reader.files) for reader in self.__readers)

    def __iter__(self) -> Iterator[DataContainer]:
        return (self[i] for i in range(len(self)))
