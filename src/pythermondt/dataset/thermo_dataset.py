from collections.abc import Iterator, Sequence
from itertools import chain

import torch

from ..data import DataContainer
from ..readers.base_reader import BaseReader
from ..transforms.utils import _BaseTransform
from .base import BaseDataset


class ThermoDataset(BaseDataset):
    """PyTorch dataset that combines multiple readers into a single dataset.

    Automatically handles file indexing across different data sources (local, S3, etc.)
    and applies transforms consistently. Compatible with PyTorch DataLoaders.
    """

    def __init__(self, data_source: BaseReader | Sequence[BaseReader], transform: _BaseTransform | None = None):
        """Create dataset from one or more readers.

        Files are discovered and downloaded (if remote) during initialization for
        predictable performance during training.

        Parameters:
            data_source: Single reader or list of readers to combine
            transform: Optional transform applied to each container when loaded

        Raises:
            ValueError: If duplicate files found or no files available
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

        # Eagerly load files from all readers
        for reader in self.__readers:
            # Download files if remote and download_files is True
            if reader.remote_source and reader.download_files:
                reader.download()

        # Build the index map
        self._build_index()

    def _validate_readers(self, readers: list[BaseReader]):
        """Validate readers and check for duplicates."""
        # Check if the readers have found any files and if there are any duplicates
        # Check if all readers have enabled file caching
        for reader in readers:
            # Check for stable file list during training
            if not reader.cache_files:
                print(
                    f"Warning: {reader.__class__.__name__} has cache_files=False. File list may change during training."
                )

            # Check for efficient remote access
            if reader.remote_source and not reader.download_files:
                print(
                    f"Warning: {reader.__class__.__name__} is remote but download_files=False. "
                    f"This will be slower for repeated access."
                )

        # Group all the readers by type
        readers_by_type: dict[type[BaseReader], list[BaseReader]] = {}
        for reader in readers:
            readers_by_type[type(reader)] = readers_by_type.get(type(reader), []) + [reader]

        # Check if any of the readers that are of the same type find duplicate or invalid data
        for reader_type, readers in readers_by_type.items():
            # When there a multiple readers ==> check for duplicate files
            if len(readers) > 1:
                all_files: set[str] = set()
                duplicate_files: set[str] = set()

                for reader in readers:
                    # Check if the reader has found any files
                    if not reader.files:
                        raise ValueError(f"No files found for reader of type {reader_type.__qualname__}")

                    # Check for duplicate files
                    reader_files = set(reader.files)
                    new_duplicates = reader_files.intersection(all_files)
                    if new_duplicates:
                        duplicate_files.update(new_duplicates)

                    all_files.update(reader_files)

                if duplicate_files:
                    raise ValueError(
                        f"Duplicate files found for reader of type {reader_type.__qualname__}: \n {duplicate_files}"
                    )

            # Else duplicates are not possible ==> Check if the reader has found any files
            else:
                if len(readers[0].files) == 0:
                    raise ValueError(f"No files found for reader of type {reader_type.__qualname__}")

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

    def _load_raw_data(self, idx: int) -> DataContainer:
        """Load raw data from readers - required by BaseDataset."""
        # Extract reader and file index from the index map
        r_idx = int(self.__reader_index[idx].item())
        f_idx = int(self.__file_index[idx].item())

        try:
            return self.__readers[r_idx][f_idx]
        except FileNotFoundError:
            print(f"File not found for reader {self.__readers[r_idx].__class__.__name__} at index {f_idx}")
            return DataContainer()
        except Exception as e:
            print(f"Error reading file for reader {self.__readers[r_idx].__class__.__name__} at index {f_idx}: {e}")
            return DataContainer()

    @property
    def files(self) -> list[str]:
        return [file for reader in self.__readers for file in reader.files]

    def __len__(self) -> int:
        return sum([len(reader.files) for reader in self.__readers])

    def __getitem__(self, idx) -> DataContainer:
        """Get data container at index with transforms applied."""
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        # Load raw data
        data = self._load_raw_data(idx)

        # Apply transform if present
        if self.transform:
            data = self.transform(data)

        return data

    def __iter__(self) -> Iterator[DataContainer]:
        return (self.transform(data) if self.transform else data for data in chain.from_iterable(self.__readers))
