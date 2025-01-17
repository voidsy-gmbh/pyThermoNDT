import torch
from itertools import chain
from typing import Type, Dict, List, Optional, Iterator, Sequence
from torch.utils.data import Dataset
from .datacontainer import DataContainer
from ..transforms import ThermoTransform
from ..readers.base_reader import BaseReader

class ThermoDataset(Dataset):
    """ Custom dataset class used for combining data, read from multiple readers.

    This dataset is used to combine multiple readers into a single dataset. The dataset can be used to read data from multiple sources and apply a transform to the data.
    Like a normal PyTorch dataset, the dataset can be used to iterate over the data using the __getitem__ method and it is also compatible with PyTorch dataloaders.
    """
    def __init__( self, data_source: BaseReader | List[BaseReader], transform: Optional[ThermoTransform] = None):
        """ Initialize a custom PyTorch dataset from a list of readers.
        
        The sources are used to read the data and create the dataset. First the readers are grouped by type.
        Then all readers of the same type are checked for duplicate files. If any duplicates are found, an error is raised.

        **Note**: When combining readers, it is recommend that all readers enable file caching, especially in cases where files need to be fetched from a remote location and these files are changing at runtime. The readers accomodate for this by raising an error if a file is not found and
        taking a snapshot of the files list when a iterator is created. The Dataset will catch these errors when trying to read the data and return an **empty Datacontainer**
        
        However, enabling file caching is still recommended to avoid issues with changing files, especially when using a dataset for model training. A warning is printed if any reader does not have file caching enabled.

        Parameters:
            data_source (List[BaseReader]): List of readers to be used as a data source for the dataset
            transform (ThermoTransform, optional): Optional transform to be directly applied to the data when it is read
            
        Raises:
            ValueError: If any of the readers of the same type find duplicate or invalid data
            ValueError: If any of the readers of the same type do not find any files
        """
        # Convert single reader to list
        data_source = data_source if isinstance(data_source, list) else [data_source]

        # Validate the readers
        self._validate_readers(data_source)

        # Write the readers and transforms to the private attributes
        self.__readers = data_source
        self.__transform = transform

        # Build the index map
        self._build_index()

    def _validate_readers(self, readers: List[BaseReader]):
        """Validate readers and check for duplicates."""
        # Check if the readers have found any files and if there are any duplicates
        # Check if all readers have enabled file caching
        for reader in readers:
            if not reader.cache_files:
                print(f"Warning: Reader {reader.__repr__()} does not have file caching enabled. This can lead to issues with changing files. Consider enabling file caching.")

        # Group all the readers by type
        readers_by_type: Dict[Type[BaseReader], List[BaseReader]] = {}  
        for reader in readers:
            readers_by_type[type(reader)] = readers_by_type.get(type(reader), []) + [reader]

        # Check if any of the readers that are of the same type find duplicate or invalid data
        for reader_type, readers in readers_by_type.items():
            # When there a multiple readers ==> check for duplicate files
            if len(readers) > 1:
                all_files = set()
                duplicate_files = set()

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
                    raise ValueError(f"Duplicate files found for reader of type {reader_type.__qualname__}: \n {duplicate_files}")
                
            # Else duplicates are not possible ==> Check if the reader has found any files
            else:
                if len(readers[0].files) == 0:
                    raise ValueError(f"No files found for reader of type {reader_type.__qualname__}")

    def _build_index(self):
        """Build an index map using 2 torch Tensors for fast and memory efficient mapping of reader and file index to the global index of the dataset."""
        reader_indices = []
        file_indices = []
        for reader_idx, reader in enumerate(self.__readers):
            reader_indices.extend([reader_idx] * len(reader.files))
            file_indices.extend(range(len(reader.files)))
        
        self.__reader_index = torch.tensor(reader_indices, dtype=torch.uint8, requires_grad=False)
        self.__file_index = torch.tensor(file_indices, dtype=torch.int32, requires_grad=False)

    @property
    def files(self) -> List[str]:
        return [file for reader in self.__readers for file in reader.files]
    
    def __len__(self) -> int:
        return sum([len(reader.files) for reader in self.__readers])
    
    def __getitem__(self, idx) -> DataContainer:
        # Check if the index is valid
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        
        # Extract the reader and file index from the index map
        reader_idx = int(self.__reader_index[idx].item())
        file_idx = int(self.__file_index[idx].item())

        # Try to read the file
        try:
            # Read the file from the reader
            data = self.__readers[reader_idx][file_idx]

            # Apply the transform if any is given
            if self.__transform:
                data = self.__transform(data)
            
            return data

        except FileNotFoundError:
            print(f"File not found for reader {self.__readers[reader_idx].__repr__()} at index {file_idx}")
        
        except Exception as e:
            print(f"Error reading file for reader {self.__readers[reader_idx].__repr__()} at index {file_idx}: {e}")

        # Return an empty DataContainer if the file could not be read
        return DataContainer()

    def __iter__(self) -> Iterator[DataContainer]:
        return chain.from_iterable(self.__readers)

class IndexedThermoDataset(ThermoDataset):
    """Extension of ThermoDataset that supports indexing with optional additional transforms.

    The IndexedThermoDataset maintains a subset of the parent dataset and allows for an additional transform to be applied to the data. 
    This can be useful when a subset of the data needs to be selected and a different transform needs to be applied to the subset, e.g. for random splits of 
    train, validation and test data. The IndexedThermoDataset maintains the transform chain of the parent dataset and appends the additional transform to it.
    """
    
    def __init__(self, dataset: ThermoDataset, indices: Sequence[int], transform: Optional[ThermoTransform] = None):
        """Initialize an indexed dataset with optional additional transform.

        Parameters:
            dataset (ThermoDataset): Parent dataset to index into
            indices (Sequence[int]): Sequence of indices to select from parent
            transform (Optional[ThermoTransform]): Optional transform to apply after parent's transform

        Raises:
            IndexError: If any of the provided indices are out of range
        """
        # Validate the indices
        if not all(0 <= i < len(dataset) for i in indices):
            raise IndexError(f"Provided indices are out of range. Must be within [0, {len(dataset)-1}]")

        # Store parent dataset and indices
        self.__dataset = dataset  # Original dataset
        self.__indices = indices  # Indices for subset
        self.__transform = transform  # Additional transform

    def __len__(self) -> int:
        """Return length of indexed dataset."""
        return len(self.__indices)

    def __getitem__(self, idx: int) -> DataContainer:
        """Get an item with proper transform chain.
        
        Args:
            idx (int): Index into the subset
            
        Returns:
            DataContainer: Transformed data container
        """
        # Validate index
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
            
        # Get data using parent dataset's underlying logic ==> Apply parent transform
        data = self.__dataset[self.__indices[idx]]
        
        # Apply additional transform if specified
        if self.__transform:
            data = self.__transform(data)
            
        return data

    @property
    def files(self) -> List[str]:
        """Return list of files corresponding to indexed subset."""
        return [self.__dataset.files[i] for i in self.__indices]