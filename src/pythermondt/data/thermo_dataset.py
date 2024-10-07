import torch
from itertools import chain
from typing import Type, Dict, List, Optional, Iterator
from torch.utils.data import Dataset
from .datacontainer import DataContainer
from ..transforms import ThermoTransform
from ..readers.base_reader import BaseReader

class ThermoDataset(Dataset):
    def __init__( self, data_source: BaseReader |List[BaseReader], transform: Optional[ThermoTransform] = None):
        """ Initialize a PyTorch dataset from a list of readers.
        
        The sources are used to read the data and create the dataset. First the readers are grouped by type.
        Then all readers of the same type are checked for duplicate files. If any duplicates are found, an error is raised.

        **Note**: When combining readers, it is recommend that all readers enable file caching. The readers accomodate for this by raising an error if a file is not found and
        taking a snapshot of the files when a iterator is created. However, enabling file caching is still recommended to avoid issues with changing files. A warning is printed if a reader does not have file caching enabled.

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

        # Read the file from the reader
        data = self.__readers[reader_idx][file_idx]

        # Apply the transform if any is given
        if self.__transform:
            data = self.__transform(data)
        
        return data
    
    def __iter__(self) -> Iterator[DataContainer]:
        return chain.from_iterable(self.__readers)