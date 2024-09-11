from typing import Type, Dict, List, Optional
from torch.utils.data import Dataset
from .datacontainer import DataContainer
from ..transforms import ThermoTransform
from ..readersv2.base_reader import BaseReader

class ThermoDataset(Dataset):
    def __init__( self, data_source: BaseReader |List[BaseReader], transform: Optional[ThermoTransform] = None, cache_source: bool = False):
        """ Initialize a PyTorch dataset from a list of readers.
        
        The sources are used to read the data and create the dataset. First the readers are grouped by type.
        Then all readers of the same type are checked for duplicate files. If any duplicates are found, an error is raised.

        Parameters:
            data_source (List[BaseReader]): List of readers to be used as a data source for the dataset
            transform (ThermoTransform, optional): Optional transform to be directly applied to the data when it is read
            cache_source (bool, optional): If True, all the file paths are cached in memory. Therefore changes to the file sources will not be noticed at runtime. Default is False.

        Raises:
            ValueError: If any of the readers of the same type find duplicate or invalid data
            ValueError: If any of the readers of the same type do not find any files
        """
        # Convert single reader to list
        data_source = data_source if isinstance(data_source, list) else [data_source]

        # Check if the readers have found any files and if there are any duplicates
        # Group all the readers by type
        readers_by_type: Dict[Type[BaseReader], List[BaseReader]] = {}  
        for reader in data_source:
            readers_by_type[type(reader)] = readers_by_type.get(type(reader), []) + [reader]

        # Check if any of the readers that are of the same type find duplicate or invalid data
        for reader_type, readers in readers_by_type.items():
            # When there a multiple readers ==> check for duplicate files
            if len(readers) > 1:
                # A list of all list of file paths that the readers have found
                file_lists = [reader.files for reader in readers]

                # Check if any of the found lists intersect with each other ==> If so, there are duplicate files found
                intersection = set(file_lists[0]).intersection(*file_lists[1:])
                if len(intersection) > 0:
                    raise ValueError(f"Duplicate files found for reader of type {reader_type.__qualname__}: \n {intersection}")
                
            # Else duplicates are not possible ==> Check if the reader has found any files
            else:
                if len(readers[0].files) == 0:
                    raise ValueError(f"No files found for reader of type {reader_type.__qualname__}")
        
        # Write the readers to the private attribute
        self.__readers = data_source

    @property
    def files(self) -> List[str]:
        return [file for reader in self.__readers for file in reader.files]
    
    def __len__(self) -> int:
        return sum([len(reader.files) for reader in self.__readers])
    
    def __getitem__(self, idx) -> DataContainer:
        # Every call of self.files would trigger the readers to check for new files ==> get files list once
        files = self.files

        # Check if the index is valid
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        
        # Find the reader that contains the file
        reader_idx = 0
        while idx >= len(files):
            idx -= len(self.__readers[reader_idx].files)
            reader_idx += 1
        
        # Read the file from the reader
        return self.__readers[reader_idx].read(files[idx])