import os
import re
from glob import glob
from abc import ABC, abstractmethod
from typing import Generator, List, Tuple, Optional
from ..data import DataContainer
from ..transforms import ThermoTransform

class BaseReader(ABC):
    @abstractmethod
    def __init__(self, source: str, file_extension: str | Tuple[str, ...], cache_paths: bool = True, transform: Optional[ThermoTransform] = None):
        """Initialize the Reader with a single source.

        Parameters:
            source (str): Source expression to match files. Can either point to a single file, a directory or a regex pattern to match multiple files.
            file_extension (str or Tuple[str]): File extension(s) of the data files to load. Can be a single string or a tuple of strings.
            cache_paths (bool, optional): If True, all file paths in the source directory will be cached. Therefore updates to the source directory will not be reflected at runtime. Default is True.
            transform (ThermoTransform, optional): Optional transform to be applied on the data before it is loaded. Default is None.
        """
        # Convert file_extension to a tuple if it is a single string
        extensions = file_extension if isinstance(file_extension, tuple) else (file_extension,)

        # Check if file_extension is valid
        if not all(isinstance(ext, str) for ext in extensions):
            raise TypeError("All items in the list must be strings.")
        if not all(ext.startswith('.') for ext in extensions):
            raise ValueError("All file extensions must start with a dot.")
        self.file_extensions = extensions

        # Set the source expression
        self.source = source

        # Transforms to apply
        self.transform = transform

        # Boolean flag to enable caching of file paths
        self.cache_paths = cache_paths

        # Index to keep track of the current file being read ==> used for iteration
        self._current_file_index = 0

        # Lists that contain Cached file names and paths
        self._cached_file_names = None
        self._cached_file_paths = None

    def __iter__(self):
        self._current_file_index = 0
        return self

    def __next__(self) -> DataContainer:
        # Get all file paths in the source directory
        files = self.file_paths()

        # Check if the current file index is out of bounds
        if self._current_file_index >= len(files):
            raise StopIteration

        data = self[self._current_file_index]
        self._current_file_index += 1

        return data

    def __getitem__(self, index: int) -> DataContainer:
        # Get all file paths in the source directory
        files = self.file_paths()

        # Check if the index is out of bounds
        if index < 0 or index >= len(files):
            raise IndexError("Index out of bounds.")

        # Load data from the file at the specified index
        return self.read_data(files[index])
    
    @property
    def source(self) -> str:
        """
        The source expression used to match files. Can either point to a single file, a directory or a regex pattern to match multiple files.

        Returns:
            str: The source expression.
        """
        return self.__source
    
    @source.setter
    def source(self, source: str):
        # Check if source is a valid regex pattern
        try:
            re.compile(source)
            valid_regex = True
        except re.error:
            valid_regex = False

        # Check if the provided source is either a file, a directory or a regex pattern
        if os.path.isfile(source):
            self._source_type = "file"

        elif os.path.isdir(source):
            self._source_type = "directory"

        elif valid_regex:
            self._source_type = "regex"

        else:
            raise ValueError("The provided source must either be a file, a directory or a valid regex pattern.")
        
        # Write the source expression to the private variable
        self.__source = source

    @property
    def num_files(self) -> int:
        """
        Get the number of files in the source directory

        Returns:
            int: The number of files in the source directory.
        """
        return len(self.file_paths())

    def file_names(self) -> List[str]:
        """
        Get all the file names in the source directory, specified by the source expression and file extension of the reader.

        Returns:
            List[str]: A list of file names.
        """
        # If caching is on and the file names are already cached, return the cached file names
        if self._cached_file_names is not None and self.cache_paths:
            return self._cached_file_names
        # If caching is off, reset the cached file names
        elif not self.cache_paths:
            self._cached_file_names = None

        file_names = [os.path.basename(f) for f in self.file_paths()]

        # Cache the file names if caching is enabled
        if self.cache_paths:
            self._cached_file_names = file_names

        return file_names
    
    def file_paths(self) -> List[str]:
        """
        Get all the file paths in the source directory, specified by the source expression and file extension of the reader.

        Returns:
            List[str]: A list of file paths.
        """
        # If caching is on and the file paths are already cached, return the cached file names
        if self._cached_file_paths is not None and self.cache_paths:
            return self._cached_file_paths
        # If caching is off, reset the cached file names
        elif not self.cache_paths:
            self._cached_file_paths = None

        # Resolve the source pattern based on the source type
        match self._source_type:
            case "file":
                file_paths = [self.source]

            case "directory":
                file_paths = glob(os.path.join(self.source, "*"))

            case "regex":
                file_paths = glob(self.source)

            case _:
                raise ValueError("Invalid source type.")

        # Check if the found files match the specified file extension
        file_paths = [f for f in file_paths if any(f.endswith(ext) for ext in self.file_extensions)]
        if not file_paths:
            raise ValueError("No files found. Please check the source expression and file extensions")

        # Cache the file paths if caching is enabled
        if self.cache_paths:
            self._cached_file_paths = file_paths
        
        return file_paths

    def read_data(self, file_path: str) -> DataContainer:
        """
        Read data from a given file path and return it as a DataContainer. Also checks if the file extension is valid.

        Parameters:
            file_path (str): The file path from which to load the data.

        Returns:
            DataContainer: A DataContainer instance containing data loaded from the file.
        """
        # Check if the file extension of the file is valid
        if not any(file_path.endswith(ext) for ext in self.file_extensions):
            raise ValueError("Invalid file extension. Must be one of: " + str(self.file_extensions))

        # Load the data from the file
        data = self._read_data(file_path)

        # Apply the transform if it is not None
        if self.transform is not None:
            return self.transform(data)
        else:
            return data

    @abstractmethod
    def _read_data(self, file_path: str) -> DataContainer:
        """
        Actual implementation of reading data from a given file path and returning it as a DataContainer. Should be implemented by subclasses. Checking if the path is valid 
        is handled inside the _BaseReader class.

        Parameters:
            file_path (str): The file path from which to load the data.

        Returns:
            DataContainer: A DataContainer instance containing data loaded from the file.
        """
        pass

    def read_data_batch(self, batch_size: int) -> Generator[List[DataContainer], None, None]:
        """
        Generator to yield batches of data dynamically from the list of files in the source directory. Files are filtered by the file extension specified in the reader.

        Parameters:
            batch_size (int): Number of files to load per batch.

        Yields:
            List[DataContainer]: A batch of DataContainers loaded from batch_size files.
        """
        # Check if source is a directory
        if not os.path.isdir(self.source):
            raise ValueError("The source must be a directory for batch processing.")
        
        # Get all file names in the source directory
        files = self.file_paths()

        # Validate arguments
        if not files:
            raise ValueError("No files found in the specified directory.")

        if batch_size < 1:
            raise ValueError("Invalid batch size. Must be at least 1.")
        
        if batch_size > len(files):
            raise ValueError("Batch size is greater than the number of files in the directory.")

        # Load data in batches
        for i in range(0, len(files), batch_size):
            # Check if the last batch is smaller than the batch size ==> just take the remaining files
            if i + batch_size > len(files):
                batch_files = files[i:]

            # Normal case: Take the next batch_size files
            else:
                batch_files = files[i:i + batch_size]

            # Load data from each file in the batch
            batch_data = [self.read_data(f) for f in batch_files]

            # Type check the loaded data
            if not all(isinstance(data, DataContainer) for data in batch_data):
                raise TypeError("All items in the batch must be instances of DataContainer.")
            
            # Yield the batch of data
            yield batch_data

    def read_data_all(self) -> List[DataContainer]:
        """
        Load all data from the list of files in the source directory. Files are filtered by the file extension specified in the reader. Be careful with large amounts of files. Memory usage can be high!

        Returns:
            List[DataContainer]: A list containing all DataContainers loaded from the files.
        """
        return [self.read_data(f) for f in self.file_paths()]