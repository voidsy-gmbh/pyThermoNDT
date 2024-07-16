import os
from abc import ABC, abstractmethod
from typing import Generator, List, Tuple
from ..data import DataContainer

class _BaseReader(ABC):
    @abstractmethod
    def __init__(self, source: str, file_extension: str | Tuple[str, ...], filter_files: bool = True, cache_paths: bool = True):
        """
        Initialize the DataReader with a single source.

        Parameters:
        - source (str): Path to the directory containing data files or a single data file.
        - file_extension (str or Tuple[str]): File extension(s) of the data files to load. Can be a single string or a tuple of strings.
        - filter_files (bool): If True, only files with the specified file extension will be loaded. Default is True.
        - cache_paths (bool): If True, all file paths in the source directory will be cached. Therefore updates to the source directory 
            will not be reflected at runtime. Default is True.
        """
        # Check if source is a valid path
        if not os.path.exists(source):
            raise ValueError("The source path does not exist.")
        
        if not os.path.isdir(source) and not os.path.isfile(source):
            raise ValueError("The source must be a directory or a file.")
        self.source = source

        # Convert file_extension to a tuple if it is a single string
        extensions = file_extension if isinstance(file_extension, tuple) else (file_extension,)

        # Check if file_extension is valid
        if not all(isinstance(ext, str) for ext in extensions):
            raise TypeError("All items in the list must be strings.")
        if not all(ext.startswith('.') for ext in extensions):
            raise ValueError("All file extensions must start with a dot.")
        self.file_extension = extensions

        # Boolean flag to filter files by file extension
        self.filter_files = filter_files

        # Boolean flag to enable caching of file paths
        self.cache_paths = cache_paths

        # Index to keep track of the current file being read ==> used for iteration
        self._current_file_index = 0

        # Cached file names and paths
        self._cached_file_names = None
        self._cached_file_paths = None

    def __iter__(self):
        self._current_file_index = 0
        return self

    def __next__(self) -> DataContainer:
        # Get all file paths in the source directory
        files = self.file_paths(filter_files=self.filter_files)

        # Check if the current file index is out of bounds
        if self._current_file_index >= len(files):
            raise StopIteration

        data = self[self._current_file_index]
        self._current_file_index += 1

        return data

    def __getitem__(self, index: int) -> DataContainer:
        # Get all file paths in the source directory
        files = self.file_paths(filter_files=self.filter_files)

        # Check if the index is out of bounds
        if index < 0 or index >= len(files):
            raise IndexError("Index out of bounds.")

        # Load data from the file at the specified index
        return self.read_data(files[index]) 

    @property
    def num_files(self) -> int:
        """
        Get the number of files in the source directory

        Returns:
        - int: The number of files in the source directory.
        """
        return len(self.file_names())
    

    def file_names(self, filter_files: bool | None = None) -> List[str]:
        """
        Get all the file names in the source directory. Optionally filter files by file extension.

        Parameters:
        - filter_files (bool): If True, only files with the specified file extension will be returned. Default is specified in the class attribute.
        Returns:
        - List[str]: A list of file names in the source directory.
        """
        # Use the function attribute if filter_files is not provided ==> use class attribute
        if filter_files is None:
            filter_files = self.filter_files
        
        # If caching is on and the file names are already cached, return the cached file names
        if self._cached_file_names is not None and self.cache_paths:
            return self._cached_file_names
        # If caching is off, reset the cached file names
        elif not self.cache_paths:
            self._cached_file_names = None

        # Retrieve all files in the source directory
        files = [f for f in os.listdir(self.source) if os.path.isfile(os.path.join(self.source, f))]

        # Filter files by file extension
        if filter_files:
            files = [f for f in files if f.endswith(self.file_extension)]

        # Cache the file names if caching is enabled
        if self.cache_paths:
            self._cached_file_names = files

        return files
    
    def file_paths(self, filter_files: bool | None = None) -> List[str]:
        """
        Get all the file paths in the source directory. Optionally filter files by file extension.

        Parameters:
        - filter_files (bool): If True, only files with the specified file extension will be returned. Default is specified in the class attribute.

        Returns:
        - List[str]: A list of file paths in the source directory.
        """
        # Use the function attribute if filter_files is not provided ==> use class attribute
        if filter_files is None:
            filter_files = self.filter_files

        # If caching is on and the file names are already cached, return the cached file names
        if self._cached_file_paths is not None and self.cache_paths:
            return self._cached_file_paths
        # If caching is off, reset the cached file names
        elif not self.cache_paths:
            self._cached_file_paths = None

        # Get all file names in the source directory
        files = self.file_names(filter_files=filter_files)

        # Get the full file path for each file
        paths = [os.path.join(self.source, f) for f in files]

        # Cache the file paths if caching is enabled
        if self.cache_paths:
            self._cached_file_paths = paths
        
        return paths

    def read_data(self, file_path: str) -> DataContainer:
        """
        Read data from a given file path and return it as a DataContainer. Also checks if the file extension is valid.

        Parameters:
        - file_path (str): The file path from which to load the data.

        Returns:
        - DataContainer: A DataContainer instance containing data loaded from the file.
        """
        # Check if the file extension of the file is valid
        if not any(file_path.endswith(ext) for ext in self.file_extension):
            raise ValueError("Invalid file extension. Must be one of: " + str(self.file_extension))

        # Load the data from the file and return
        return self._read_data(file_path)

    @abstractmethod
    def _read_data(self, file_path: str) -> DataContainer:
        """
        Actual implementation of reading data from a given file path and returning it as a DataContainer. Should be implemented by subclasses.

        Parameters:
        - file_path (str): The file path from which to load the data.

        Returns:
        - DataContainer: A DataContainer instance containing data loaded from the file.
        """
        pass

    def read_data_batch(self, batch_size: int, filter_files:bool | None = None) -> Generator[List[DataContainer], None, None]:
        """
        Generator to yield batches of data dynamically from the source directory.

        Parameters:
        - batch_size (int): Number of files to load per batch.
        - filter_files (bool): If True, only files with the specified file extension will be loaded. Default is specified in the class attribute.

        Yields:
        - List[DataContainer]: A batch of DataContainers loaded from batch_size files.
        """
        # Use the class attribute if filter_files is not provided
        if filter_files is None:
            filter_files = self.filter_files

        # Check if source is a directory
        if not os.path.isdir(self.source):
            raise ValueError("The source must be a directory for batch processing.")
        
        # Get all file names in the source directory
        files = self.file_paths(filter_files=filter_files)

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

    def read_data_all(self, filter_files: bool | None = None) -> List[DataContainer]:
        """
        Load all data from the source directory and return it as a list of DataContainers if the dataset size is manageable. Be careful with large datasets!

        Parameters:
        - filter_files (bool): If True, only files with the specified file extension will be loaded. Default is specified in the class attribute.

        Returns:
        - List[DataContainer]: A list of all DataContainers loaded from the source directory.
        """
        # Use the class attribute if filter_files is not provided
        if filter_files is None:
            filter_files = self.filter_files

        return [self.read_data(f) for f in self.file_paths(filter_files=filter_files)]