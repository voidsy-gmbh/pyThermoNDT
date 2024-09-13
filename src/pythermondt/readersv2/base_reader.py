import io
import re
import os
from typing import Tuple, Type, Optional, Dict, List
from abc import ABC, abstractmethod
from ..data import DataContainer
from .parsers import BaseParser, HDF5Parser, SimulationParser

FILE_EXTENSIONS: Dict[Type[BaseParser], Tuple[str, ...]] = {
    HDF5Parser: ('.hdf5', '.h5'),
    SimulationParser: ('.mat',),
    # Add more file extensions for future parsers here
}

# Lookup table for file extensions ==> for fast validation of file extensions
FILE_EXTENSIONS_LUT = {ext:parser for parser, extensions in FILE_EXTENSIONS.items() for ext in extensions}

class BaseReader(ABC):
    """ Base class for all readers. This class defines the interface for all readers, subclassing this class.
    """
    @abstractmethod
    def __init__(self, parser: Type[BaseParser], source: str, cache_files: bool = True):
        """ Constructor for the BaseReader class.

        Parameters:
            parser (Type[BaseParser]): The parser that the reader uses to parse the data.
            source (str): The source of the data. This can be a file path, a directory path, a regular expression. In case of cloud storage, this can be a URL.
            cache_files (bool, optional): If True, the reader caches the file paths. If False, the reader retrieves the file list each time. For cloud storage readers, this flag should also determine if the files are downloaded to a local directory. Default is True.
        """
        # Set the parser
        self.__parser = parser

        # Set the file extensions based on what parser is used
        try:
            self.__file_extensions = FILE_EXTENSIONS[self.parser]
        except KeyError:
            raise ValueError(f"The specified Parser: {parser.__name__} is not supported by the {self.__class__.__name__} class.")
        
        # validate that the source expression does not contain an invalid file extension ==> File extensions are defined by the parser
        ext = re.findall(r'\.[a-zA-Z0-9]+$', source)
        correct_parser = FILE_EXTENSIONS_LUT.get(ext[0], None) if len(ext) > 0 else self.parser

        if correct_parser is None:
            raise ValueError(f"The source contains an invalid file extension: '({ext[0]})'! Use a file extensions that is supported by the {self.parser.__name__}: {self.file_extensions}")
        elif correct_parser is not self.parser:
             raise ValueError(f"Wrong parser selected for the file extension: '({ext[0]})'! Use the {correct_parser.__name__} for this file extension instead")
             
        # Set the source
        self.__source = source

        # Set the cache_files flag and the cached_files attribute
        self.__cache_files = cache_files
        self.__cached_paths: Optional[List[str]] = None

        # If caching is on for a remote source ==> create a local directory for the cached files and download the files
        if self.remote_source and self.__cache_files:
            # Create the local directory for the cached files
            self.__local_dir = os.path.join(os.getcwd(), ".pyThermoNDT_cache", self._create_safe_folder_name())
            if not os.path.isdir(self.__local_dir):
                os.makedirs(self.__local_dir)

            # Loop over the file list and download the files
            for file in self.files:
                file_path = os.path.join(self.__local_dir, file.split('/')[-1])
                if not os.path.isfile(file_path):
                    with open(file_path, 'wb') as f:
                        f.write(self._read_file(file).getbuffer())

            # Set the cached paths to the local file paths
            self.__cached_paths = [os.path.join(self.__local_dir, file_name) for file_name in self.file_names]

    def __str__(self):
        return "{}(parser={}, source={}, cache_paths={})".format(
            self.__class__.__name__, 
            self.parser.__name__, 
            self.__source,
            self.__cache_files
        )
    
    def __len__(self) -> int:
        return self.num_files
    
    def __getitem__(self, idx: int) -> DataContainer:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of range. Must be between 0 and {len(self)}")
        return self.read(self.files[idx])
    
    @ property
    def source(self) -> str:
        """ Returns the source of the reader."""
        return self.__source

    @property
    def parser(self) -> Type[BaseParser]:
        """ Returns the parser class that the reader uses to parse the data."""
        return self.__parser

    @property
    def file_extensions(self) -> Tuple[str, ...]:
        """ Returns the file extensions that the reader can read."""
        return self.__file_extensions
    
    @property
    def file_names(self) -> List[str]:
        """ Returns a list of all the file names that the reader can read."""
        return [path.split('/')[-1] for path in self.files]
    
    @property
    def num_files(self) -> int:
        """ Returns the number of files that the reader can read."""
        return len(self.files)
    
    @property
    def files(self) -> List[str]:
        """ Returns a list of all the paths to the files that the reader can read."""
        # If caching is on and files are already cached, return the cached files
        if self.__cache_files and self.__cached_paths is not None:
            return self.__cached_paths
        
        # If the caching is off ==> reset cached files and retrieve the file list
        elif not self.__cache_files:
            self.__cached_paths = None
            return self._get_file_list()
        
        # Else cache the files and return them
        self.__cached_paths = self._get_file_list()
        return self.__cached_paths
    
    def _sanitize_string(self, s: str) -> str:
        """Sanitizes a string by replacing non-alphanumeric characters with underscores and removing leading/trailing underscores.

        Parameters:
            s (str): The string to be sanitized.
        
        Returns:
            str: The sanitized string.
        """
        # Replace non-alphanumeric characters (except underscores) with underscores
        s = re.sub(r'[^\w\-_\. ]', '_', s)
        # Replace multiple underscores with a single underscore
        s = re.sub(r'_+', '_', s)
        # Remove leading/trailing underscores
        return s.strip('_')
    
    def _create_safe_folder_name(self) -> str:
        """Creates a safe folder name for the downloaded files.

        Used to create a folder name for the downloaded files, that is persistend and does not change between runs.

        Returns:
            str: The safe folder name.   
        """
        safe_class_name = self._sanitize_string(self.__class__.__qualname__)
        safe_source_expr = self._sanitize_string(self.source)

        # limit the folder name to 255 characters
        return f"{safe_class_name}_{safe_source_expr}"[:255]
    
    @property
    @abstractmethod
    def remote_source(self) -> bool:
        """ Returns True if the reader reads files from a remote source, False otherwise. This property must be implemented by the subclass."""
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    def _read_file(self, path: str) -> io.BytesIO:
        """ Actual implementation of how a single file is read into memory. This method must be implemented by the subclass."""
        raise NotImplementedError("Method must be implemented by subclass")
    
    @abstractmethod
    def _get_file_list(self) -> List[str]:
        """Actual implementation of how the reader gets the list of files. This method must be implemented by the subclass."""
        raise NotImplementedError("Method must be implemented by subclass")
    
    @abstractmethod
    def _close(self):
        """ Closes any open connections or resources that the reader might have opened. 
        If the reader does not open any connections or resources, this method can be passed. Must be implemented by the subclass.
        """
        raise NotImplementedError("Method must be implemented by subclass")

    def read(self, path: str) -> DataContainer:
        """Reads and parse the file at the given path into a DataContainer object using the specified parser.
        
        Parameters:
            path (str): The path to the file to be read.

        Returns:
            DataContainer: The parsed data in a DataContainer
        """
        # If the reader reads from a remote source and files are cached, read the file from the local directory
        if self.remote_source and self.__cache_files and self.__cached_paths is not None:
            with open(path, 'rb') as f:
                return self.parser.parse(io.BytesIO(f.read()))
        
        # Else read the file directly from the source
        return self.parser.parse(self._read_file(path))
    
    def clear_cache(self):
        """ Clears the cached file paths. Therefore the reader will check for new files on the next call of the files property."""
        self.__cached_paths = None