import io
from typing import List, Tuple, Type, Dict
from abc import ABC, abstractmethod
from ..data import DataContainer
from .parsers import BaseParser, HDF5Parser, SimulationParser

FILE_EXTENSIONS: Dict[Type[BaseParser], Tuple[str, ...]] = {
    HDF5Parser: ('.hdf5', '.h5'),
    SimulationParser: ('.mat',),
    # Add more file extensions for future parsers here
}

class BaseReader(ABC):
    """ Base class for all readers. This class defines the interface for all readers, subclassing this class.
    """
    @abstractmethod
    def __init__(self, parser: Type[BaseParser]):
        """ Constructor for the BaseReader class.

        Parameters:
            parser (Type[BaseParser]): The parser that the reader uses to parse the data.
        """
        # Set the file extensions based on what parser is used
        try:
            self.__file_extensions = FILE_EXTENSIONS[parser]
        except KeyError:
            raise ValueError(f"The specified Parser: {parser.__name__} is not supported by the {self.__class__.__name__} class.")
        self.__parser = parser

    def __str__(self):
        return f"{self.__class__.__name__}(parser={self.parser.__name__}):\n" + "\n".join(self.file_names) + "\n"

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
    @abstractmethod
    def files(self) -> List[str]:
        """ Returns a list of all the paths to the files that the reader can read."""
        raise NotImplementedError("Property must be implemented by subclass")

    @abstractmethod
    def _read(self, path: str) -> io.BytesIO:
        """
        Actual implementation of how a single file is read. This method must be implemented by the subclass.
        """
        raise NotImplementedError("Method must be implemented by subclass")

    def read(self, path: str) -> DataContainer:
        return self.parser.parse(self._read(path))