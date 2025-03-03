from abc import ABC, abstractmethod

from ..io import BaseBackend, BaseParser, HDF5Parser, SimulationParser

# Define which parser should be used for which file extensions
FILE_EXTENSIONS: dict[type[BaseParser], tuple[str, ...]] = {
    HDF5Parser: (".hdf5", ".h5"),
    SimulationParser: (".mat",),
    # Add more file extensions for future parsers here
}

# Lookup table for file extensions ==> for fast validation if the file extension is supported by the parser
FILE_EXTENSIONS_LUT = {ext: parser for parser, extensions in FILE_EXTENSIONS.items() for ext in extensions}


class BaseReader(ABC):
    @abstractmethod
    def __init__(self, backend: BaseBackend, parser: BaseParser):
        self.__backend = backend
        self.__parser = parser

    @property
    @abstractmethod
    def paths_prefix(self) -> str:
        """String that is prepended to all file paths.

        Needed to distinguihs between different readers once they are combined in a single dataset.
        """
        raise NotImplementedError("The method must be implemented by the subclass!")

    @property
    def backend(self) -> BaseBackend:
        return self.__backend

    @property
    def parser(self) -> BaseParser:
        return self.__parser

    @property
    def files(self) -> list[str]:
        return self.backend.get_file_list(pattern=".*", extensions=None)

    def __str__(self):
        return f"{self.__class__.__name__}(backend={self.backend.__class__.__name__}, parser={self.parser.__name__}"

    def read_file(self, file_path: str):
        return self.parser.parse(self.backend.read_file(file_path))
