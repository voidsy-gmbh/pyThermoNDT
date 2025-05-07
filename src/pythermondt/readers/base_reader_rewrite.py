import os
from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..data import DataContainer
from ..io.backends import BaseBackend
from ..io.parsers import BaseParser, find_parser_for_extension, get_all_supported_extensions


class BaseReader(ABC):
    @abstractmethod
    def __init__(self, backend: BaseBackend, parser: type[BaseParser] | None = None, num_files: int | None = None):
        """Initialize an instance of the BaseReader class.

        Parameters:
            backend (BaseBackend): The backend that the reader uses to read the data.
            parser (Type[BaseParser], optional): The parser that the reader uses to parse the data. If not specified,
                the parser will be auto selected based on the file extension. Default is None.
            num_files (int, optional): The number of files to read. If not specified, all files will be read.
                Default is None.
        """
        # Assign private attributes
        self.__backend = backend
        self.__parser = parser
        self.__supported_extensions = tuple(parser.supported_extensions if parser else get_all_supported_extensions())
        self.__num_files = num_files
        self.__files_cache = None

    @property
    def backend(self) -> BaseBackend:
        """The backend that the reader uses to read the data."""
        return self.__backend

    @property
    def parser(self) -> type[BaseParser] | None:
        """The parser that the reader uses to parse the data."""
        return self.__parser

    @property
    def num_files(self) -> int | None:
        """The number of files to read."""
        return self.__num_files

    @property
    def files(self) -> list[str]:
        """List of files that the reader is able to read."""
        if self.__files_cache is None:
            self.__files_cache = self.backend.get_file_list(
                extensions=self.__supported_extensions, num_files=self.num_files
            )
        return self.__files_cache

    def __str__(self):
        return (
            f"{self.__class__.__name__}(backend={self.backend.__class__.__name__}"
            f"{', parser=' + self.parser.__name__ if self.parser else ''})"
        )

    def __getitem__(self, idx: int) -> DataContainer:
        if idx < 0 or idx >= len(self.files):
            raise IndexError(f"Index out of bounds. Must be in range [0, {len(self.files)})")
        return self.read_file(self.files[idx])

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterator[DataContainer]:
        # Take a snapshot of the file list ==> to avoid undefined behavior when the file list changes during iteration
        # and caching is of
        file_paths = self.files

        for file in file_paths:
            yield self.read_file(file)

    def read_file(self, file_path: str) -> DataContainer:
        """Read a fiel from the specified path and return it as a DataContainer object.

        Args:
            file_path (str): The path to the file to be read.

        Returns:
            DataContainer: The data contained in the file, parsed and returned as a DataContainer object.

        Raises:
            ValueError: If the file type cannot be determined or if no parser is found for the file extension.
        """
        # Get raw file data from backend
        file_data = self.backend.read_file(file_path)

        # If parser was specified during initialization, use it
        if self.__parser is not None:
            return self.__parser.parse(file_data)

        # Otherwise, choose a parser based on file extension
        _, ext = os.path.splitext(file_path)
        if not ext:
            raise ValueError(f"Cannot determine file type for {file_path} - no extension found")

        # Find appropriate parser for this extension
        parser_cls = find_parser_for_extension(ext)
        if parser_cls is None:
            raise ValueError(f"No parser found for file extension: {ext}")

        # Parse the file with the selected parser
        return parser_cls.parse(file_data)
