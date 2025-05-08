import hashlib
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..config import settings
from ..data import DataContainer
from ..io.backends import BaseBackend
from ..io.parsers import BaseParser, find_parser_for_extension, get_all_supported_extensions
from ..utils import IOPathWrapper


class BaseReader(ABC):
    @abstractmethod
    def __init__(
        self,
        num_files: int | None = None,
        download_remote_files: bool = False,
        cache_files: bool = True,
        parser: type[BaseParser] | None = None,
    ):
        """Initialize an instance of the BaseReader class.

        Parameters:
            num_files (int, optional): The number of files to read. If not specified, all files will be read.
                Default is None.
            download_remote_files (bool, optional): Wether to download remote files to local storage. Recommended to set
                to True if frequent access to the same files is needed. Default is False to avoid unnecessary downloads.
            cache_files (bool, optional): Wether to cache the files list in memory. If set to False, changes to the
                detected files will be reflected at runtime. Default is True.
            parser (Type[BaseParser], optional): The parser that the reader uses to parse the data. If not specified,
                the parser will be auto selected based on the file extension. Default is None.
        """
        # Assign private attributes
        self.__parser = parser
        self.__supported_extensions = tuple(parser.supported_extensions if parser else get_all_supported_extensions())
        self.__num_files = num_files
        self.__files = None
        self.__cache_files = cache_files
        self.__download_remote_files = download_remote_files

    @abstractmethod
    def _create_backend(self) -> BaseBackend:
        """Create a new backend instance.

        This method must be implemented by subclasses to create or
        recreate their backend when needed or after unpickling.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _get_reader_params(self) -> str:
        """Get a string representation of the reader parameters used to create the backend."""
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def remote_source(self) -> bool:
        """Return True if the reader is reading from a remote source, False otherwise."""
        return self.backend.remote_source

    @property
    def cache_files(self) -> bool:
        """Return True if the reader caches the files/file-paths, False otherwise."""
        return self.__cache_files

    @property
    def backend(self) -> BaseBackend:
        """The backend that the reader uses to read the data."""
        if not hasattr(self, "_BaseReader__backend"):
            self.__backend = self._create_backend()
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
        # If caching is disabled return the file list from the backend
        if not self.__cache_files:
            return self.backend.get_file_list(extensions=self.__supported_extensions, num_files=self.num_files)

        # If files have never been loaded, load them from the backend
        if self.__files is None:
            self.__files = self.backend.get_file_list(extensions=self.__supported_extensions, num_files=self.num_files)

        # Return the cached files list
        return self.__files

    def __getstate__(self):
        """Prepare object for pickling by removing the backend."""
        state = self.__dict__.copy()
        # Remove backend reference - will be recreated when needed
        if "_BaseReader__backend" in state:
            del state["_BaseReader__backend"]
        # Clear files cache to force reloading
        state["_BaseReader__files_cache"] = None
        return state

    def __setstate__(self, state):
        """Restore object from pickled state."""
        # Just restore the state dictionary - backend will be created
        # lazily when first accessed
        self.__dict__.update(state)

    def __str__(self):
        return (
            f"{self.__class__.__name__}({self._get_reader_params()}, num_files={self.num_files}, "
            f"download_remote_files={self.__download_remote_files}, cache_files={self.cache_files}, "
            f"parser={self.__parser.__name__ if self.__parser else None})"
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

    def _setup_cache_dir(self, reader_id: str) -> str:
        """Step up the cache directory for the reader instance.

        Parameters:
            reader_id (str): A unique identifier for the reader instance.

        Returns:
            str: The path to the cache directory.
        """
        # Create base cache dir in users home directory
        base_dir = os.path.join(os.path.expanduser("~"), ".pythermondt_cache")

        # Hash the reader ID for a consistent directory name
        dir_hash = hashlib.md5(reader_id.encode()).hexdigest()

        # Create full path
        cache_dir = os.path.join(base_dir, dir_hash, "raw")
        os.makedirs(cache_dir, exist_ok=True)

        return cache_dir

    def read_file(self, file_path: str) -> DataContainer:
        """Read a fiel from the specified path and return it as a DataContainer object.

        Args:
            file_path (str): The path to the file to be read.

        Returns:
            DataContainer: The data contained in the file, parsed and returned as a DataContainer object.

        Raises:
            ValueError: If the file type cannot be determined or if no parser is found for the file extension.
        """
        if self.remote_source and self.__download_remote_files:
            # 1. Calculate deterministic local path
            reader_id = f"{self.__class__.__name__}_{self._get_reader_params()}"
            cache_dir = self._setup_cache_dir(reader_id)
            local_filename = hashlib.md5(file_path.encode()).hexdigest() + os.path.splitext(file_path)[1]
            local_path = os.path.join(cache_dir, local_filename)

            # 2. Download if not exists
            if not os.path.exists(local_path):
                # Download file directly to disk
                self.backend.download_file(file_path, local_path)

            file_data = IOPathWrapper(local_path)

        else:
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
