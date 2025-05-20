import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator

from tqdm.auto import tqdm

from ..config import settings
from ..data import DataContainer
from ..io.parsers import PARSER_REGISTRY, BaseParser, find_parser_for_extension
from ..io.utils import IOPathWrapper


class BaseReader(ABC):
    """Base class for all readers. This class defines the interface for all readers, subclassing this class."""

    @abstractmethod
    def __init__(
        self,
        source: str,
        cache_files: bool = True,
        parser: type[BaseParser] | None = None,
        num_files: int | None = None,
    ):
        """Constructor for the BaseReader class.

        Parameters:
            source (str): The source of the data. This can be a file path, a directory path, a regular expression.
                In case of cloud storage, this can be a URL.
            cache_files (bool, optional): If True, the reader caches the file paths. If False, the reader retrieves the
                file list each time. For cloud storage readers, this flag should also determine if the files are
                downloaded to a local directory. Default is True.
            parser (Type[BaseParser], optional): The parser that the reader uses to parse the data.
                If not specified, the parser will be auto selected based on the file extension. Default is None.
            num_files (int, optional): Limit the number of files that the reader can read.
                If None, the reader reads all files. Default is None.
        """
        # Extract file extension from the source
        ext = re.findall(r"\.[a-zA-Z0-9]+$", source)

        # Try to auto select the parser based on the file extension if no parser is specified
        if parser is None:
            # Auto select the parser based on the file extension
            parser = find_parser_for_extension(ext[0]) if len(ext) > 0 else None

            # Raise an error if no file extension is found
            if not ext:
                raise ValueError(
                    f"Could not auto select a parser for the source: {source}. "
                    f"Source does not contain a file extension."
                )

            # Try to auto select the parser based on the file extension
            parser = find_parser_for_extension(ext[0])

            if parser is None:
                raise ValueError(
                    f"Could not auto select a parser for the source: {source}. Please specify the parser manually."
                )

        # Write parser to private attribute
        self.__parser = parser

        # Set the file extensions based on what parser is used
        self.__file_extensions = self.parser.supported_extensions
        if self.parser not in PARSER_REGISTRY:
            raise ValueError(
                f"The specified Parser: {parser.__name__} is not supported by the {self.__class__.__name__} class."
            )

        # validate that the source expression does not contain an invalid file extension ==>
        #  File extensions are defined by the parser
        correct_parser = find_parser_for_extension(ext[0]) if len(ext) > 0 else self.parser

        if correct_parser is None:
            raise ValueError(
                f"The source contains an invalid file extension: '({ext[0]})'! "
                f"Use a file extensions that is supported by the {self.parser.__name__}: {self.file_extensions}"
            )
        elif correct_parser is not self.parser:
            raise ValueError(
                f"Wrong parser selected for the file extension: '({ext[0]})'! "
                f"Use the {correct_parser.__name__} for this file extension instead"
            )

        # Set args
        self.__source = source
        self.__num_files = num_files

        # Set the cache_files flag and the cached_files attribute
        self.__cache_files = cache_files
        self.__cached_paths: list[str] | None = None

        # If caching is on for a remote source ==> create a local directory for the cached files and download the files
        if self.remote_source and self.__cache_files:
            # Download the files to the cache
            self._download_files_to_cache(self.files)

    def _download_files_to_cache(self, files: list[str]):
        # Extract the file names from the files provided
        file_names = [os.path.basename(file) for file in files]

        # Create the local directory for the cached files
        dir = settings.download_dir
        expanded = os.path.expanduser(dir)
        absolute = os.path.abspath(expanded)

        self.__local_dir = os.path.join(absolute, ".pyThermoNDT_cache", self._create_safe_folder_name())
        if not os.path.isdir(self.__local_dir):
            os.makedirs(self.__local_dir)

        # Collect the list of files that need to be downloaded
        files_to_download = []
        for file in files:
            cached_path = os.path.join(self.__local_dir, os.path.basename(file))
            if not os.path.isfile(cached_path):
                files_to_download.append((cached_path, file))

        # Only proceed if there are files to download
        if files_to_download:
            # Define custom widgets and the progress bar
            bar = tqdm(
                total=len(files_to_download),
                desc=f"Downloading Files for {self.__repr__()}",
                unit="file" if len(files_to_download) == 1 else "files",
                leave=True,  # Set to False if you don't want the bar to persist after completion
            )

            # Download the files
            with bar:
                for cached_path, file in files_to_download:
                    try:
                        with open(cached_path, "wb") as f:
                            f.write(self._read_file(file).file_obj.getbuffer())
                    except Exception as e:
                        print(f"Error downloading file: {file} - {e}")
                    finally:
                        bar.update(1)

        # Set the cached paths to the local file paths
        self.__cached_paths = [os.path.join(self.__local_dir, file_name) for file_name in file_names]

    def __str__(self):
        return (
            f"{self.__class__.__name__}(parser={self.parser.__name__}, "
            f"source={self.__source}, cache_paths={self.__cache_files})"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(source={self.source})"

    def __len__(self) -> int:
        """Returns the number of files that the reader can read."""
        return self.num_files

    def __getitem__(self, idx: int) -> DataContainer:
        """Returns the parsed data in a DataContainer object at file path at the given index."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of range. Must be between 0 and {len(self)}")
        return self.read(self.files[idx])

    def __iter__(self) -> Iterator[DataContainer]:
        """Creates an iterator that reads and parses all the files in the reader.

        In case caching is disabled, a snapshot of the file list is taken to avoid undefined behavior when the file list
        changes during iteration. For maximum performance, its is recommend to enable caching when iterating over the
        files.

        Returns:
            Iterator[DataContainer]: An iterator that yields the parsed data in DataContainer objects.
        """
        # Take a snapshot of the file list ==> to avoid undefined behavior when the file list changes during iteration
        # and caching is of
        file_paths = self.files

        for file in file_paths:
            yield self.read(file)

    def __next__(self) -> DataContainer:
        return next(iter(self))

    @property
    def source(self) -> str:
        """Returns the source of the reader."""
        return self.__source

    @property
    def parser(self) -> type[BaseParser]:
        """Returns the parser class that the reader uses to parse the data."""
        return self.__parser

    @property
    def cache_files(self) -> bool:
        """Returns True if the reader caches the file paths, False otherwise."""
        return self.__cache_files

    @property
    def file_extensions(self) -> tuple[str, ...]:
        """Returns the file extensions that the reader can read."""
        return self.__file_extensions

    @property
    def file_names(self) -> list[str]:
        """Returns a list of all the file names that the reader can read."""
        return [os.path.basename(path) for path in self.files]

    @property
    def num_files(self) -> int:
        """Returns the number of files that the reader can read."""
        return len(self.files)

    @property
    def files(self) -> list[str]:
        """Returns a list of all the paths to the files that the reader can read."""
        # If caching is off, return the file list directly
        if not self.__cache_files:
            return self._get_file_list(num_files=self.__num_files)

        # If caching is on and files are not cached, cache the files and return them
        if self.__cached_paths is None:
            self.__cached_paths = self._get_file_list(num_files=self.__num_files)

        # Else return the cached files
        return self.__cached_paths

    def _sanitize_string(self, s: str) -> str:
        """Sanitizes a given string to be used as a folder name.

        The string is processed by:
        * replacing non-alphanumeric characters with underscores
        * removing leading/trailing underscores.

        Parameters:
            s (str): The string to be sanitized.

        Returns:
            str: The sanitized string.
        """
        # Replace non-alphanumeric characters (except underscores) with underscores
        s = re.sub(r"[^\w\-_\. ]", "_", s)
        # Replace multiple underscores with a single underscore
        s = re.sub(r"_+", "_", s)
        # Remove leading/trailing underscores
        return s.strip("_")

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
        """Returns True if the reader reads files from a remote source, False otherwise.

        This property must be implemented by the subclass.
        """
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    def _read_file(self, path: str) -> IOPathWrapper:
        """Actual implementation of how a single file is read into memory.

        This method must be implemented by the subclass.
        """
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    def _get_file_list(self, num_files: int | None = None) -> list[str]:
        """Actual implementation of how the reader gets the list of files.

        This method must be implemented by the subclass.

        Parameters:
            num_files (int, optional): Limit the number of files that the reader can read. If None, the reader reads
                all files. Default is None.
        """
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    def _close(self):
        """Closes any open connections or resources that the reader might have opened.

        If the reader does not open any connections or resources, this method can be passed.
        Must be implemented by the subclass.
        """
        raise NotImplementedError("Method must be implemented by subclass")

    def read(self, path: str) -> DataContainer:
        """Reads and parse the file at the given path into a DataContainer object using the specified parser.

        Parameters:
            path (str): The path to the file to be read.

        Returns:
            DataContainer: The parsed data in a DataContainer

        Raises:
            FileNotFoundError: If the file is not found in the cached files. Clear the cache and try again.
            Exception: If an error occurs while reading the file.
        """
        try:
            # If the reader reads from a remote source and files are cached, read the file from the local directory
            if self.remote_source and self.__cache_files and self.__cached_paths is not None:
                return self.parser.parse(IOPathWrapper(path))
        except FileNotFoundError:
            raise FileNotFoundError("File not found in cached files. Clear the cache and try again.") from None

        # Else read the file directly from the source
        try:
            return self.parser.parse(self._read_file(path))
        except Exception as e:
            raise Exception(f"Error reading file: {path} - {e}") from e

    def clear_cache(self):
        """Clears the cached file paths.

        Therefore the reader will check for new files on the next call of the files property.
        """
        # Clear cached paths
        self.__cached_paths = None

        # Delete the local directory if it exists
        if self.remote_source and self.__cache_files and os.path.isdir(self.__local_dir):
            for file in os.listdir(self.__local_dir):
                os.remove(os.path.join(self.__local_dir, file))
            os.rmdir(self.__local_dir)

    def rebuild_cache(self):
        """Rebuilds the cache by first clearing the cache and then downloading the files again."""
        # Clear the cache
        self.clear_cache()

        # Rebuild the cache
        if self.remote_source and self.__cache_files:
            self._download_files_to_cache(self.files)
