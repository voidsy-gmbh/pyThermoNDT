import os
from glob import glob
from urllib.parse import urlparse
from urllib.request import pathname2url, url2pathname

from ..utils import IOPathWrapper
from .base_backend import BaseBackend


class LocalBackend(BaseBackend):
    def __init__(self, pattern: str, recursive: bool = False) -> None:
        """Initialize an instance of the LocalBackend class.

        This class is used to read data from local files, or directories, using the standard Python file I/O operations.

        Args:
            pattern (str): The pattern that will be used to match files to read. This can either be a file path, a
                directory path, or a glob pattern.
            recursive (bool): If True, the pattern will be applied recursively to all subdirectories. This will only
                be effective if the pattern is a directory path or a glob pattern. Defaults to False.
        """
        # Validate the pattern
        if not isinstance(pattern, str):
            raise ValueError(f"Invalid pattern type: {type(pattern)}. Must be a string.")
        if pattern == "":
            raise ValueError(f"Invalid pattern: {pattern!r}. Must be a non-empty string.")

        # Initialize the pattern either as url or path
        parsed_input = self._parse_input(pattern)

        # Determine the type of the source based on the provided pattern
        self.__source_type = None
        if os.path.isfile(parsed_input):
            self.__source_type = "file"
        elif os.path.isdir(parsed_input):
            self.__source_type = "directory"
        else:
            self.__source_type = "glob"

        # Internal state
        self.__pattern_str = parsed_input
        self.__recursive = recursive

    @property
    def remote_source(self) -> bool:
        return False

    @property
    def scheme(self) -> str:
        return "file"

    @property
    def pattern(self) -> str:
        return self.__pattern_str

    def read_file(self, file_path: str) -> IOPathWrapper:
        path = self._parse_input(file_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return IOPathWrapper(path)

    def write_file(self, data: IOPathWrapper, file_path: str) -> None:
        path = self._parse_input(file_path)
        with open(path, "wb") as file:
            file.write(data.file_obj.read())

    def exists(self, file_path: str) -> bool:
        return os.path.exists(self._parse_input(file_path))

    def close(self) -> None:
        # Nothing to close for local files
        pass

    def get_file_list(self, extensions: tuple[str, ...] | None = None, num_files: int | None = None) -> list[str]:
        # Handle different pattern types
        all_files = []
        match self.__source_type:
            case "file":
                all_files = [self.pattern]
            case "directory":
                if self.__recursive:
                    all_files = [os.path.join(root, name) for root, _, names in os.walk(self.pattern) for name in names]
                else:
                    all_files = [f.path for f in os.scandir(self.pattern) if f.is_file()]
            case "glob":
                all_files = glob(self.pattern, recursive=self.__recursive)

        # Filter by extension if provided
        if extensions:
            all_files = [f for f in all_files if any(f.endswith(ext) for ext in extensions)]

        # Sort for deterministic behavior
        all_files.sort()

        # Limit number of results if specified
        if num_files is not None:
            all_files = all_files[:num_files]

        # Convert to absolute paths and normalize before returning
        all_files = [self._to_url(os.path.normpath(os.path.abspath(f))) for f in all_files]

        return all_files

    def get_file_size(self, file_path: str) -> int:
        path = self._parse_input(file_path)
        if os.path.isdir(path):
            raise IsADirectoryError(f"Path is a directory, not a file: {path}")
        return os.path.getsize(path)

    def download_file(self, source_path: str, destination_path: str) -> None:
        raise NotImplementedError("Direct download is not supported for local files.")

    def _parse_input(self, input_path: str) -> str:
        if urlparse(input_path).scheme == self.scheme:
            return url2pathname(input_path.replace(f"{self.scheme}:", ""))
        return input_path

    def _to_url(self, internal_path: str) -> str:
        url_path = pathname2url(internal_path)
        return f"file:{url_path}" if url_path.startswith("///") else f"file://{url_path}"
