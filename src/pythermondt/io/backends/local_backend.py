import logging
import os
from datetime import datetime, timezone
from glob import glob
from urllib.parse import urlparse
from urllib.request import pathname2url, url2pathname

from ..utils import IOPathWrapper
from .base_backend import BaseBackend, FileInfo

logger = logging.getLogger(__name__)


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
        logger.debug("Source type determined as: %s", self.__source_type)

        # Internal state
        self.__pattern_str = parsed_input
        self.__recursive = recursive
        logger.debug("LocalBackend(pattern=%s, recursive=%s) initialized.", pattern, recursive)

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

    def _get_raw_file_paths(self) -> list[str]:
        """Return raw (un-normalized) file paths matching the configured source type."""
        # Handle different pattern types ==> return [] on no match
        match self.__source_type:
            case "file":
                return [self.pattern]
            case "directory":
                if self.__recursive:
                    return [os.path.join(root, name) for root, _, names in os.walk(self.pattern) for name in names]
                return [f.path for f in os.scandir(self.pattern) if f.is_file()]
            case "glob":
                return glob(self.pattern, recursive=self.__recursive)
        return []

    def get_file_list(self) -> list[str]:
        # Normalize and convert to URLs
        return [self._to_url(os.path.normpath(os.path.abspath(f))) for f in self._get_raw_file_paths()]

    def get_file_list_with_metadata(self) -> list[FileInfo]:
        """Return all files with metadata (single stat per file, no extra overhead)."""
        # Build FileInfo object for each file in raw file paths
        return [self._build_file_info(f) for f in self._get_raw_file_paths()]

    def _identity_from_stat(self, stat_result: os.stat_result) -> str:
        """Build a local-file identity string from a stat result."""
        device = getattr(stat_result, "st_dev", 0)
        inode = getattr(stat_result, "st_ino", 0)
        return f"{device}:{inode}:{stat_result.st_size}:{stat_result.st_mtime_ns}"

    def _build_file_info(self, internal_path: str) -> FileInfo:
        """Stat a local path and build a ``FileInfo`` entry (normalized, URL-encoded)."""
        normalised = os.path.normpath(os.path.abspath(internal_path))
        stat_result = os.stat(normalised)
        return FileInfo(
            path=self._to_url(normalised),
            last_modified=datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc),
            size=stat_result.st_size,
            file_identity=self._identity_from_stat(stat_result),
        )

    def get_file_size(self, file_path: str) -> int:
        path = self._parse_input(file_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if os.path.isdir(path):
            raise IsADirectoryError(f"Path is a directory, not a file: {path}")
        return os.path.getsize(path)

    def get_file_identity(self, file_path: str) -> str:
        """Return a low-overhead identity string for local files.

        The identity is based on filesystem metadata and intended for change
        detection, not cryptographic integrity verification.
        """
        path = self._parse_input(file_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if os.path.isdir(path):
            raise IsADirectoryError(f"Path is a directory, not a file: {path}")

        stat_result = os.stat(path)
        return self._identity_from_stat(stat_result)

    def download_file(self, source_path: str, destination_path: str) -> None:
        raise NotImplementedError("Direct download is not supported for local files.")

    def _parse_input(self, input_path: str) -> str:
        parsed = urlparse(input_path)
        return url2pathname(parsed.path) if parsed.scheme == self.scheme else input_path

    def _to_url(self, internal_path: str) -> str:
        url_path = pathname2url(internal_path)
        return f"file:{url_path}" if url_path.startswith("///") else f"file://{url_path}"
