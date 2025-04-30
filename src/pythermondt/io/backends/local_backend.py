import os
import re
from glob import glob
from re import Pattern

from ...utils import IOPathWrapper
from .base_backend import BaseBackend


class LocalBackend(BaseBackend):
    def __init__(self, pattern: Pattern | str) -> None:
        """Initialize an instance of the LocalBackend class.

        This class is used to read data from local files, or directories, using the standard Python file I/O operations.

        Parameters:
            pattern (Pattern | str): The source of the data. This must be a valid file path, directory path, or a regex
                pattern. If a regex pattern is provided, it will be used to determine the files using glob.
        """
        # Convert pattern to string if it is a re.Pattern object
        pattern_str = pattern.pattern if isinstance(pattern, Pattern) else pattern

        # Replace backslashes with forward slashes to avoid escaping issues on windows
        pattern_str = pattern_str.replace("\\", "/")

        # Determine the type of the source pattern
        # Check if source is a valid regex pattern
        try:
            re.compile(pattern_str)
            valid_regex = True
        except re.error:
            valid_regex = False

        # Check if the provided source is either a file, a directory or a regex pattern
        if isinstance(pattern_str, str):
            if os.path.isfile(pattern_str):
                self.__source_type = "file"
            elif os.path.isdir(pattern_str):
                self.__source_type = "directory"
            elif valid_regex:
                self.__source_type = "regex"
            else:
                raise ValueError("The provided source must either be a file, a directory or a valid regex pattern.")
        else:
            raise ValueError("The provided source must be a string or a regex pattern.")
        self.__pattern_str = pattern_str

    @property
    def remote_source(self) -> bool:
        return False

    @property
    def pattern(self) -> str:
        return self.__pattern_str

    def read_file(self, file_path: str) -> IOPathWrapper:
        return IOPathWrapper(file_path)

    def write_file(self, bytes: IOPathWrapper, file_path: str) -> None:
        with open(file_path, "wb") as file:
            file.write(bytes.file_obj.read())

    def exists(self, file_path: str) -> bool:
        return os.path.exists(file_path)

    def close(self) -> None:
        # Nothing to close for local files
        pass

    def get_file_list(self, extensions: tuple[str, ...] | None = None, num_files: int | None = None) -> list[str]:
        # Handle different pattern types
        all_files = []
        if isinstance(self.pattern, str):
            match self.__source_type:
                case "file":
                    all_files = [self.pattern]
                case "directory":
                    with os.scandir(self.pattern) as entries:
                        all_files = [entry.path for entry in entries if entry.is_file()]
                case "regex" | "pattern":
                    all_files = glob(self.pattern)
        elif isinstance(self.pattern, Pattern) and self.__source_type == "pattern":
            all_files = glob(self.pattern.pattern)
        else:
            raise ValueError("Invalid source type.")

        # Filter by extension if provided
        if extensions:
            all_files = [f for f in all_files if any(f.endswith(ext) for ext in extensions)]

        # Limit number of results if specified
        if num_files is not None:
            all_files = all_files[:num_files]

        return all_files
