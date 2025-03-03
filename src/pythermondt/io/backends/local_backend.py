import os
import re
from glob import glob
from io import BytesIO
from re import Pattern

from .base_backend import BaseBackend


class LocalBackend(BaseBackend):
    def __init__(self, pattern: Pattern | str) -> None:
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
        super().__init__(pattern_str)

    @property
    def remote_source(self) -> bool:
        return False

    def read_file(self, file_path: str) -> BytesIO:
        with open(file_path, "rb") as file:
            return BytesIO(file.read())

    def write_file(self, bytes: BytesIO, file_path: str) -> None:
        with open(file_path, "wb") as file:
            file.write(bytes.read())

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
