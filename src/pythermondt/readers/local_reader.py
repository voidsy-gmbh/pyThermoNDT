from collections.abc import Callable

from ..io import BaseParser, FileInfo, LocalBackend
from .base_reader import BaseReader


class LocalReader(BaseReader):
    def __init__(
        self,
        pattern: str,
        recursive: bool = False,
        num_files: int | None = None,
        listing_ttl: float | None = None,
        parser: type[BaseParser] | None = None,
        file_filter: Callable[[FileInfo], bool] | None = None,
    ):
        """Initialize an instance of the LocalReader class.

        Uses the LocalBackend to read files from the local file system.

        Args:
            pattern (str): The pattern that will be used to match files to read. This can either be a file path, a
                directory path, or a glob pattern.
            recursive (bool): If True, the pattern will be applied recursively to all subdirectories. This will only
                be effective if the pattern is a directory path or a glob pattern. Defaults to False.
            num_files (int, optional): The number of files to read. If not specified, all files will be read.
                Default is None.
            listing_ttl (float | None, optional): Seconds to cache the file listing. None caches indefinitely;
                0 refreshes on every access. Default is None.
            parser (Type[BaseParser], optional): The parser that the reader uses to parse the data. If not specified,
                the parser will be auto selected based on the file extension. Default is None.
            file_filter (Callable[[FileInfo], bool], optional): Metadata-aware filter applied during file
                discovery. Must be picklable whenever the reader needs to be picklable. Default: None.
        """
        # Initialize baseclass with parser
        super().__init__(num_files, False, listing_ttl, parser, file_filter)

        # Maintain state for what is needed to create the backend
        self.__pattern = pattern
        self.__recursive = recursive

    def _create_backend(self) -> LocalBackend:
        """Create a new LocalBackend instance.

        This method is called to create or recreate the backend when needed or after unpickling.
        """
        return LocalBackend(self.__pattern, self.__recursive)

    def _get_reader_params(self) -> str:
        """Get a string representation of the reader parameters used to create the backend."""
        return f"pattern={self.__pattern}"
