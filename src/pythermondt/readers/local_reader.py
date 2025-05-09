from re import Pattern

from ..io import BaseParser, IOPathWrapper
from ..io.backends import LocalBackend
from ..io.parsers import BaseParser
from .base_reader import BaseReader


class LocalReader(BaseReader):
    def __init__(
        self,
        pattern: Pattern | str,
        num_files: int | None = None,
        cache_files: bool = True,
        parser: type[BaseParser] | None = None,
    ):
        """Initialize an instance of the LocalReader class.

        Uses the LocalBackend to read files from the local file system.

        Parameters:
            pattern (Pattern | str): The pattern to match files. Can be a string or a compiled regex pattern.
            num_files (int, optional): The number of files to read. If not specified, all files will be read.
                Default is None.
            cache_files (bool, optional): Wether to cache the files list in memory. If set to False, changes to the
                detected files will be reflected at runtime. Default is True.
            parser (Type[BaseParser], optional): The parser that the reader uses to parse the data. If not specified,
                the parser will be auto selected based on the file extension. Default is None.
        """
        # Initialize baseclass with parser
        super().__init__(num_files, False, cache_files, parser)

        # Maintain state for what is needed to create the backend
        self.__pattern = pattern

    def _create_backend(self) -> LocalBackend:
        """Create a new LocalBackend instance.

        This method is called to create or recreate the backend when needed or after unpickling.
        """
        return LocalBackend(self.__pattern)

    def _get_reader_params(self) -> str:
        """Get a string representation of the reader parameters used to create the backend."""
        return f"pattern={self.__pattern}"
