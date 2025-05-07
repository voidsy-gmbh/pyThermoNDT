from re import Pattern

from ..io.backends import LocalBackend
from ..io.parsers import BaseParser
from .base_reader_rewrite import BaseReader


class LocalReader(BaseReader):
    def __init__(
        self,
        pattern: Pattern | str,
        cache_files: bool = True,
        parser: type[BaseParser] | None = None,
        num_files: int | None = None,
    ):
        # Initialize baseclass with parser
        super().__init__(parser, num_files)

        # Maintain state for what is needed to create the backend
        self.__pattern = pattern
        self.__cache_files = cache_files

    def _create_backend(self) -> LocalBackend:
        """Create a new LocalBackend instance.

        This method is called to create or recreate the backend when needed or after unpickling.
        """
        return LocalBackend(self.__pattern)

    @property
    def cache_files(self) -> bool:
        return self.__cache_files
