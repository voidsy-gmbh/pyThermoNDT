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
        super().__init__(LocalBackend(pattern), parser, num_files)
        self.__cache_files = cache_files

    @property
    def paths_prefix(self) -> str:
        return "local"

    @property
    def cache_files(self) -> bool:
        return self.__cache_files
