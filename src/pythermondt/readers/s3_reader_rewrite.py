from ..io.backends import S3Backend
from ..io.parsers import BaseParser
from .base_reader_rewrite import BaseReader


class S3Reader(BaseReader):
    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        cache_files: bool = False,
        parser: type[BaseParser] | None = None,
        num_files: int | None = None,
    ):
        super().__init__(S3Backend(bucket, prefix), parser, num_files)
        self.__cache_files = cache_files

    @property
    def cache_files(self) -> bool:
        return self.__cache_files
