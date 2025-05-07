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
        # Initialize baseclass with parser
        super().__init__(parser, num_files)

        # Maintain state for what is needed to create the backend
        self.__bucket = bucket
        self.__prefix = prefix
        self.__cache_files = cache_files

    def _create_backend(self) -> S3Backend:
        """Create a new S3Backend instance.

        This method is called to create or recreate the backend when needed or after unpickling.
        """
        return S3Backend(self.__bucket, self.__prefix)

    @property
    def cache_files(self) -> bool:
        return self.__cache_files
