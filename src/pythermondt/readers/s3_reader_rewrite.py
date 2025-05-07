from ..io.backends import S3Backend
from ..io.parsers import BaseParser
from .base_reader_rewrite import BaseReader


class S3Reader(BaseReader):
    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        parser: type[BaseParser] | None = None,
        num_files: int | None = None,
    ):
        super().__init__(S3Backend(bucket, prefix), parser, num_files)
