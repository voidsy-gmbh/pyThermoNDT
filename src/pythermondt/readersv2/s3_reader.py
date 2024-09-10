import io
from .base_reader import BaseReader
from .parsers import BaseParser
from ..data import DataContainer


class S3Reader(BaseReader):
    def __init__(self, parser: BaseParser):
        pass

    def _read(self, path: str) -> io.BytesIO:
        raise NotImplementedError