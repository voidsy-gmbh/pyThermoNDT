import io
from .base_reader import BaseReader
from .parsers import BaseParser
from ..data import DataContainer


class AWSReader(BaseReader):
    def __init__(self, parser: BaseParser):
        pass

    def _read(self) -> io.BytesIO:
        raise NotImplementedError