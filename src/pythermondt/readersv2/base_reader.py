import io
from abc import ABC, abstractmethod
from ..data import DataContainer
from .parsers import BaseParser

class BaseReader(ABC):
    @abstractmethod
    def __init__(self, parser: BaseParser):
        self.parser = parser

    @abstractmethod
    def _read(self) -> io.BytesIO:
        raise NotImplementedError("Method must be implemented by subclass")

    def read(self) -> DataContainer:
        return self.parser.parse(self._read())