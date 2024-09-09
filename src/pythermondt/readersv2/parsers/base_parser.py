import io
from abc import ABC, abstractmethod
from ...data import DataContainer

class BaseParser(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def parse(self, data_bytes: io.BytesIO) -> DataContainer:
        pass