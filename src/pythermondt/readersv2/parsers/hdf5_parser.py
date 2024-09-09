import io
from .base_parser import BaseParser

class HDF5Parser(BaseParser):
    def __init__(self):
        pass

    def parse(self, data_bytes: io.BytesIO):
        raise NotImplementedError