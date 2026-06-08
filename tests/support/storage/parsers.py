from pythermondt.data import DataContainer
from pythermondt.io import IOPathWrapper
from pythermondt.io.parsers import BaseParser


class PlainTextParser(BaseParser):
    supported_extensions = (".test",)

    @staticmethod
    def parse(data: IOPathWrapper) -> DataContainer:
        container = DataContainer()
        container.add_group("/", "MetaData")
        container.add_attribute("/MetaData", "payload", data.file_obj.read().decode())
        return container
