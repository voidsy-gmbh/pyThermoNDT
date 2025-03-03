from io import BytesIO
from typing import Protocol

from ...data import DataContainer


class BaseParser(Protocol):
    """A base class for all parsers that are used to parse data from a BytesIO object into a DataContainer object.

    All Subclasses must implement the parse method, which reads the data from the BytesIO object and returns it as a
    DataContainer object. Integrity checks and error handling should be implemented by the subclasses.
    """

    def __init__(self) -> None:
        raise TypeError("This class is static and should not be instantiated.")

    @staticmethod
    def supported_extensions() -> tuple[str, ...]:
        """Return the file extensions this parser supports.

        Returns:
            tuple[str, ...]: A tuple of strings containing the file extensions this parser supports.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def parse(data_bytes: BytesIO) -> DataContainer:
        """Parses the data from the given BytesIO object into a DataContainer object.

        Subclasses must implement this method.

        Parameters:
            data_bytes (io.BytesIO): The BytesIO object containing the data to be parsed.

        Returns:
            DataContainer: The parsed data as a DataContainer object.
        """
        raise NotImplementedError("Subclasses must implement this method.")
