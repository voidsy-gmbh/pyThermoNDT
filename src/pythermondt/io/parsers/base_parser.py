import io
from abc import ABC, abstractmethod

from ...data import DataContainer


class BaseParser(ABC):
    """ A base class for all parsers that are used to parse data from a BytesIO object into a DataContainer object.

    All Subclasses must implement the parse method, which reads the data from the BytesIO object and returns it as a DataContainer object.
    Integrity checks and error handling should be implemented by the subclasses.
    """
    def __init__(self) -> None:
        raise TypeError("This class is static and should not be instantiated.")

    @staticmethod
    @abstractmethod
    def parse(data_bytes: io.BytesIO) -> DataContainer:
        """ Parses the data from the given BytesIO object, that was read using one of the BaseReaders subclasses into a DataContainer object.

        Subclasses must implement this method.

        Parameters:
            data_bytes (io.BytesIO): The BytesIO object containing the data to be parsed.

        Returns:
            DataContainer: The parsed data as a DataContainer object.
        """
        raise NotImplementedError("Subclasses must implement this method.")
