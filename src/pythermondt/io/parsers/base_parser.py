from typing import Protocol, runtime_checkable

from ...data import DataContainer
from ..utils import IOPathWrapper


@runtime_checkable
class BaseParser(Protocol):
    """A base class for all parsers that are used to parse data from a BytesIO object into a DataContainer object.

    All Subclasses must implement the parse method, which reads the data from the BytesIO object and returns it as a
    DataContainer object. Integrity checks and error handling should be implemented by the subclasses.
    """

    supported_extensions: tuple[str, ...] = ("",)

    def __init__(self) -> None:
        raise TypeError("This class is static and should not be instantiated.")

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # Skip validation for ParserBase itself
        if cls.__name__ == "ParserBase":
            return

        # Check if supported_extensions is defined and not empty
        if (
            not hasattr(cls, "supported_extensions")
            or cls.supported_extensions == ("",)
            or not isinstance(cls.supported_extensions, tuple)
        ):
            raise TypeError(
                f"Parser class '{cls.__name__}' must define a classvariable called 'supported_extensions' "
                "as a tuple with at least one file extension in it"
            )

    @staticmethod
    def parse(data: IOPathWrapper) -> DataContainer:
        """Parses the data from the given IOPathWrapper object into a DataContainer object.

        Subclasses must implement this method.

        Args:
            data (IOPathWrapper): IOPathWrapper object containing the data to be parsed.

        Returns:
            DataContainer: The parsed data as a DataContainer object.
        """
        raise NotImplementedError("Subclasses must implement this method.")
