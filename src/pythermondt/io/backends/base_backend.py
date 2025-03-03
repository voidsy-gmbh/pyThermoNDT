from abc import ABC, abstractmethod
from io import BytesIO
from re import Pattern


class BaseBackend(ABC):
    def __init__(self, pattern: Pattern | str) -> None:
        self.__pattern = pattern

    @property
    @abstractmethod
    def remote_source(self) -> bool:
        # Property to determine if the source is remote
        raise NotImplementedError("The method must be implemented by the subclass!")

    @property
    def pattern(self) -> Pattern | str:
        return self.__pattern

    @abstractmethod
    def read_file(self, file_path: str) -> BytesIO:
        # Actual implementation of how to read one file
        raise NotImplementedError("The method must be implemented by the subclass!")

    @abstractmethod
    def write_file(self, file_path: str) -> None:
        # Actual implementation of how to write one file
        raise NotImplementedError("The method must be implemented by the subclass!")

    @abstractmethod
    def exists(self, file_path: str) -> bool:
        # Check if a file exists
        raise NotImplementedError("The method must be implemented by the subclass!")

    @abstractmethod
    def close(self) -> None:
        # Close the IO handler if necessary
        raise NotImplementedError("The method must be implemented by the subclass!")

    @abstractmethod
    def get_file_list(self, extensions: tuple[str, ...] | None = None, num_files: int | None = None) -> list[str]:
        # Get list of files matching pattern/extensions
        # This centralizes file discovery logic
        raise NotImplementedError("The method must be implemented by the subclass!")
