from abc import ABC, abstractmethod

from ..utils import IOPathWrapper


class BaseBackend(ABC):
    @property
    @abstractmethod
    def remote_source(self) -> bool:
        """Determine if the source is remote."""
        # Property to determine if the source is remote
        raise NotImplementedError("The method must be implemented by the subclass!")

    @abstractmethod
    def read_file(self, file_path: str) -> IOPathWrapper:
        """Read a file and return its content as a IOPathWrapper object."""
        # Actual implementation of how to read one file
        raise NotImplementedError("The method must be implemented by the subclass!")

    @abstractmethod
    def write_file(self, data: IOPathWrapper, file_path: str) -> None:
        """Write a file to the specified path."""
        # Actual implementation of how to write one file
        raise NotImplementedError("The method must be implemented by the subclass!")

    @abstractmethod
    def exists(self, file_path: str) -> bool:
        """Check if a file exists."""
        # Check if a file exists
        raise NotImplementedError("The method must be implemented by the subclass!")

    @abstractmethod
    def close(self) -> None:
        """Close the IO handler."""
        # Close the IO handler if necessary
        raise NotImplementedError("The method must be implemented by the subclass!")

    @abstractmethod
    def get_file_list(self, extensions: tuple[str, ...] | None = None, num_files: int | None = None) -> list[str]:
        """Get a list of files matching the specified pattern/extensions."""
        # Get list of files matching pattern/extensions
        # This centralizes file discovery logic
        raise NotImplementedError("The method must be implemented by the subclass!")

    @abstractmethod
    def get_file_size(self, file_path: str) -> int:
        """Get the size of the file at the specified file path."""
        raise NotImplementedError("The method must be implemented by the subclass!")

    @abstractmethod
    def download_file(self, source_path: str, destination_path: str) -> None:
        """Directly download a file from the source to the destination path.

        This is used for remote sources to download files directly to the local filesystem.
        """
        raise NotImplementedError("The method must be implemented by the subclass!")
