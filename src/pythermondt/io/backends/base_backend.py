from abc import ABC, abstractmethod

from ..utils import IOPathWrapper


class BaseBackend(ABC):
    @property
    @abstractmethod
    def remote_source(self) -> bool:
        """Determine if the source is remote."""
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abstractmethod
    def scheme(self) -> str:
        """Return the scheme of the backend (e.g., 'file', 's3')."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def read_file(self, file_path: str) -> IOPathWrapper:
        """Read a file and return its content as a IOPathWrapper object."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def write_file(self, data: IOPathWrapper, file_path: str) -> None:
        """Write a file to the specified path."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def exists(self, file_path: str) -> bool:
        """Check if a file exists."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def close(self) -> None:
        """Close the IO handler."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_file_list(self) -> list[str]:
        """Return all files under the configured pattern or prefix as unsorted URIs."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_file_size(self, file_path: str) -> int:
        """Get the size of the file at the specified file path in bytes."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_file_identity(self, file_path: str) -> str:
        """Get a backend-specific file identity for change detection.

        Returns a non-empty identity string (e.g. ETag for remote objects).
        Implementations should raise a descriptive error when identity cannot be
        determined for the given file.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def download_file(self, source_path: str, destination_path: str) -> None:
        """Directly download a file from the source to the destination path.

        This is used for remote sources to download files directly to the local filesystem.
        """
        raise NotImplementedError("Subclasses must implement this method")
