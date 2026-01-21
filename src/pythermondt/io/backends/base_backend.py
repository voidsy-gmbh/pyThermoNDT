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
    def get_file_list(self, extensions: tuple[str, ...] | None = None, num_files: int | None = None) -> list[str]:
        """Get a list of files matching the specified pattern/extensions."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_file_size(self, file_path: str) -> int:
        """Get the size of the file at the specified file path in bytes."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def download_file(self, source_path: str, destination_path: str) -> None:
        """Directly download a file from the source to the destination path.

        This is used for remote sources to download files directly to the local filesystem.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _parse_input(self, input_path: str) -> str:
        """Parse input path (URL or path) to internal representation.

        Args:
            input_path (str): The input path, which can be a URL or file path.

        Returns:
            str: Internal representation of the path which the backend can use.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _to_url(self, internal_path: str) -> str:
        """Convert internal representation back to unified URL format.

        Args:
            internal_path (str): The internal representation of the path.

        Returns:
            str: The unified URL format of the path including scheme.
        """
        raise NotImplementedError("Subclasses must implement this method")
