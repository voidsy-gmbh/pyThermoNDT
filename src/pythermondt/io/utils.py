import logging
import os
import tempfile
from io import BytesIO
from typing import TypeAlias

logger = logging.getLogger(__name__)

IOPathSource: TypeAlias = str | BytesIO | bytes | None


class IOPathWrapper:
    """Provides unified access to file content via both path and buffer interfaces."""

    def __init__(self, source: IOPathSource = None):
        """Provides unified access to file content via both path and buffer interfaces.

        Initialize with a file path, file content, or an empty writable buffer.

        Args:
            source: A file path (str), file-like object, bytes, or None for an empty writable buffer.
        """
        self.__source: IOPathSource = BytesIO() if source is None else source
        self.__file_obj: BytesIO | None = None
        self.__temp_path: str | None = None

    @property
    def file_obj(self) -> BytesIO:
        """Get file-like object, loading from path if needed."""
        if self.__file_obj is None:
            if isinstance(self.__source, str):
                # Path provided - load file when first needed
                with open(self.__source, "rb") as f:
                    self.__file_obj = BytesIO(f.read())
            elif isinstance(self.__source, bytes):
                self.__file_obj = BytesIO(self.__source)
            elif isinstance(self.__source, BytesIO):
                # File-like object provided (from boto3 etc.)
                self.__file_obj = self.__source
            else:
                raise ValueError("Unsupported source type. Must be str, bytes, or BytesIO.")

        # Reset position and return
        self.__file_obj.seek(0)
        return self.__file_obj

    @property
    def file_path(self) -> str:
        """Get file path, using original path or creating temp file."""
        if isinstance(self.__source, str) and os.path.exists(self.__source):
            # Source is already a valid path - use directly
            return self.__source

        # Create temporary file if needed
        if not self.__temp_path or not os.path.exists(self.__temp_path):
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                self.file_obj.seek(0)
                temp.write(self.file_obj.getbuffer())
                self.__temp_path = temp.name

        return self.__temp_path

    def close(self):
        """Close resources and remove temporary file."""
        # Remove temp file
        if self.__temp_path and os.path.exists(self.__temp_path):
            try:
                os.remove(self.__temp_path)
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to remove temporary file %s: %s",
                    self.__temp_path,
                    e,
                    exc_info=True,  # Include traceback for debugging
                )
            self.__temp_path = None

    def __enter__(self) -> "IOPathWrapper":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def write(self, data: str | bytes) -> int:
        """Write data to the internal buffer.

        Args:
            data: Data to write. Strings are encoded to UTF-8 bytes.

        Returns:
            Number of bytes written.
        """
        if isinstance(data, str):
            data = data.encode()
        if self.__file_obj is None:
            _ = self.file_obj
        assert self.__file_obj is not None
        return self.__file_obj.write(data)

    def getvalue(self) -> bytes:
        """Return the current contents of the internal buffer as bytes."""
        if self.__file_obj is None:
            _ = self.file_obj
        assert self.__file_obj is not None
        return self.__file_obj.getvalue()

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        self.close()
