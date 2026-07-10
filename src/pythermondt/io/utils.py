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

    def _init_buffer(self) -> BytesIO:
        """Lazy-init and return the internal buffer without seeking."""
        if self.__file_obj is not None:
            return self.__file_obj

        buffer: BytesIO
        if isinstance(self.__source, str):
            with open(self.__source, "rb") as f:
                buffer = BytesIO(f.read())
        elif isinstance(self.__source, bytes):
            buffer = BytesIO(self.__source)
        elif isinstance(self.__source, BytesIO):
            buffer = self.__source
        else:
            raise ValueError("Unsupported source type. Must be str, bytes, or BytesIO.")
        self.__file_obj = buffer
        return buffer

    @property
    def file_obj(self) -> BytesIO:
        """Get file-like object, loading from path if needed."""
        buffer = self._init_buffer()
        buffer.seek(0)
        return buffer

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
        return self._init_buffer().write(data)

    def read(self, size: int = -1) -> bytes:
        """Read up to size bytes from the current buffer position.

        Args:
            size: Number of bytes to read (default: -1 reads all).

        Returns:
            Bytes read from the buffer.
        """
        return self._init_buffer().read(size)

    def getvalue(self) -> bytes:
        """Return the current contents of the internal buffer as bytes."""
        return self._init_buffer().getvalue()

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        self.close()
