import os
import tempfile
from io import BytesIO


class IOPathWrapper:
    """Provides unified access to file content via both path and buffer interfaces."""

    def __init__(self, source: str | BytesIO | bytes):
        """Provides unified access to file content via both path and buffer interfaces.

        Initialize with either a file path or file content.

        Args:
            source: Either a file path (str), file-like object, or bytes
        """
        self.__source = source
        self.__file_obj = None
        self.__temp_path = None

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
            temp = tempfile.NamedTemporaryFile(delete=False)
            self.file_obj.seek(0)
            temp.write(self.file_obj.getbuffer())
            temp.close()
            self.__temp_path = temp.name

        return self.__temp_path

    def close(self):
        """Close resources and remove temporary file."""
        # Remove temp file
        if self.__temp_path and os.path.exists(self.__temp_path):
            try:
                os.remove(self.__temp_path)
            except Exception:
                pass
            self.__temp_path = None

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        self.close()
