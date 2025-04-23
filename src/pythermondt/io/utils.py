import os
import tempfile
from io import BytesIO


class IOPathWrapper:
    """Wraps a file-like object (e.g., BytesIO) to provide both bytes and path-like access."""

    def __init__(self, file_obj: BytesIO, original_path: str | None = None):
        """Initialize with a file-like object and optional original path.

        Args:
            file_obj: File-like object with read/seek methods
            original_path: Original file path if available and valid
        """
        self._file = file_obj
        self._original_path = original_path
        self._temp_path = None

    @property
    def file_obj(self) -> BytesIO:
        """Get file object for direct IO access."""
        self._file.seek(0)  # Reset position for consistency
        return self._file

    @property
    def path(self) -> str:
        """Get file path, creating temporary file if needed."""
        # If original path exists and is accessible, use it
        if self._original_path and os.path.exists(self._original_path):
            return self._original_path

        # Create temporary file if needed
        if not self._temp_path or not os.path.exists(self._temp_path):
            self._file.seek(0)
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(self._file.read())
            temp_file.close()
            self._temp_path = temp_file.name

        return self._temp_path

    def __del__(self):
        """Clean up temporary file if created."""
        if self._temp_path and os.path.exists(self._temp_path):
            try:
                os.remove(self._temp_path)
            except Exception:
                pass  # Best effort cleanup
