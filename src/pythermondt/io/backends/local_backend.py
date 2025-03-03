from io import BytesIO
from re import Pattern

from .base_backend import BaseBackend


class LocalBackend(BaseBackend):
    def __init__(self):
        pass

    def read_file(self, file_path: str) -> BytesIO:
        pass

    def write_file(self, file_path: str) -> None:
        pass

    def exists(self, file_path: str) -> bool:
        pass

    def close(self) -> None:
        pass

    def get_file_list(self, pattern: Pattern | str, extensions: tuple[str, ...] | None = None) -> list[str]:
        pass
