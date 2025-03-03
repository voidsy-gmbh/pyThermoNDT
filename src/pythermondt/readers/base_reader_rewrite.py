import re
from abc import ABC, abstractmethod

from ..io.backends import BaseBackend
from ..io.parsers import BaseParser, find_parser_for_extension


class BaseReader(ABC):
    @abstractmethod
    def __init__(self, backend: BaseBackend, parser: type[BaseParser] | None = None):
        # Extract file extension from the source
        source = backend.pattern if isinstance(backend.pattern, str) else backend.pattern.pattern
        ext = re.findall(r"\.[a-zA-Z0-9]+$", source)

        # Try to auto select the parser based on the file extension if no parser is specified
        if parser is None:
            # Auto select the parser based on the file extension
            parser = find_parser_for_extension(ext[0]) if len(ext) > 0 else None

            # Raise an error if no file extension is found
            if not ext:
                raise ValueError(
                    f"Could not auto select a parser for the source: {source}. "
                    f"Source does not contain a file extension."
                )

            # Try to auto select the parser based on the file extension
            parser = find_parser_for_extension(ext[0])

            if parser is None:
                raise ValueError(
                    f"Could not auto select a parser for the source: {source}. Please specify the parser manually."
                )

        # validate that the source expression does not contain an invalid file extension ==>
        #  File extensions are defined by the parser
        correct_parser = find_parser_for_extension(ext[0]) if len(ext) > 0 else parser

        if correct_parser is None:
            raise ValueError(
                f"The source contains an invalid file extension: '({ext[0]})'! "
                f"Use a file extensions that is supported by the {self.parser.__name__}: "
                f"{parser.supported_extensions}"
            )
        elif correct_parser is not parser:
            raise ValueError(
                f"Wrong parser selected for the file extension: '({ext[0]})'! "
                f"Use the {correct_parser.__name__} for this file extension instead"
            )

        # Assign private attributes
        self.__backend = backend
        self.__parser = parser

    @property
    @abstractmethod
    def paths_prefix(self) -> str:
        """String that is prepended to all file paths.

        Needed to distinguihs between different readers once they are combined in a single dataset.
        """
        raise NotImplementedError("The method must be implemented by the subclass!")

    @property
    def backend(self) -> BaseBackend:
        return self.__backend

    @property
    def parser(self) -> type[BaseParser]:
        return self.__parser

    @property
    def files(self) -> list[str]:
        return self.backend.get_file_list(extensions=self.parser.supported_extensions)

    def __str__(self):
        return f"{self.__class__.__name__}(backend={self.backend.__class__.__name__}, parser={self.parser.__name__}"

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.files):
            raise IndexError(f"Index out of bounds. Must be in range [0, {len(self.files)})")
        return self.read_file(self.files[idx])

    def read_file(self, file_path: str):
        return self.parser.parse(self.backend.read_file(file_path))
