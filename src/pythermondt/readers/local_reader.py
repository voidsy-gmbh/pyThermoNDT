from re import Pattern

from ..io import BaseParser, IOPathWrapper
from ..io.backends import LocalBackend
from ..io.parsers import BaseParser
from .base_reader import BaseReader


class LocalReader(BaseReader):
    def __init__(
        self,
        pattern: Pattern | str,
        num_files: int | None = None,
        cache_files: bool = True,
        parser: type[BaseParser] | None = None,
    ):
        """Initialize an instance of the LocalReader class.

        This class is used to read data from the local file system.

        Parameters:
            source (str): The source of the data. This can either be a file path, a directory path, or a regular
                expression.
            cache_paths (bool, optional): If True, all the file paths are cached in memory. This means the reader only
                checks for new files once, so changes to the file sources will not be noticed at runtime. Default is
                True.
            parser (Type[BaseParser], optional): The parser that the reader uses to parse the data.
                If not specified, the parser will be auto selected based on the file extension. Default is None.
            num_files (int, optional): Limit the number of files that the reader can read. If None, the reader reads
                all files. Default is None.
        """
        # Check if source is a valid regex pattern
        try:
            re.compile(source)
            valid_regex = True
        except re.error:
            valid_regex = False

        # Check if the provided source is either a file, a directory or a regex pattern
        if os.path.isfile(source):
            self.__source_type = "file"

        elif os.path.isdir(source):
            self.__source_type = "directory"

        elif valid_regex:
            self.__source_type = "regex"

        else:
            raise ValueError("The provided source must either be a file, a directory or a valid regex pattern.")

        # Call the constructor of the BaseReader class
        super().__init__(source, cache_files, parser, num_files)
        # Initialize baseclass with parser
        super().__init__(num_files, False, cache_files, parser)

        # Maintain state for what is needed to create the backend
        self.__pattern = pattern

    def _create_backend(self) -> LocalBackend:
        """Create a new LocalBackend instance.

        This method is called to create or recreate the backend when needed or after unpickling.
        """
        return LocalBackend(self.__pattern)
