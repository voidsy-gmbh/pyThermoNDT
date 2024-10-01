import io
import re
import os
from glob import glob
from typing import List, Type
from .base_reader import BaseReader
from .parsers import BaseParser

class LocalReader(BaseReader):
    def __init__(self, parser: Type[BaseParser], source: str, cache_files: bool = True):
        """ Initliaze an instance of the LocalReader class.

        This class is used to read data from the local file system.

        Parameters:
            parser (BaseParser): The parser to be used for parsing the data.
            source (str): The source of the data. This can either be a file path, a directory path, or a regular expression.
            cache_paths (bool, optional): If True, all the file paths are cached in memory. This means the reader only checks for new files once, so changes to the file sources will not be noticed at runtime. Default is True.
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
        super().__init__(parser, source, cache_files)

    @property
    def remote_source(self) -> bool:
        return False

    def _get_file_list(self) -> List[str]:
        # Resolve the source pattern based on the source type
        match self.__source_type:
            case "file":
                file_paths = [self.source]

            case "directory":
                file_paths = glob(os.path.join(self.source, "*"))

            case "regex":
                file_paths = glob(self.source)

            case _:
                raise ValueError("Invalid source type.")
            
        # Check if the found files match the specified file extension
        file_paths = [f for f in file_paths if any(f.endswith(ext) for ext in self.file_extensions)]
        if not file_paths:
            raise ValueError("No files found. Please check the source expression and file extensions")
        
        return file_paths

    def _read_file(self, path: str) -> io.BytesIO:
        # Open file in binary mode and return it as BytesIO object
        with open(path, 'rb') as file:
            return io.BytesIO(file.read())
        
    def _close(self):
        pass  # No need to close any resources for LocalReader