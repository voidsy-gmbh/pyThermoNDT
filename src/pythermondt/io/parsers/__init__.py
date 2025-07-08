from ...plugins import load_parser_plugins
from .base_parser import BaseParser
from .hdf5_parser import HDF5Parser
from .simulation_parser import SimulationParser

# Parser registry of all available parsers
PARSER_REGISTRY = [HDF5Parser, SimulationParser] + load_parser_plugins()


def find_parser_for_extension(extension: str) -> type[BaseParser] | None:
    """Find a parser that supports the given file extension.

    Parameters:
        extension: File extension (with or without leading dot)

    Returns:
        Parser class that supports the extension, or None if not found
    """
    # Normalize extension to include leading dot
    normalized_ext = extension if extension.startswith(".") else f".{extension}"

    # Find first parser supporting this extension
    for parser_cls in PARSER_REGISTRY:
        if normalized_ext in parser_cls.supported_extensions:
            return parser_cls

    return None


def get_all_supported_extensions() -> set[str]:
    """Get all supported extensions of all registered parsers.

    Returns:
        Set of all supported extensions
    """
    return {ext for parser_cls in PARSER_REGISTRY for ext in parser_cls.supported_extensions}


__all__ = [
    "BaseParser",
    "HDF5Parser",
    "SimulationParser",
    "find_parser_for_extension",
    "get_all_supported_extensions",
]
