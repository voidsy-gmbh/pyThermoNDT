from importlib.metadata import entry_points

from .base_parser import BaseParser
from .hdf5_parser import HDF5Parser
from .simulation_parser import SimulationParser


def _load_parser_plugins() -> list[type[BaseParser]]:
    """Load parser plugins via entry points."""
    plugins = []
    for ep in entry_points(group="pythermondt.parsers"):
        try:
            parser_cls = ep.load()
            plugins.append(parser_cls)
        except Exception as e:
            print(f"Failed to load parser plugin {ep.name}: {e}")
    return plugins


# Parser registry of all available parsers
PARSER_REGISTRY: list[type[BaseParser]] = [HDF5Parser, SimulationParser] + _load_parser_plugins()


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


def get_all_parsers() -> list[type[BaseParser]]:
    """Get all registered parsers.

    Returns:
        List of all parser classes
    """
    return PARSER_REGISTRY.copy()


__all__ = [
    "BaseParser",
    "HDF5Parser",
    "SimulationParser",
    "find_parser_for_extension",
    "get_all_supported_extensions",
    "get_all_parsers",
]
