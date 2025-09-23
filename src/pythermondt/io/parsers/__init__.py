from functools import lru_cache
from importlib.metadata import entry_points

from .base_parser import BaseParser
from .edevis_parser import EdevisParser
from .hdf5_parser import HDF5Parser
from .simulation_parser import SimulationParser


def _load_parser_plugins() -> tuple[type[BaseParser], ...]:
    """Load parser plugins via entry points."""
    plugins = []
    for ep in entry_points(group="pythermondt.parsers"):
        try:
            parser_cls = ep.load()
            plugins.append(parser_cls)
        except Exception as e:  # pylint: disable=broad-except
            print(f"Warning: Failed to load parser plugin '{ep.name}': {e}")
    return tuple(plugins)


@lru_cache(maxsize=1)
def _get_registry() -> tuple[type[BaseParser], ...]:
    """
    Lazily build and cache the parser registry on first use.

    This function constructs the registry of available parser classes, including both
    built-in and plugin parsers, only when it is first called. The result is cached
    using functools.lru_cache (with maxsize=1), so subsequent calls return the same
    registry instance without rebuilding. This replaces previous module-level initialization
    and ensures that plugin discovery is performed only once per process.
    """
    builtins = (HDF5Parser, SimulationParser, EdevisParser)
    plugins = _load_parser_plugins()
    return builtins + plugins


def find_parser_for_extension(extension: str) -> type[BaseParser] | None:
    """Find a parser that supports the given file extension.

    Args:
        extension: File extension (with or without leading dot)

    Returns:
        Parser class that supports the extension, or None if not found
    """
    # Normalize extension to include leading dot
    normalized_ext = extension if extension.startswith(".") else f".{extension}"

    # Find first parser supporting this extension
    for parser_cls in _get_registry():
        if normalized_ext in parser_cls.supported_extensions:
            return parser_cls

    return None


def get_all_supported_extensions() -> set[str]:
    """Get all supported extensions of all registered parsers.

    Returns:
        Set of all supported extensions
    """
    return {ext for parser_cls in _get_registry() for ext in parser_cls.supported_extensions}


def get_all_parsers() -> tuple[type[BaseParser], ...]:
    """Get all registered parsers.

    Returns:
        Tuple of all parser classes
    """
    return _get_registry()


__all__ = [
    "BaseParser",
    "HDF5Parser",
    "SimulationParser",
    "find_parser_for_extension",
    "get_all_supported_extensions",
    "get_all_parsers",
]
