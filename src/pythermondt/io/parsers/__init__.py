import logging
from functools import lru_cache
from importlib.metadata import entry_points

from .base_parser import BaseParser
from .edevis_parser import EdevisParser
from .hdf5_parser import HDF5Parser
from .simulation_parser import SimulationParser

logger = logging.getLogger(__name__)


def _load_parser_plugins() -> tuple[type[BaseParser], ...]:
    """Load parser plugins via entry points."""
    plugins = []
    for ep in entry_points(group="pythermondt.parsers"):
        try:
            parser_cls = ep.load()
            plugins.append(parser_cls)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to load parser plugin '%s': %s", ep.name, e)
    return tuple(plugins)


@lru_cache(maxsize=1)
def _get_registry() -> tuple[type[BaseParser], ...]:
    """Lazily build and cache the parser registry on first use.

    Constructs the registry of built-in and plugin parsers only when first called.
    The result is cached so subsequent calls return the same registry instance. A tuple of parser classes is returned,
    ensuring immutability and thread safety, exactly as the previous module level CONSTANT list did.
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
