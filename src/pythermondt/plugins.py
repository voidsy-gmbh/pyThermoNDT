from importlib.metadata import entry_points

from .io.parsers.base_parser import BaseParser


def load_parser_plugins() -> list[type[BaseParser]]:
    """Load parser plugins via entry points."""
    plugins = []
    for ep in entry_points(group="pythermondt.parsers"):
        try:
            parser_cls = ep.load()
            plugins.append(parser_cls)
        except Exception as e:
            print(f"Failed to load parser plugin {ep.name}: {e}")
    return plugins
