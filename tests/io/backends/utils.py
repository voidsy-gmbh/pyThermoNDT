from dataclasses import dataclass

from pythermondt.io import BaseBackend


@dataclass
class TestConfig:
    """Configuration for backend testing."""

    backend_cls: type[BaseBackend]
    is_remote: bool
