import io

import pytest

from pythermondt.data import DataContainer
from pythermondt.io.parsers import BaseParser


def test_init_subclass_validation():
    """Test that subclasses must define supported_extensions."""
    # Test missing supported_extensions
    with pytest.raises(TypeError, match="must define a classvariable called 'supported_extensions'"):
        type("InvalidParser", (BaseParser,), {})

    # Test empty supported_extensions
    with pytest.raises(TypeError, match="must define a classvariable called 'supported_extensions'"):
        type("InvalidParser", (BaseParser,), {"supported_extensions": ("",)})

    # Test non-tuple supported_extensions
    with pytest.raises(TypeError, match="must define a classvariable called 'supported_extensions'"):
        type("InvalidParser", (BaseParser,), {"supported_extensions": [".txt"]})


def test_protocol_conformance():
    """Test proper Protocol conformance checking."""

    # Valid implementation with required attributes and methods
    class ValidParser:
        supported_extensions = (".test",)

        @staticmethod
        def parse(data_bytes: io.BytesIO) -> DataContainer:
            return DataContainer()

    # Missing parse method
    class IncompleteParser:
        supported_extensions = (".test",)

    # Check protocol conformance
    valid = ValidParser()
    incomplete = IncompleteParser()

    # Check if valid parser is an instance of BaseParser
    assert isinstance(valid, BaseParser)

    # Check if incomplete parser is not an instance of BaseParser
    assert not isinstance(incomplete, BaseParser)
