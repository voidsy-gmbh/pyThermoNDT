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

    # Wrong parse method signature (will still pass runtime checking)
    class WrongSignatureParser:
        supported_extensions = (".test",)

        @staticmethod
        def parse(wrong_param: str) -> str:
            return "wrong"

    # Check protocol conformance
    valid = ValidParser()
    incomplete = IncompleteParser()
    wrong_sig = WrongSignatureParser()

    assert isinstance(valid, BaseParser)
    assert not isinstance(incomplete, BaseParser)

    # This will actually be True - runtime checking only verifies method names
    assert isinstance(wrong_sig, BaseParser)

    # Note: Real signature validation would happen during actual usage when
    # calling the method with real parameters, not during isinstance checks
