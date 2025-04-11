# tests/integration/test_parsers.py
from pathlib import Path

import pytest

from pythermondt.readers import LocalReader

from ..utils import containers_equal
from .utils import IntegrationTestCase, discover_test_cases

# Get all test cases
INTEGRATION_DIR = Path(__file__).parent / ".." / "assets" / "integration"
TEST_CASES = discover_test_cases(INTEGRATION_DIR)
TEST_IDS = [case.id for case in TEST_CASES]


@pytest.mark.parametrize("test_case", TEST_CASES, ids=TEST_IDS)
def test_local_reader_integration(test_case: IntegrationTestCase):
    """Test LocalReader integration using source files and expected outputs."""
    # Create readers
    print(test_case.source_path)
    source_reader = LocalReader(test_case.source_path)
    expected_reader = LocalReader(test_case.expected_path)

    # Compare all conainers in the reader
    for i, (source_container, expected_container) in enumerate(zip(source_reader, expected_reader, strict=True)):
        # Compare containers
        assert containers_equal(source_container, expected_container), (
            f"Test case '{test_case.id}': {source_reader.files[i]} and {expected_reader.files[i]} are not equal"
        )
