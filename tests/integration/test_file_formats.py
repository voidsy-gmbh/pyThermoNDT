"""Check that every sample file decodes to its expected HDF5 output."""

import pytest

from pythermondt.data import container_diff
from tests.support.integration_data import INTEGRATION_PREFIX, FilePair, IntegrationData, collect_integration_data
from tests.utils import format_container_diff

pytestmark = pytest.mark.cloud

# A placeholder keeps the test visible when cloud tests are skipped or discovery failed.
PAIRS = collect_integration_data().pairs or (None,)


@pytest.mark.parametrize("pair", PAIRS, ids=lambda pair: pair.id if pair else "unavailable")
def test_file_format_matches_expected(integration_data: IntegrationData, pair: FilePair):
    """Parse a source file and compare the container with its expected output."""
    if not pair.has_expected:
        pytest.fail(
            f"{pair.expected_name} is missing for {pair.source_uri}. Create it with: "
            f"uv run scripts/regenerate_expected.py {INTEGRATION_PREFIX}/{pair.id} --upload",
            pytrace=False,
        )
    assert integration_data.reader is not None
    source = integration_data.reader.read_file(pair.source_uri)
    expected = integration_data.reader.read_file(pair.expected_uri)

    differences = list(container_diff(expected, source))
    assert not differences, f"{pair.source_uri} vs {pair.expected_uri}:\n{format_container_diff(differences)}"
