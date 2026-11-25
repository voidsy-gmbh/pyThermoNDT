import pytest

from tests.support.integration_data import IntegrationData, collect_integration_data


@pytest.fixture(scope="session")
def integration_data() -> IntegrationData:
    """Sync all paired files once in parallel; tests then read from the shared cache without freshness checks."""
    data = collect_integration_data()
    if data.error is not None or data.reader is None:
        pytest.fail(data.error or "Integration data is unavailable.", pytrace=False)
    uris = [uri for pair in data.complete_pairs for uri in (pair.source_uri, pair.expected_uri)]
    # sync() treats an empty list as "all listed files", which would also download unpaired and nested blobs.
    if uris:
        data.reader.sync(file_paths=uris)
    return data
