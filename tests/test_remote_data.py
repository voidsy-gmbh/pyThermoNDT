import pytest

from tests.support.remote_data import (
    CONNECTION_STRING_ENV,
    REMOTE_TESTS_ENV,
    TEST_DATA_URI_ENV,
    cloud_skip_reason,
    load_remote_config,
    remote_tests_enabled,
)

CONNECTION_STRING = (
    "DefaultEndpointsProtocol=https;AccountName=testaccount;AccountKey=abc==;EndpointSuffix=core.windows.net"
)
TEST_DATA_URI = "https://testaccount.blob.core.windows.net/test-container/test-data"


def _set_env(
    monkeypatch: pytest.MonkeyPatch,
    connection_string: str | None = CONNECTION_STRING,
    uri: str | None = TEST_DATA_URI,
):
    """Set the remote test environment variables; None deletes a variable."""
    for name, value in ((CONNECTION_STRING_ENV, connection_string), (TEST_DATA_URI_ENV, uri)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


@pytest.mark.parametrize(
    "github_actions, remote_tests, expected",
    [
        (None, None, False),
        (None, "1", True),
        (None, "true", True),
        (None, "on", True),
        (None, "0", False),
        (None, "false", False),
        ("true", None, True),
    ],
)
def test_remote_tests_enabled(
    monkeypatch: pytest.MonkeyPatch, github_actions: str | None, remote_tests: str | None, expected: bool
):
    """Cloud tests run in GitHub Actions or after an explicit opt-in."""
    if github_actions is None:
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    else:
        monkeypatch.setenv("GITHUB_ACTIONS", github_actions)
    if remote_tests is None:
        monkeypatch.delenv(REMOTE_TESTS_ENV, raising=False)
    else:
        monkeypatch.setenv(REMOTE_TESTS_ENV, remote_tests)

    assert remote_tests_enabled() is expected


def test_skip_reason_when_disabled(monkeypatch: pytest.MonkeyPatch):
    """Without opt-in, cloud tests skip with the opt-in instructions."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv(REMOTE_TESTS_ENV, raising=False)

    reason = cloud_skip_reason()

    assert reason is not None and REMOTE_TESTS_ENV in reason


def test_skip_reason_for_fork_pull_request(monkeypatch: pytest.MonkeyPatch):
    """Pull requests without credentials skip instead of fail."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.delenv(CONNECTION_STRING_ENV, raising=False)

    reason = cloud_skip_reason()

    assert reason is not None and CONNECTION_STRING_ENV in reason


@pytest.mark.parametrize("github_event", ["pull_request", "push"])
def test_no_skip_reason_with_enabled_and_credentials(monkeypatch: pytest.MonkeyPatch, github_event: str):
    """Enabled cloud tests with credentials do not skip."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv(REMOTE_TESTS_ENV, "1")
    monkeypatch.setenv("GITHUB_EVENT_NAME", github_event)
    _set_env(monkeypatch)

    assert cloud_skip_reason() is None


def test_load_remote_config_parses_settings(monkeypatch: pytest.MonkeyPatch):
    """Test data URI and connection string are parsed into the config fields."""
    _set_env(monkeypatch)

    config = load_remote_config()

    assert config.account_url == "https://testaccount.blob.core.windows.net"
    assert config.connection_string == CONNECTION_STRING
    assert config.container_name == "test-container"
    assert config.prefix == "test-data"


def test_load_remote_config_connection_string_is_optional(monkeypatch: pytest.MonkeyPatch):
    """Without a connection string the config falls back to DefaultAzureCredential."""
    _set_env(monkeypatch, connection_string=None)

    config = load_remote_config()

    assert config.connection_string is None
    assert config.container_name == "test-container"


def test_load_remote_config_accepts_sas_connection_string(monkeypatch: pytest.MonkeyPatch):
    """SAS connection strings reach the Azure SDK unchanged and stay out of repr."""
    from azure.storage.blob import BlobServiceClient

    connection_string = (
        "BlobEndpoint=https://testaccount.blob.core.windows.net;"
        "SharedAccessSignature=sv=2024-11-04&sr=c&sp=rl&sig=fake-signature"
    )
    _set_env(monkeypatch, connection_string=connection_string)

    config = load_remote_config()

    assert config.connection_string == connection_string
    assert "fake-signature" not in repr(config)
    assert "SharedAccessSignature" not in repr(config)
    with BlobServiceClient.from_connection_string(config.connection_string) as client:
        assert client.account_name == "testaccount"


def test_remote_config_repr_hides_account_key(monkeypatch: pytest.MonkeyPatch):
    """Config representations must not expose account-key credentials either."""
    _set_env(monkeypatch)

    assert "AccountKey" not in repr(load_remote_config())


@pytest.mark.parametrize(
    "uri",
    [
        TEST_DATA_URI + "?sig=private-token",
        TEST_DATA_URI + "#private-token",
        "https://user:private-token@testaccount.blob.core.windows.net/test-container",
        "https://[private-token/test-container",
    ],
)
def test_remote_config_rejects_credentials_in_uri_without_echoing_them(monkeypatch: pytest.MonkeyPatch, uri: str):
    """Credentials belong in the secret, never in the public data URL or errors."""
    _set_env(monkeypatch, uri=uri)

    with pytest.raises(RuntimeError, match=TEST_DATA_URI_ENV) as error:
        load_remote_config()

    assert "private-token" not in str(error.value)
    assert error.value.__cause__ is None


def test_load_remote_config_prefix_is_optional(monkeypatch: pytest.MonkeyPatch):
    """A container-only URI gives an empty prefix."""
    _set_env(monkeypatch, uri="https://testaccount.blob.core.windows.net/test-container")

    config = load_remote_config()

    assert config.container_name == "test-container"
    assert config.prefix == ""


@pytest.mark.parametrize(
    "uri",
    [
        "",
        "test-data",
        "az://test-container/test-data",
        "https://",
        "https://testaccount.blob.core.windows.net",
        "http://testaccount.blob.core.windows.net/test-container",
    ],
)
def test_load_remote_config_invalid_uri(monkeypatch: pytest.MonkeyPatch, uri: str):
    """A missing or malformed test data URI raises with the expected format."""
    _set_env(monkeypatch, uri=uri)

    with pytest.raises(RuntimeError, match=TEST_DATA_URI_ENV):
        load_remote_config()
