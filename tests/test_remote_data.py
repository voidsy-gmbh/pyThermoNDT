import pytest
from pydantic import ValidationError

from tests.support.remote_data import (
    CloudTestSettings,
    RemoteTestConfig,
    cloud_skip_reason,
    load_remote_config,
    remote_tests_enabled,
)

ACCOUNT_URL_ENV = "TEST_ACCOUNT_URL"
CONTAINER_ENV = "TEST_CONTAINER_NAME"
PREFIX_ENV = "TEST_PREFIX"
REMOTE_TESTS_ENV = "REMOTE_TESTS"

ACCOUNT_URL = "https://testaccount.blob.core.windows.net"


def _set_env(
    monkeypatch: pytest.MonkeyPatch,
    account_url: str | None = ACCOUNT_URL,
    container: str | None = "test-container",
    prefix: str | None = "test-data",
):
    """Set the remote test environment variables; None deletes a variable."""
    for name, value in (
        (ACCOUNT_URL_ENV, account_url),
        (CONTAINER_ENV, container),
        (PREFIX_ENV, prefix),
    ):
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
        (None, " yes ", True),
        (None, " TRUE ", True),
        (None, "0", False),
        (None, "false", False),
        (None, "", False),
        ("false", None, False),
        ("true", None, False),
        ("true", "0", False),
        ("true", "1", True),
    ],
)
def test_remote_tests_enabled(
    monkeypatch: pytest.MonkeyPatch, github_actions: str | None, remote_tests: str | None, expected: bool
):
    """Cloud tests require explicit opt-in, including in GitHub Actions."""
    if github_actions is None:
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    else:
        monkeypatch.setenv("GITHUB_ACTIONS", github_actions)
    if remote_tests is None:
        monkeypatch.delenv(REMOTE_TESTS_ENV, raising=False)
    else:
        monkeypatch.setenv(REMOTE_TESTS_ENV, remote_tests)

    assert remote_tests_enabled() is expected


@pytest.mark.parametrize("enabled", [False, True])
def test_cloud_gating_does_not_validate_azure_location(monkeypatch: pytest.MonkeyPatch, enabled: bool):
    """Cloud gating stays usable offline even with missing or invalid Azure settings."""
    _set_env(monkeypatch, account_url="not-an-account-url", container=None)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("REMOTE_TESTS", "1" if enabled else "0")

    assert CloudTestSettings().remote_tests is enabled
    assert remote_tests_enabled() is enabled
    assert (cloud_skip_reason() is None) is enabled


def test_ci_without_azure_login_skips_with_location_unset(monkeypatch: pytest.MonkeyPatch):
    """CI runs without a successful login can collect offline tests without Azure settings."""
    _set_env(monkeypatch, account_url=None, container=None)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("REMOTE_TESTS", "false")

    assert "REMOTE_TESTS=1" in cloud_skip_reason()


def test_legacy_connection_string_is_not_loaded(monkeypatch: pytest.MonkeyPatch):
    """An obsolete SAS secret cannot become part of the test configuration."""
    _set_env(monkeypatch)
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "private-token")

    for settings in (CloudTestSettings(), load_remote_config()):
        assert not hasattr(settings, "connection_string")
        assert "private-token" not in repr(settings)
        assert "connection_string" not in settings.model_dump()


def test_invalid_cloud_flags_fail_without_echoing_input(monkeypatch: pytest.MonkeyPatch):
    """Invalid boolean values receive Pydantic validation without exposing their input."""
    monkeypatch.setenv("REMOTE_TESTS", "private-token")

    with pytest.raises(ValidationError) as error:
        CloudTestSettings()

    assert "private-token" not in str(error.value)


def test_skip_reason_when_disabled(monkeypatch: pytest.MonkeyPatch):
    """Without opt-in, cloud tests skip with the opt-in instructions."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv(REMOTE_TESTS_ENV, raising=False)

    reason = cloud_skip_reason()

    assert reason is not None and REMOTE_TESTS_ENV in reason


@pytest.mark.parametrize("github_event", ["pull_request", "push"])
def test_no_skip_reason_after_explicit_opt_in(monkeypatch: pytest.MonkeyPatch, github_event: str):
    """Local Azure login works without static credentials, regardless of event context."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv(REMOTE_TESTS_ENV, "1")
    monkeypatch.setenv("GITHUB_EVENT_NAME", github_event)
    _set_env(monkeypatch)

    assert cloud_skip_reason() is None


def test_load_remote_config_reads_separate_settings(monkeypatch: pytest.MonkeyPatch):
    """Account URL, container, and prefix are loaded independently."""
    _set_env(monkeypatch)

    config = load_remote_config()

    assert str(config.account_url) == "https://testaccount.blob.core.windows.net/"
    assert config.container_name == "test-container"
    assert config.prefix == "test-data"


def test_remote_config_explicit_values_override_environment(monkeypatch: pytest.MonkeyPatch):
    """Explicit settings win over environment values without changing them."""
    _set_env(monkeypatch)

    config = RemoteTestConfig(
        account_url="https://other.blob.core.windows.net",
        container_name="other-container",
        prefix="/other-data/",
    )

    assert str(config.account_url) == "https://other.blob.core.windows.net/"
    assert config.container_name == "other-container"
    assert config.prefix == "other-data"
    assert load_remote_config().container_name == "test-container"


def test_remote_config_is_frozen(monkeypatch: pytest.MonkeyPatch):
    """The settings retain the previous dataclass's immutability."""
    _set_env(monkeypatch)
    config = load_remote_config()

    with pytest.raises(ValidationError, match="frozen"):
        config.prefix = "changed"


def test_remote_config_does_not_load_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Test location settings cannot come from an implicit .env file."""
    _set_env(monkeypatch, account_url=None)
    (tmp_path / ".env").write_text(f"TEST_ACCOUNT_URL={ACCOUNT_URL}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError, match="account_url"):
        load_remote_config()


@pytest.mark.parametrize("prefix", [None, "", "/"])
def test_load_remote_config_prefix_is_optional(monkeypatch: pytest.MonkeyPatch, prefix: str | None):
    """An unset or empty prefix selects the container root."""
    _set_env(monkeypatch, prefix=prefix)

    config = load_remote_config()

    assert config.container_name == "test-container"
    assert config.prefix == ""


def test_load_remote_config_preserves_account_and_blob_prefix(monkeypatch: pytest.MonkeyPatch):
    """Endpoint ports and blob prefix characters stay intact."""
    _set_env(monkeypatch, account_url=ACCOUNT_URL + ":8443/", prefix="/test data/measurements%20one/")

    config = load_remote_config()

    assert str(config.account_url) == ACCOUNT_URL + ":8443/"
    assert config.prefix == "test data/measurements%20one"


@pytest.mark.parametrize("suffix", ["", "/"])
def test_remote_config_accepts_https_account_endpoint(monkeypatch: pytest.MonkeyPatch, suffix: str):
    """Account endpoints work with or without a trailing slash."""
    _set_env(monkeypatch, account_url=ACCOUNT_URL + suffix)

    assert str(load_remote_config().account_url) == ACCOUNT_URL + "/"


@pytest.mark.parametrize(
    "account_url",
    [
        "not-a-url",
        "http://testaccount.blob.core.windows.net",
        "ftp://testaccount.blob.core.windows.net",
        "https://",
        "https://[invalid-host",
        ACCOUNT_URL + "/container",
        ACCOUNT_URL + "/container/blob.mat",
        ACCOUNT_URL + "?",
        ACCOUNT_URL + "#",
    ],
)
def test_remote_config_rejects_invalid_account_endpoint(monkeypatch: pytest.MonkeyPatch, account_url: str):
    """Only an HTTPS account endpoint is accepted, not a container URL or other scheme."""
    _set_env(monkeypatch, account_url=account_url)

    with pytest.raises(ValidationError, match="account_url"):
        load_remote_config()


@pytest.mark.parametrize(
    "account_url",
    [
        ACCOUNT_URL + "?sv=2024-11-04&sig=private-token",
        ACCOUNT_URL + "#private-token",
        "https://private-token@testaccount.blob.core.windows.net",
        "https://user:private-token@testaccount.blob.core.windows.net",
        "https://[private-token",
    ],
)
def test_remote_config_rejects_url_credentials_without_echoing_them(monkeypatch: pytest.MonkeyPatch, account_url: str):
    """Neither Pydantic URL errors nor custom validation errors reveal rejected secrets."""
    _set_env(monkeypatch, account_url=account_url)

    with pytest.raises(ValidationError, match="account_url") as error:
        load_remote_config()

    assert "private-token" not in str(error.value)
    assert "private-token" not in repr(error.value)


@pytest.mark.parametrize("variable, field_name", [(ACCOUNT_URL_ENV, "account_url"), (CONTAINER_ENV, "container_name")])
@pytest.mark.parametrize("value", [None, "", " "])
def test_load_remote_config_requires_account_and_container(
    monkeypatch: pytest.MonkeyPatch, variable: str, field_name: str, value: str | None
):
    """Missing required settings fail clearly without echoing other input values."""
    _set_env(monkeypatch)
    if value is None:
        monkeypatch.delenv(variable)
    else:
        monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError, match=field_name) as error:
        load_remote_config()

    assert "input_value" not in str(error.value)


def test_remote_config_uses_only_test_prefixed_location_settings(monkeypatch: pytest.MonkeyPatch):
    """Library settings and the previous container variable cannot configure cloud tests."""
    _set_env(monkeypatch, account_url=None, container=None, prefix=None)
    monkeypatch.setenv("PYTHERMONDT_ACCOUNT_URL", ACCOUNT_URL)
    monkeypatch.setenv("PYTHERMONDT_CONTAINER_NAME", "library-container")
    monkeypatch.setenv("PYTHERMONDT_PREFIX", "library-prefix")
    monkeypatch.setenv("TEST_CONTAINER", "old-container")

    with pytest.raises(ValidationError) as error:
        load_remote_config()

    assert {item["loc"] for item in error.value.errors()} == {("account_url",), ("container_name",)}

    monkeypatch.setenv("TEST_ACCOUNT_URL", ACCOUNT_URL)
    monkeypatch.setenv("TEST_CONTAINER_NAME", "test-container")
    config = load_remote_config()

    assert config.container_name == "test-container"
    assert config.prefix == ""
