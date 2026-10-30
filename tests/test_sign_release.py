"""Test release orchestration, not the external signing or upload services."""

import runpy
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import certifi
import pytest
import requests

SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "sign_release.py"


@dataclass
class ReleaseTools:
    """Replace network and CLI boundaries while recording the files passed between them."""

    download: MagicMock = field(default_factory=MagicMock)
    command: MagicMock = field(default_factory=MagicMock)
    signed: dict[str, bytes] = field(default_factory=dict)
    uploaded: dict[str, bytes] = field(default_factory=dict)
    omit_bundle: bool = False


@pytest.fixture
def release_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReleaseTools:
    """Provide local distributions and fake external tools; never sign or upload for real."""
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "GITHUB_REPOSITORY": "owner/project",
        "GITHUB_REF_NAME": "release/v1.0",
        "GITHUB_REF": "refs/tags/release/v1.0",
        "RUNNER_TEMP": str(tmp_path),
        "GH_TOKEN": "upload-token",
        "GITHUB_TOKEN": "other-token",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "oidc-token",
    }.items():
        monkeypatch.setenv(key, value)
    Path("dist").mkdir()
    for name, data in {"package.whl": b"wheel", "package.tar.gz": b"sdist", "unrelated.txt": b"exclude"}.items():
        (Path("dist") / name).write_bytes(data)
    tools = ReleaseTools()
    response = tools.download.return_value.__enter__.return_value
    response.iter_content.side_effect = lambda **kwargs: iter([b"source ", b"archive"])

    def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        # Inspect the trust directory while it is in use, before temporary-file cleanup.
        assert not list(Path(kwargs["env"]["SSL_CERT_DIR"]).iterdir())
        files = [Path(value) for value in command if value.endswith((".whl", ".tar.gz", ".zip", ".sigstore.json"))]
        if "sigstore" in command:
            tools.signed.update((path.name, path.read_bytes()) for path in files)
            for path in files:
                if tools.omit_bundle and path.suffix == ".zip":
                    continue
                Path(f"{path}.sigstore.json").write_bytes(b"bundle")
        else:
            tools.uploaded.update((path.name, path.read_bytes()) for path in files)
        return subprocess.CompletedProcess(command, 0)

    tools.command.side_effect = run
    monkeypatch.setattr(requests, "get", tools.download)
    monkeypatch.setattr(subprocess, "run", tools.command)
    return tools


def test_release_contents_and_credentials(release_tools: ReleaseTools) -> None:
    """Upload only intended files and bundles, with separate signing and upload credentials."""
    runpy.run_path(str(SCRIPT), run_name="__main__")

    expected = {
        "package.whl": b"wheel",
        "package.tar.gz": b"sdist",
        "release-v1.0.zip": b"source archive",
        "release-v1.0.tar.gz": b"source archive",
    }
    assert release_tools.signed == expected
    assert release_tools.uploaded == expected | {f"{name}.sigstore.json": b"bundle" for name in expected}
    assert {call.args[0] for call in release_tools.download.call_args_list} == {
        f"https://github.com/owner/project/archive/refs/tags/release/v1.0.{extension}"
        for extension in ("zip", "tar.gz")
    }
    for call in release_tools.download.call_args_list:
        assert call.kwargs["verify"] == certifi.where()
        assert call.kwargs["stream"]
        assert all(0 < timeout < float("inf") for timeout in call.kwargs["timeout"])
    signing, upload = release_tools.command.call_args_list
    assert "GH_TOKEN" not in signing.kwargs["env"]
    assert "GITHUB_TOKEN" not in signing.kwargs["env"]
    assert signing.kwargs["env"]["ACTIONS_ID_TOKEN_REQUEST_TOKEN"] == "oidc-token"
    assert upload.kwargs["env"]["GH_TOKEN"] == "upload-token"
    assert signing.kwargs["env"]["SSL_CERT_FILE"] == certifi.where()
    assert upload.kwargs["env"]["SSL_CERT_FILE"] == certifi.where()
    assert signing.kwargs["env"]["SSL_CERT_DIR"] == upload.kwargs["env"]["SSL_CERT_DIR"]
    assert not Path(upload.kwargs["env"]["SSL_CERT_DIR"]).exists()
    assert signing.kwargs["check"] and upload.kwargs["check"]


@pytest.mark.parametrize("failure", ["download", "sign", "missing-bundle"])
def test_failure_prevents_upload(release_tools: ReleaseTools, failure: str, tmp_path: Path) -> None:
    """Do not publish incomplete or unsigned assets; remove temporary files on failure."""
    expected_error: type[Exception] = FileNotFoundError
    if failure == "download":
        response = release_tools.download.return_value.__enter__.return_value
        response.raise_for_status.side_effect = requests.HTTPError("Download failed")
        expected_error = requests.HTTPError
    elif failure == "sign":
        release_tools.command.side_effect = subprocess.CalledProcessError(1, ["sigstore"])
        expected_error = subprocess.CalledProcessError
    else:
        release_tools.omit_bundle = True
    with pytest.raises(expected_error):
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert not release_tools.uploaded
    if failure == "download":
        release_tools.command.assert_not_called()
    assert not list(tmp_path.glob("release-signing-*"))


@pytest.mark.parametrize("missing", ["package.whl", "package.tar.gz"])
def test_missing_distribution_prevents_network_access(release_tools: ReleaseTools, missing: str) -> None:
    """Reject an incomplete build before downloading or publishing release assets."""
    (Path("dist") / missing).unlink()
    with pytest.raises(FileNotFoundError, match="No release distributions found"):
        runpy.run_path(str(SCRIPT), run_name="__main__")
    release_tools.download.assert_not_called()
    release_tools.command.assert_not_called()
