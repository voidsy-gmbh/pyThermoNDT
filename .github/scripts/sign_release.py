"""Sign and upload release assets.

Artifact coverage follows https://github.com/sigstore/gh-action-sigstore-python.
"""

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import certifi
import requests


def main() -> None:
    """Attach signed distributions and repository snapshots to the published GitHub release.

    Run from the repository root after downloading the wheel and sdist into ``dist/``.
    GitHub Actions supplies the release ref, repository, temporary directory, and OIDC identity.
    Nothing is uploaded unless signing succeeds and every signature bundle exists.
    """
    repository = os.environ["GITHUB_REPOSITORY"]
    tag = os.environ["GITHUB_REF_NAME"]
    ref = os.environ["GITHUB_REF"]
    artifacts: list[Path] = []
    for pattern in ("*.whl", "*.tar.gz"):
        distributions = sorted(Path("dist").glob(pattern))
        if not distributions:
            raise FileNotFoundError(f"No release distributions found for dist/{pattern}")
        for distribution in distributions:
            if not distribution.is_file():
                raise FileNotFoundError(f"Release distribution is not a file: {distribution}")
        artifacts.extend(distributions)

    # Sigstore uses OIDC; it does not need the token that can modify release assets.
    signing_environment = os.environ.copy()
    for token in ("GH_TOKEN", "GITHUB_TOKEN"):
        signing_environment.pop(token, None)

    with TemporaryDirectory(prefix="release-signing-", dir=os.environ["RUNNER_TEMP"]) as directory:
        release_directory = Path(directory)
        ca_directory = release_directory / "ca"
        ca_directory.mkdir()
        # Constrain OpenSSL and Go clients to patched certifi, excluding system CA directories.
        ca_environment = {"SSL_CERT_FILE": certifi.where(), "SSL_CERT_DIR": str(ca_directory)}
        signing_environment.update(ca_environment)
        github_environment = os.environ | ca_environment

        archive_name = tag.replace("/", "-")
        for extension in ("zip", "tar.gz"):
            archive = release_directory / f"{archive_name}.{extension}"
            url = f"https://github.com/{repository}/archive/{ref}.{extension}"
            with requests.get(url, stream=True, timeout=(10, 60), verify=certifi.where()) as response:
                response.raise_for_status()
                with archive.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        output.write(chunk)
            artifacts.append(archive)

        # Keep Rekor v1 compatibility with the action this script replaces.
        subprocess.run(
            [sys.executable, "-m", "sigstore", "sign", "--rekor-version", "1", *map(str, artifacts)],
            check=True,
            env=signing_environment,
        )
        bundles = [Path(f"{artifact}.sigstore.json") for artifact in artifacts]
        for bundle in bundles:
            if not bundle.is_file():
                raise FileNotFoundError(f"Sigstore did not produce a signature bundle: {bundle}")
        # Upload the signed copies, not GitHub's regenerable source archive links.
        subprocess.run(
            [
                "gh",
                "release",
                "upload",
                "--repo",
                repository,
                "--clobber",
                "--",
                tag,
                *map(str, artifacts + bundles),
            ],
            check=True,
            env=github_environment,
        )


if __name__ == "__main__":
    main()
