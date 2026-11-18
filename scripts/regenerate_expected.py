"""Regenerate changed expected outputs of the Azure integration data locally for manual review."""

import argparse
import sys
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from pydantic_settings import SettingsConfigDict

# Direct script execution puts scripts/, not the repository root, on the import path.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from pythermondt.data import container_diff  # noqa: E402
from pythermondt.readers import AzureBlobReader  # noqa: E402
from tests.support.integration_data import INTEGRATION_PREFIX, FilePair, load_file_pairs  # noqa: E402
from tests.support.remote_data import RemoteTestConfig  # noqa: E402
from tests.utils import format_container_diff  # noqa: E402


class _LocalTestConfig(RemoteTestConfig):
    """Load CLI settings from the repository .env, with environment variables taking priority."""

    model_config = SettingsConfigDict(
        env_file=REPOSITORY_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # The repository .env may also configure other tools.
    )


def _select_pairs(pairs: tuple[FilePair, ...], selected_path: str) -> list[FilePair]:
    """Select pairs by exact source path or folder boundary."""
    return [
        pair
        for pair in pairs
        if (source_path := f"{INTEGRATION_PREFIX}/{pair.id}") == selected_path
        or source_path.startswith(f"{selected_path}/")
    ]


def _regenerate_pair(reader: AzureBlobReader, pair: FilePair, output: Path) -> bool:
    """Save changed output locally; return True only after a successful write."""
    # Always compare against the current Azure state, unlike the test session cache.
    reader.sync(file_paths=[pair.source_uri, pair.expected_uri])
    source_container = reader.read_file(pair.source_uri)
    expected_container = reader.read_file(pair.expected_uri)
    differences = list(container_diff(expected_container, source_container))
    if not differences:
        print(f"[UNCHANGED] {pair.source_name} -> {pair.expected_name}\n", flush=True)
        return False

    output.parent.mkdir(parents=True, exist_ok=True)
    with source_container.serialize_to_hdf5() as buffer, output.open("wb") as file:
        file.write(buffer.getbuffer())
    print(
        f"[CHANGED] {pair.source_name} -> {pair.expected_name}\n"
        f"{format_container_diff(sorted(differences))}\n"
        f"  Saved: {output.name}\n",
        flush=True,
    )
    return True


def _upload_changed(config: RemoteTestConfig, changed_outputs: list[tuple[FilePair, Path]]) -> int:
    """Require confirmation before sequential uploads without automatic rollback."""
    container_uri = f"az://{config.container_name}/"
    uploads = [(pair.expected_uri.removeprefix(container_uri), output) for pair, output in changed_outputs]
    print("\nChanged files to upload:")
    for blob_name, output in uploads:
        print(f"  {output} -> az://{config.container_name}/{blob_name}")
    print("Warning: this overwrites expected outputs. Recovery depends on Azure retention/versioning settings.")
    try:
        answer = input(f"Upload these {len(uploads)} changed files? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer.strip().lower() not in {"y", "yes"}:
        print("\nUpload cancelled. Local outputs kept. Azure was not modified.")
        return 0

    uploaded = 0
    try:
        with (
            DefaultAzureCredential() as credential,
            BlobServiceClient(account_url=str(config.account_url), credential=credential) as client,
        ):
            for blob_name, output in uploads:
                with output.open("rb") as file:
                    client.get_blob_client(config.container_name, blob_name).upload_blob(file, overwrite=True)
                uploaded += 1
                print(f"[UPLOADED] az://{config.container_name}/{blob_name}", flush=True)
    except Exception as error:
        # Keep potentially sensitive SDK error messages out of terminal output.
        print(f"[FAILED] Upload ({type(error).__name__}). Stopped after {uploaded} of {len(uploads)} uploads.")
        print("Successful uploads listed above were not rolled back. Local outputs kept.")
        return 1
    print(f"Uploaded {uploaded} changed files. Local outputs kept.")
    return 0


def main() -> int:
    """Regenerate locally, optionally confirm uploads, and return a process exit code."""
    parser = argparse.ArgumentParser(
        description="Compare Azure expected outputs and save changed HDF5 files locally for review.",
        epilog="Loads TEST_* settings from the repository .env or environment, and uses Azure login.",
    )
    parser.add_argument(
        "path", help="Source file or folder relative to TEST_PREFIX, e.g. integration/edevis/source3.di or integration."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "expected-review",
        help="Local review folder (default: expected-review/ in the repository root).",
    )
    parser.add_argument("--upload", action="store_true", help="Offer to upload changed outputs after confirmation.")
    args = parser.parse_args()
    selected_path = args.path.rstrip("/")
    if (
        not selected_path
        or selected_path.startswith("/")
        or any(part in {".", "..", ""} for part in selected_path.split("/"))
    ):
        parser.error("Use a relative Azure file or folder path without '.', '..', or empty path components.")

    try:
        config = _LocalTestConfig()
        reader, pairs = load_file_pairs(config)
        selected_pairs = _select_pairs(pairs, selected_path)
    except Exception as error:
        # Azure exception messages can contain credentials or signed URLs.
        print(f"Discovery failed ({type(error).__name__}). Check TEST_* settings and Azure access.", file=sys.stderr)
        return 1
    if not selected_pairs:
        parser.error(f"No paired integration source files match {args.path!r} relative to TEST_PREFIX.")

    pair_outputs = [(pair, args.output / pair.folder / pair.expected_name) for pair in selected_pairs]
    # Remove results of earlier runs, so the folder shows only this run for the selected pairs.
    for _, output in pair_outputs:
        output.unlink(missing_ok=True)

    print(
        f"Azure: {config.account_url}\nComparison: container1 = current expected; container2 = regenerated.\n",
        flush=True,
    )
    unchanged = failed = 0
    changed_outputs: list[tuple[FilePair, Path]] = []
    current_folder = None
    for pair, output in pair_outputs:
        if pair.folder != current_folder:
            print(f"Folder: {pair.source_uri.rsplit('/', 1)[0]}/\nOutput: {output.parent}/\n", flush=True)
            current_folder = pair.folder
        try:
            if _regenerate_pair(reader, pair, output):
                changed_outputs.append((pair, output))
            else:
                unchanged += 1
        except Exception as error:
            print(
                f"[FAILED] {pair.source_name} -> {pair.expected_name}\n"
                f"  {type(error).__name__}: Check Azure access and input/output files.\n",
                flush=True,
            )
            failed += 1

    print(f"Summary: {len(changed_outputs)} changed, {unchanged} unchanged, {failed} failed.\nAzure was not modified.")
    if not args.upload:
        return 1 if failed else 0
    if failed:
        print("Upload skipped because regeneration failed. Fix the errors before updating Azure.")
        return 1
    if changed_outputs:
        return _upload_changed(config, changed_outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
