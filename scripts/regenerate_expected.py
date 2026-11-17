"""Regenerate changed Azure integration baselines locally for manual review."""

import argparse
import sys
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from pydantic_settings import SettingsConfigDict

# Direct script execution puts scripts/, not the repository root, on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pythermondt.data import container_diff
from tests.integration.utils import (
    INTEGRATION_PREFIX,
    IntegrationTestCase,
    discover_test_cases,
    list_integration_files,
    make_case_readers,
)
from tests.support.remote_data import RemoteTestConfig
from tests.utils import format_container_diff


class LocalTestConfig(RemoteTestConfig):
    """Load CLI settings from the repository .env, with environment variables taking priority."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def regenerate_case(config: RemoteTestConfig, case: IntegrationTestCase, output: Path) -> bool:
    """Compare one source with its baseline and save only changed output without overwriting."""
    source_reader, expected_reader = make_case_readers(config, case)
    actual = source_reader[0]
    expected = expected_reader[0]
    differences = list(container_diff(expected, actual))
    if not differences:
        print(f"[UNCHANGED] {case.source_files[0]} -> {case.expected_files[0]}\n", flush=True)
        return False

    output.parent.mkdir(parents=True, exist_ok=True)
    with actual.serialize_to_hdf5() as buffer, output.open("xb") as file:
        file.write(buffer.getbuffer())
    print(
        f"[CHANGED] {case.source_files[0]} -> {case.expected_files[0]}\n"
        f"{format_container_diff(sorted(differences))}\n"
        f"  Saved: {output.name}\n",
        flush=True,
    )
    return True


def upload_changed(config: RemoteTestConfig, changed_outputs: list[tuple[IntegrationTestCase, Path]]) -> int:
    """Upload reviewed local outputs only after confirmation; report partial updates on failure."""
    uploads = [
        (
            "/".join(part for part in (config.prefix, INTEGRATION_PREFIX, case.type, output.name) if part),
            output,
        )
        for case, output in changed_outputs
    ]
    print("\nChanged files to upload:")
    for blob_name, output in uploads:
        print(f"  {output} -> az://{config.container_name}/{blob_name}")
    print("Warning: this overwrites current baselines. Recovery depends on Azure retention/versioning settings.")
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
    """Run local baseline regeneration using TEST_* settings and Azure login."""
    parser = argparse.ArgumentParser(
        description="Compare Azure integration baselines and save changed HDF5 files locally for review.",
        epilog="Loads TEST_* settings from the repository .env or environment, and uses Azure login.",
    )
    parser.add_argument(
        "path", help="Source file or folder relative to TEST_PREFIX, e.g. integration/edevis/source3.di or integration."
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Local review folder. Existing outputs are not replaced."
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
        config = LocalTestConfig()
        cases = []
        for case in discover_test_cases(list_integration_files(config)):
            if case.name == "glob":
                continue
            source_path = f"{INTEGRATION_PREFIX}/{case.type}/{case.source_files[0]}"
            if source_path == selected_path or source_path.startswith(f"{selected_path}/"):
                cases.append(case)
    except Exception as error:
        # Azure exception messages can contain credentials or signed URLs.
        print(f"Discovery failed ({type(error).__name__}). Check TEST_* settings and Azure access.", file=sys.stderr)
        return 1
    if not cases:
        parser.error(f"No paired integration source files match {args.path!r} relative to TEST_PREFIX.")

    outputs = [args.output / case.type / case.expected_files[0] for case in cases]
    for output in outputs:
        if output.exists():
            parser.error(f"Refusing to overwrite {output}. Choose a new output folder.")

    print(
        f"Azure: {config.account_url}\nComparison: container1 = current expected; container2 = regenerated.\n",
        flush=True,
    )
    changed = unchanged = failed = 0
    changed_outputs = []
    current_type = None
    for case, output in zip(cases, outputs, strict=True):
        if case.type != current_type:
            remote_folder = "/".join(
                part for part in (config.container_name, config.prefix, INTEGRATION_PREFIX, case.type) if part
            )
            print(f"Folder: az://{remote_folder}/\nOutput: {output.parent}/\n", flush=True)
            current_type = case.type
        try:
            if regenerate_case(config, case, output):
                changed += 1
                changed_outputs.append((case, output))
            else:
                unchanged += 1
        except Exception as error:
            print(
                f"[FAILED] {case.source_files[0]} -> {case.expected_files[0]}\n"
                f"  {type(error).__name__}: Check Azure access and input/output files.\n",
                flush=True,
            )
            failed += 1

    print(f"Summary: {changed} changed, {unchanged} unchanged, {failed} failed.\nAzure was not modified.")
    if args.upload and failed:
        print("Upload skipped because regeneration failed. Fix the errors before updating Azure.")
    elif args.upload and changed_outputs:
        return upload_changed(config, changed_outputs)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
