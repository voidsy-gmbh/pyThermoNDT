import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("context", ["offline", "fork-pr"])
def test_disabled_cloud_benchmarks_do_not_download_at_collection(tmp_path: Path, context: str):
    """Collection must not touch S3 when cloud benchmarks will be skipped."""
    plugin = tmp_path / "block_cloud_downloads.py"
    plugin.write_text(
        "def pytest_configure(config):\n"
        "    from pythermondt import S3Reader\n"
        "    def fail(*args, **kwargs):\n"
        "        raise AssertionError('S3 download attempted during offline collection')\n"
        "    S3Reader.download = fail\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        REMOTE_TESTS="0",
        GITHUB_ACTIONS="true" if context == "fork-pr" else "false",
        GITHUB_EVENT_NAME="pull_request" if context == "fork-pr" else "push",
        PYTHONPATH=os.pathsep.join([str(tmp_path), str(Path.cwd()), env.get("PYTHONPATH", "")]),
    )
    env.pop("AZURE_STORAGE_CONNECTION_STRING", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "block_cloud_downloads",
            "tests/benchmark/test_transforms.py",
            "--collect-only",
            "--benchmark-skip",
            "-q",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "fraunhofer" in output
    assert "small" in output
    assert "S3 download attempted" not in output
