import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("pytest_exit_code", [0, 1])
@pytest.mark.skipif(os.name != "posix", reason="The coverage job uses a POSIX Bash runner.")
def test_coverage_pipeline_preserves_pytest_exit_code(tmp_path: Path, pytest_exit_code: int):
    """The coverage log must not hide a failing pytest command."""
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/tests.yml").read_text()
    coverage_step = next(step for step in workflow.split("\n      - name:") if "| tee pytest-coverage.txt" in step)
    settings = dict(line.strip().split(": ", 1) for line in coverage_step.splitlines() if ": " in line)
    assert settings["shell"].split("#", 1)[0].strip() == "bash"

    pytest_stub = tmp_path / "pytest"
    pytest_stub.write_text(f"#!/bin/sh\necho 'test output'\nexit {pytest_exit_code}\n")
    pytest_stub.chmod(0o755)
    environment = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}
    # GitHub uses this invocation for an explicit `shell: bash`.
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", settings["run"]],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == pytest_exit_code
    assert (tmp_path / "pytest-coverage.txt").read_text() == "test output\n"
