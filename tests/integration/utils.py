from dataclasses import dataclass
from pathlib import Path


@dataclass
class IntegrationTestCase:
    """Represents an integration test case based on folder structure."""

    type: str  # The test type (subdirectory name like "simulation", "edevis", etc.)
    name: str  # Test case name (typically number or descriptive name)
    source_path: str  # Path to source file
    expected_path: str  # Path to expected output file

    @property
    def id(self) -> str:
        """Returns a test ID for readable pytest output."""
        return f"{self.type}_{self.name}"


# tests/integration/utils.py
def discover_test_cases(base_dir: Path) -> list[IntegrationTestCase]:
    """Automatically discover all test cases in the integration test directory.

    Expected structure:
    base_dir/
        simulation/
            source1.mat
            expected1.hdf5
            source2.mat
            expected2.hdf5
        other-source/
            source1.xyz
            expected1.hdf5
    """
    base_path = Path(base_dir)
    test_cases = []

    # Get all subfolders (test types)
    for test_type_dir in base_path.iterdir():
        if not test_type_dir.is_dir():
            continue

        test_type = test_type_dir.name
        source_files = sorted(test_type_dir.glob("source*.*"))

        if not source_files:
            continue

        # 1. Add individual file test cases
        for source_file in source_files:
            name = source_file.stem.replace("source", "")
            expected_file = test_type_dir / f"expected{name}.hdf5"

            if not expected_file.exists():
                continue

            test_cases.append(
                IntegrationTestCase(
                    type=test_type,
                    name=name,
                    source_path=str(source_file.absolute()),
                    expected_path=str(expected_file.absolute()),
                )
            )

        # 2. Add glob pattern test case if all files have same extension
        extensions = {f.suffix for f in source_files}
        if len(extensions) == 1:
            ext = next(iter(extensions))
            test_cases.append(
                IntegrationTestCase(
                    type=test_type,
                    name="glob",
                    source_path=(test_type_dir.resolve() / f"source*{ext}").as_posix(),
                    expected_path=(test_type_dir.resolve() / "expected*.hdf5").as_posix(),
                )
            )

    return test_cases
