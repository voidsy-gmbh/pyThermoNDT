from dataclasses import dataclass
from pathlib import Path


@dataclass
class IntegrationTestCase:
    """Represents an integration test case based on folder structure."""

    type: str  # The test type (subdirectory name like "simulation", "edevis", etc.)
    name: str  # Test case name (typically number or descriptive name)
    source_file: Path  # Path to source file
    expected_file: Path  # Path to expected output file

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
    test_cases = []

    # Get all subfolders (test types)
    for test_type_dir in base_dir.iterdir():
        if not test_type_dir.is_dir():
            continue

        # Get all source files
        test_type = test_type_dir.name
        source_files = sorted(list(test_type_dir.glob("source*.*")))

        for source_file in source_files:
            # Extract test name from filename (e.g., "source1.mat" -> "1")
            name = source_file.stem.replace("source", "")

            # Find corresponding expected file
            expected_file = test_type_dir / f"expected{name}.hdf5"
            assert expected_file.exists(), "Expected file not found: " + str(expected_file)

            # Create test case
            test_cases.append(
                IntegrationTestCase(type=test_type, name=name, source_file=source_file, expected_file=expected_file)
            )

    return test_cases
