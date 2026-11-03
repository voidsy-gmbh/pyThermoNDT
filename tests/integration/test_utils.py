import pytest

from .utils import IntegrationTestCase, discover_test_cases


def test_discover_pairs_sources_and_expected_by_number():
    """Sources pair with their expected output by case number, plus one glob case per type."""
    cases = discover_test_cases(
        ["simulation/source2.mat", "simulation/expected1.hdf5", "simulation/source1.mat", "simulation/expected2.hdf5"]
    )

    assert cases == (
        IntegrationTestCase(
            type="simulation", name="1", source_files=("source1.mat",), expected_files=("expected1.hdf5",)
        ),
        IntegrationTestCase(
            type="simulation", name="2", source_files=("source2.mat",), expected_files=("expected2.hdf5",)
        ),
        IntegrationTestCase(
            type="simulation",
            name="glob",
            source_files=("source1.mat", "source2.mat"),
            expected_files=("expected1.hdf5", "expected2.hdf5"),
        ),
    )


def test_discover_skips_source_without_expected_output():
    """Sources without matching expected output are ignored."""
    cases = discover_test_cases(["simulation/source1.mat", "simulation/expected1.hdf5", "simulation/source2.mat"])

    assert [case.id for case in cases] == ["simulation_1", "simulation_glob"]
    assert cases[0].source_files == ("source1.mat",)


def test_discover_ignores_top_level_files_and_groups_by_type():
    """Top-level files are ignored; each type subfolder gets its own cases."""
    cases = discover_test_cases(
        [
            "readme.txt",
            "simulation/source1.mat",
            "simulation/expected1.hdf5",
            "edevis/source1.mat",
            "edevis/expected1.hdf5",
        ]
    )

    assert [case.id for case in cases] == ["edevis_1", "edevis_glob", "simulation_1", "simulation_glob"]


def test_discover_pairs_alphanumeric_case_numbers():
    """Case numbers may contain letters; pairing follows the source file name."""
    cases = discover_test_cases(["simulation/sourceA.mat", "simulation/expectedA.hdf5"])

    assert [case.id for case in cases] == ["simulation_A", "simulation_glob"]
    assert cases[0].expected_files == ("expectedA.hdf5",)


def test_discover_empty_listing_gives_no_cases():
    """An empty test data listing yields no cases."""
    assert discover_test_cases([]) == ()


@pytest.mark.parametrize(
    "paths",
    [
        [],
        ["simulation/source1.mat", "simulation/expected1.hdf5"],
    ],
)
def test_discover_cases_have_readable_ids(paths: list[str]):
    """Every discovered case exposes a stable id of type and name."""
    for case in discover_test_cases(paths):
        assert case.id == f"{case.type}_{case.name}"
