import os

from pythermondt.readers import LocalReader

from ..utils import containers_equal


def test_local_reader(integration_assets_path):
    """Test the LocalReader with simulation sources."""
    source_files = [os.path.join(integration_assets_path, "simulation", f"source{i}.mat") for i in range(1, 4)]
    expected_files = [os.path.join(integration_assets_path, "simulation", f"expected{i}.hdf5") for i in range(1, 4)]

    for source, expected in zip(source_files, expected_files, strict=False):
        source_reader = LocalReader(source)
        expected_reader = LocalReader(expected)

        source_container = next(source_reader)
        expected_container = next(expected_reader)

        assert containers_equal(source_container, expected_container, print_diff=True), (
            f"Containers at {source} and {expected} are not equal"
        )
