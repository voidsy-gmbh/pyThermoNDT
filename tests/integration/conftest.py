import os

import pytest


@pytest.fixture
def integration_assets_path():
    """Path to integration test assets."""
    return os.path.join("tests", "assets", "integration")


@pytest.fixture
def simulation_sources(integration_assets_path):
    """List of all simulation source files."""
    pattern = [os.path.join(integration_assets_path, "simulation", f"source{i}.mat") for i in range(1, 4)]
    return pattern
