import numpy as np
import pytest
import torch


@pytest.fixture
def sample_tensor():
    """Basic tensor fixture available to all tests."""
    return torch.tensor([[1, 2], [3, 4]])


@pytest.fixture
def sample_tensor2():
    """Basic tensor fixture available to all tests."""
    return torch.tensor([[5, 6], [7, 8]])


@pytest.fixture
def sample_empty_tensor():
    """Empty tensor fixture available to all tests."""
    return torch.empty(0)


@pytest.fixture
def sample_eye_tensor():
    """Identity tensor fixture available to all tests."""
    return torch.eye(3)


@pytest.fixture
def sample_ndarray():
    """Basic ndarray fixture available to all tests."""
    return np.array([[1, 2], [3, 4]])


@pytest.fixture
def sample_ndarray2():
    """Basic ndarray fixture available to all tests."""
    return np.array([[5, 6], [7, 8]])


@pytest.fixture
def sample_empty_ndarray():
    """Empty ndarray fixture available to all tests."""
    return np.empty(0)


@pytest.fixture
def sample_eye_ndarray():
    """Identity ndarray fixture available to all tests."""
    return np.eye(3)
