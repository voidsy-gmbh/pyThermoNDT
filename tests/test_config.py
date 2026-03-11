"""Tests for Settings validation in config.py."""

import pytest
from pydantic import ValidationError

from pythermondt.config import Settings


def test_num_workers_zero_raises():
    """Test that num_workers=0 raises ValidationError."""
    with pytest.raises(ValidationError, match="num_workers must be at least 1"):
        Settings(num_workers=0)


@pytest.mark.parametrize("num_workers", [-5, -1])
def test_num_workers_negative_raises(num_workers):
    """Test that negative num_workers raises ValidationError."""
    with pytest.raises(ValidationError, match="num_workers must be at least 1"):
        Settings(num_workers=num_workers)


def test_num_workers_valid():
    """Test that a valid num_workers is accepted."""
    s = Settings(num_workers=2)
    assert s.num_workers == 2


def test_invalid_log_level_raises():
    """Test that an unrecognized log_level raises ValidationError."""
    with pytest.raises(ValidationError, match="log_level must be one of"):
        Settings(log_level="INVALID")


def test_invalid_log_level_typo_raises():
    """Test that a common typo like WARN raises ValidationError."""
    with pytest.raises(ValidationError, match="log_level must be one of"):
        Settings(log_level="WARN")


def test_log_level_case_insensitive():
    """Test that log_level accepts lowercase and normalizes to uppercase."""
    s = Settings(log_level="debug")
    assert s.log_level == "DEBUG"
