import pytest

from pythermondt.data.datacontainer.node import DataNode, GroupNode, RootNode
from pythermondt.data.datacontainer.utils import format_bytes, is_datanode, is_groupnode, is_rootnode


# Type guard tests
def test_is_rootnode(root_node: RootNode, group_node: GroupNode, data_node: DataNode):
    """Test is_rootnode type guard function."""
    assert is_rootnode(root_node) is True
    assert is_rootnode(group_node) is False
    assert is_rootnode(data_node) is False


def test_is_groupnode(root_node: RootNode, group_node: GroupNode, data_node: DataNode):
    """Test is_groupnode type guard function."""
    assert is_groupnode(root_node) is False
    assert is_groupnode(group_node) is True
    assert is_groupnode(data_node) is False


def test_is_datanode(root_node: RootNode, group_node: GroupNode, data_node: DataNode):
    """Test is_datanode type guard function."""
    assert is_datanode(root_node) is False
    assert is_datanode(group_node) is False
    assert is_datanode(data_node) is True


# Byte formatting tests
def test_format_bytes_basic():
    """Test basic byte formatting."""
    assert format_bytes(0) == "0.00 B"
    assert format_bytes(512) == "512.00 B"
    assert format_bytes(1023) == "1023.00 B"


def test_format_bytes_kilobytes():
    """Test kilobyte formatting."""
    assert format_bytes(1024) == "1.00 KB"
    assert format_bytes(1536) == "1.50 KB"
    assert format_bytes(2048) == "2.00 KB"


def test_format_bytes_megabytes():
    """Test megabyte formatting."""
    assert format_bytes(1048576) == "1.00 MB"
    assert format_bytes(1572864) == "1.50 MB"
    assert format_bytes(2097152) == "2.00 MB"


def test_format_bytes_gigabytes():
    """Test gigabyte formatting."""
    assert format_bytes(1073741824) == "1.00 GB"
    assert format_bytes(1610612736) == "1.50 GB"
    assert format_bytes(2147483648) == "2.00 GB"


def test_format_bytes_terabytes():
    """Test terabyte formatting."""
    assert format_bytes(1099511627776) == "1.00 TB"
    assert format_bytes(1649267441664) == "1.50 TB"


def test_format_bytes_large_values():
    """Test formatting very large values."""
    # Test petabyte
    pb_value = 1024**5
    result = format_bytes(pb_value)
    assert "PB" in result
    assert "1.00" in result


def test_format_bytes_edge_cases():
    """Test format_bytes with edge cases."""
    # Test exactly at boundaries
    assert format_bytes(1024) == "1.00 KB"
    assert format_bytes(1048576) == "1.00 MB"
    assert format_bytes(1073741824) == "1.00 GB"
    assert format_bytes(1099511627776) == "1.00 TB"


# Only run the tests in this file if it is run directly
if __name__ == "__main__":
    pytest.main(["-v", __file__])
