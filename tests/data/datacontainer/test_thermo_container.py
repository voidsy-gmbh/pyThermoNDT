import pytest
import torch
from torch import Tensor
from pythermondt.data import ThermoContainer
from pythermondt.data.datacontainer import GroupNode, DataNode

def test_initialization(thermo_container: ThermoContainer):
    """Test initialization of ThermoContainer."""
    # Expected groups that should be present after initialization of ThermoContainer
    expected_groups = ['/Data', '/GroundTruth', '/MetaData']
    expected_datasets = ['/Data/Tdata', '/GroundTruth/DefectMask', '/MetaData/LookUpTable', '/MetaData/ExcitationSignal', '/MetaData/DomainValues']

    # Check if all expected groups are present
    for group in expected_groups:
        # Extract node from ThermoContainer and check if it is a GroupNode
        group_node = thermo_container.nodes[group]
        assert isinstance(group_node, GroupNode)
    
    for dataset in expected_datasets:
        data_node = thermo_container.nodes[dataset]
        assert isinstance(data_node, DataNode)

def test_dataset_operations(thermo_container: ThermoContainer, sample_tensor: Tensor):
    """Test dataset operations of ThermoContainer."""
    # Test updating a dataset
    data = sample_tensor
    thermo_container.update_dataset('/Data/Tdata', data)
    retrieved_data = thermo_container.get_dataset('/Data/Tdata')
    assert retrieved_data.equal(data)

def test_attribute_operations(thermo_container: ThermoContainer):
    """Test attribute operations of ThermoContainer."""
    teststring = "Test_value"
    thermo_container.add_attribute('/MetaData/DomainValues', "Test_attr", teststring)
    assert thermo_container.get_attribute('/MetaData/DomainValues', "Test_attr") == teststring

# Only run the tests in this file if it is run directly
if __name__ == '__main__':
    pytest.main(["-v", __file__])