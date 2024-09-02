import unittest
import torch
from pythermondt.data import ThermoContainer
from pythermondt.data.datacontainer.node import GroupNode, DataNode

class TestThermoContainer(unittest.TestCase):
    def setUp(self):
        self.thermo_container = ThermoContainer()

    def test_initialization(self):
        expected_groups = ['/Data', '/GroundTruth', '/MetaData']
        expected_datasets = ['/Data/Tdata', '/GroundTruth/DefectMask', '/MetaData/LookUpTable', 
                             '/MetaData/ExcitationSignal', '/MetaData/DomainValues']
        
        for group in expected_groups:
            self.assertIsInstance(self.thermo_container.nodes[group], GroupNode)
        
        for dataset in expected_datasets:
            self.assertIsInstance(self.thermo_container.nodes[dataset], DataNode)

    def test_dataset_operations(self):
        data = torch.rand(10, 10, 5)  # Example data
        self.thermo_container.update_dataset('/Data/Tdata', data)
        retrieved_data = self.thermo_container.get_dataset('/Data/Tdata')
        self.assertTrue(torch.equal(retrieved_data, data))

    def test_attribute_operations(self):
        domain_values = torch.linspace(0, 1, 5)
        self.thermo_container.update_dataset('/MetaData/DomainValues', domain_values)
        self.thermo_container.add_attribute('/MetaData/DomainValues', 'DomainType', 'Time')
        
        retrieved_values = self.thermo_container.get_dataset('/MetaData/DomainValues')
        self.assertTrue(torch.equal(retrieved_values, domain_values))
        domain_type = self.thermo_container.get_attribute('/MetaData/DomainValues', 'DomainType')
        self.assertEqual(domain_type, 'Time')

if __name__ == '__main__':
    unittest.main()