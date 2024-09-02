import unittest
import torch
import io
from pythermondt.data import DataContainer
from pythermondt.data.datacontainer.node import GroupNode, DataNode

class TestDataContainer(unittest.TestCase):
    def setUp(self):
        self.container = DataContainer()

    def test_initialization(self):
        self.assertEqual(len(self.container.get_groups()), 0)
        self.assertEqual(len(self.container.get_datasets()), 0)

    def test_group_operations(self):
        self.container.add_group('/', 'TestGroup')
        self.assertIn('TestGroup', self.container.get_groups())
        
        self.container.remove_group('/TestGroup')
        self.assertNotIn('TestGroup', self.container.get_groups())

    def test_dataset_operations(self):
        data = torch.tensor([[1, 2], [3, 4]])
        self.container.add_dataset('/', 'TestData', data)
        self.assertIn('TestData', self.container.get_datasets())
        
        retrieved_data = self.container.get_dataset('/TestData')
        self.assertTrue(torch.equal(retrieved_data, data))
        
        new_data = torch.tensor([[5, 6], [7, 8]])
        self.container.update_dataset('/TestData', new_data)
        updated_data = self.container.get_dataset('/TestData')
        self.assertTrue(torch.equal(updated_data, new_data))
        
        self.container.remove_dataset('/TestData')
        self.assertNotIn('TestData', self.container.get_datasets())

    def test_attribute_operations(self):
        self.container.add_group('/', 'TestGroup')
        self.container.add_attribute('/TestGroup', 'test_attr', 'test_value')
        self.assertEqual(self.container.get_attribute('/TestGroup', 'test_attr'), 'test_value')
        
        self.container.update_attribute('/TestGroup', 'test_attr', 'new_value')
        self.assertEqual(self.container.get_attribute('/TestGroup', 'test_attr'), 'new_value')
        
        self.container.remove_attribute('/TestGroup', 'test_attr')
        with self.assertRaises(KeyError):
            self.container.get_attribute('/TestGroup', 'test_attr')

    def test_serialization(self):
        data = torch.tensor([[1, 2], [3, 4]])
        self.container.add_group('/', 'TestGroup')
        self.container.add_dataset('/TestGroup', 'TestData', data)
        self.container.add_attribute('/TestGroup/TestData', 'test_attr', 'test_value')
        
        serialized = self.container.serialize_to_hdf5()
        self.assertIsInstance(serialized, io.BytesIO)
        
        new_container = DataContainer()
        new_container.deserialize(serialized)
        
        self.assertIn('TestGroup', new_container.get_groups())
        self.assertIn('TestData', new_container.get_datasets())
        retrieved_data = new_container.get_dataset('/TestGroup/TestData')
        self.assertTrue(torch.equal(retrieved_data, data))
        self.assertEqual(new_container.get_attribute('/TestGroup/TestData', 'test_attr'), 'test_value')

    def test_error_handling(self):
        # Try to access non-existent datasets/attributes
        with self.assertRaises(KeyError):
            self.container.get_dataset('/NonExistentData')
            self.container.get_attribute('/NonExistentGroup', 'attr')
        
        # Try to add the same group/dataset twice
        with self.assertRaises(KeyError):
            self.container.add_group('/', 'TestGroup')
            self.container.add_group('/', 'TestGroup') 
        

if __name__ == '__main__':
    unittest.main()