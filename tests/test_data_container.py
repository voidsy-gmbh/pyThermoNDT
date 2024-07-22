import unittest
import numpy as np
import torch
from pythermondt.data import DataContainer

class TestsDataContainer(unittest.TestCase):
    def setUp(self):
        self.testcontainer = DataContainer()
        self.groups = ['MetaData', 'Data', 'GroundTruth']
        self.datasets = {'Data': ['Tdata'], 'GroundTruth': ['DefectMask'], 'MetaData': ['LookUpTable', 'ExcitationSignal', 'DomainValues']}
        self.array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_inital_groups(self):
        # Check for equal count
        self.assertCountEqual(self.testcontainer.get_groups(), self.groups)

        # Check if the groups are created correctly
        for group in self.groups:
            self.assertIn(group, self.testcontainer.get_groups())

    def test_inital_datasets(self):
        for group in self.datasets:
            # Check for equal count
            self.assertCountEqual(self.testcontainer.get_datasets(group), self.datasets[group])
            for dataset in self.datasets[group]:
                # Check if the datasets are created correctly
                self.assertIn(dataset, self.testcontainer.get_datasets(group))

    def test_fill_dataset_numpy(self):
        # Fill the dataset with numpy.ndarray
        self.testcontainer.fill_dataset('Data/Tdata', self.array)
        
        # Check if the dataset is filled correctly
        self.assertTrue(torch.equal(self.testcontainer.get_dataset('Data/Tdata'), self.tensor))

    def test_fill_dataset_tensor(self):
        self.testcontainer.fill_dataset('Data/Tdata', self.tensor)

        # Check if the dataset is filled correctly
        self.assertTrue(torch.equal(self.testcontainer.get_dataset('Data/Tdata'), self.tensor))

    def test_fill_dataset_nonexist(self):
        # Try to fill a non-existing dataset
        with self.assertRaises(ValueError):
            self.testcontainer.fill_dataset('Data/NonExist', self.tensor)

            
if __name__ == '__main__':
    unittest.main()