import unittest
import torch
from pythermondt.data.datacontainer.node import RootNode, GroupNode, DataNode, NodeType

class TestNode(unittest.TestCase):
    def test_root_node(self):
        root = RootNode()
        self.assertEqual(root.name, "root")
        self.assertEqual(root.type, NodeType.ROOT)

    def test_group_node(self):
        group = GroupNode("test_group")
        self.assertEqual(group.name, "test_group")
        self.assertEqual(group.type, NodeType.GROUP)

        group.add_attribute("test_attr", "test_value")
        self.assertEqual(group.get_attribute("test_attr"), "test_value")

        group.update_attribute("test_attr", "new_value")
        self.assertEqual(group.get_attribute("test_attr"), "new_value")

        group.remove_attribute("test_attr")
        with self.assertRaises(KeyError):
            group.get_attribute("test_attr")

    def test_data_node(self):
        data = torch.tensor([[1, 2], [3, 4]])
        data_node = DataNode("test_data", data)
        self.assertEqual(data_node.name, "test_data")
        self.assertEqual(data_node.type, NodeType.DATASET)
        self.assertTrue(torch.equal(data_node.data, data))

        new_data = torch.tensor([[5, 6], [7, 8]])
        data_node.data = new_data
        self.assertTrue(torch.equal(data_node.data, new_data))

        data_node.add_attribute("shape", list(data_node.data.shape))
        self.assertEqual(data_node.get_attribute("shape"), [2, 2])

    def test_node_type_enum(self):
        self.assertEqual(NodeType.ROOT.value, "root")
        self.assertEqual(NodeType.GROUP.value, "group")
        self.assertEqual(NodeType.DATASET.value, "dataset")

if __name__ == '__main__':
    unittest.main()