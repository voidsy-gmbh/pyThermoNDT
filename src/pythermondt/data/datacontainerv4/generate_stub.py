import inspect
from typing import List, Type
import sys
import os

# Add the src directory to sys.path to allow imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, src_path)

from pythermondt.data.datacontainerv4.dataset_ops import DatasetOps
from pythermondt.data.datacontainerv4.group_ops import GroupOps
from pythermondt.data.datacontainerv4.attribute_ops import AttributeOps
from pythermondt.data.datacontainerv4.container import DataContainer

def generate_stub(ops_classes: List[Type], output_file: str = 'container.pyi'):
    with open(output_file, 'w') as f:
        f.write("from typing import Any, Dict\n")
        f.write("from pythermondt.data.datacontainerv4.node import Node\n\n")
        f.write("class DataContainer:\n")
        
        # Write __init__ method
        init_doc = inspect.getdoc(DataContainer.__init__)
        f.write(f"    def __init__(self) -> None:\n")
        f.write(f'        """{init_doc}"""\n\n')

        # Write other special methods
        for name in ['__str__', '__getattr__']:
            method = getattr(DataContainer, name)
            doc = inspect.getdoc(method)
            signature = inspect.signature(method)
            f.write(f"    def {name}{signature}:\n")
            f.write(f'        """{doc}"""\n\n')

        # Write methods from ops classes
        for ops_class in ops_classes:
            for name, method in inspect.getmembers(ops_class, inspect.isfunction):
                if not name.startswith('_'):  # Only include public methods
                    signature = inspect.signature(method)
                    doc = inspect.getdoc(method)
                    f.write(f"    def {name}{signature}:\n")
                    f.write(f'        """{doc}"""\n\n')

if __name__ == "__main__":
    generate_stub([DatasetOps, GroupOps, AttributeOps])
    print("Stub file generated successfully.")

# Usage: Run this script to generate the stub file
# python -m pythermondt.data.datacontainerv4.generate_stub