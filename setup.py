from setuptools import setup
from pathlib import Path

def get_version():
    """Get the version number."""
    pkg_init = Path("src/pythermondt/_version.py").read_text()
    locals_dict = {}
    exec(pkg_init, {}, locals_dict)
    return locals_dict["__version__"]

if __name__ == '__main__':
    setup(version=get_version())