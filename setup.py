from pathlib import Path

from setuptools import setup


def get_version():
    """Get the version number."""
    pkg_init = Path("src/pythermondt/__pkginfo__.py").read_text()
    locals_dict = {}
    exec(pkg_init, {}, locals_dict)
    return locals_dict["__version__"]


if __name__ == "__main__":
    setup(version=get_version())
