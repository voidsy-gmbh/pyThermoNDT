import re
from typing import List

def validate_path(path: str) -> str:
    """ Validates and normalizes the given HDF5 path.
    
    Parameters:
        path (str): The path to validate and normalize.

    Returns:
        A normalized valid HDF5 path.

    Raises:
        ValueError: If the path is not valid.
    """
    # Normalize: strip leading/trailing whitespace and ensure starting with a slash
    normalized_path = path.strip()
    if not normalized_path.startswith('/'):
        normalized_path = '/' + normalized_path

    # Check for double slashes or trailing slash
    if '//' in normalized_path:
        raise ValueError(f"Invalid path: {normalized_path} contains double slashes.")
    if normalized_path != "/" and normalized_path.endswith('/'):
        raise ValueError(f"Invalid path: {normalized_path} ends with a slash.")

    # Validate using a regex pattern
    pattern = r'^/[a-zA-Z0-9_/-]+$'
    if not re.match(pattern, normalized_path):
        raise ValueError(f"Invalid path: {normalized_path} contains invalid characters.")

    return normalized_path

def validate_paths(paths: List[str]) -> List[str]:
    """ Validates and normalizes the given list of HDF5 paths.
    
    Parameters:
        paths (List[str]): The list of paths to validate and normalize.

    Returns:
        A list of normalized valid HDF5 paths.

    Raises:
        ValueError: If any of the paths is not valid.
    """
    return [validate_path(path) for path in paths]