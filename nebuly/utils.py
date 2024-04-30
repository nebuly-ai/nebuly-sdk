import importlib.metadata

from packaging import version


def is_module_version_less_than(module_name: str, version_spec: str) -> bool:
    """
    Check if the installed version of a module is less than the specified version.

    Parameters:
    module_name (str): The name of the module to check.
    version_spec (str): The version to compare against in the format
    'major.minor.patch'.

    Returns:
    bool: True if the installed version is less than the specified version,
    False otherwise.
    """
    try:
        installed_version = importlib.metadata.version(module_name)
        return version.parse(installed_version) < version.parse(version_spec)
    except importlib.metadata.PackageNotFoundError:
        print(f"Module '{module_name}' is not installed.")
        return False
