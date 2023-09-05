from dataclasses import dataclass
import importlib
import sys
from types import ModuleType

from nebuly.exceptions import AlreadyImportedError


@dataclass
class Package:
    name: str
    versions: list[str]
    to_patch: list[str]


PACKAGES = [Package("openai", ["0.10.2"], ["openai.Completion.create"])]


def init() -> None:
    check_packages_already_imported(PACKAGES)
    import_and_patch_packages(PACKAGES)


def import_and_patch_packages(packages: list[Package]) -> None:
    for package in packages:
        import_and_patch_package(package)


def import_and_patch_package(package: Package) -> None:
    try:
        module = importlib.import_module(package.name)
        monkey_patch_package(module, package)
    except ImportError:
        pass


def check_packages_already_imported(packages: list[Package]) -> None:
    for package in packages:
        if package.name in sys.modules:
            raise AlreadyImportedError(f"{package.name} already imported")


def monkey_patch_package(module: ModuleType, package: Package) -> None:
    # TODO: monkey patch package
    pass
