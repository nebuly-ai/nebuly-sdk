import sys
import logging
import importlib
from types import ModuleType
from nebuly.entities import Package

from nebuly.exceptions import AlreadyImportedError
from nebuly.patcher import Observer_T, patcher


logger = logging.getLogger(__name__)


def check_no_packages_already_imported(packages: list[Package]) -> None:
    for package in packages:
        if package.name in sys.modules:
            raise AlreadyImportedError(
                f"{package.name} already imported. Make sure you initialize",
                "Nebuly before importing any packages.",
            )


def import_and_patch_packages(packages: list[Package], observer: Observer_T) -> None:
    for package in packages:
        try:
            _monkey_patch(package, observer)
        except ImportError:
            pass


def _monkey_patch(package: Package, observer: Observer_T) -> None:
    """
    Patch each attribute in package.to_patch with the observer.
    """
    module = importlib.import_module(package.name)
    for attr in package.to_patch:
        try:
            _monkey_patch_attribute(attr, module, observer)
        except AttributeError:
            logger.warning("Failed to patch %s", attr)


def _monkey_patch_attribute(
    attr: str, module: ModuleType, observer: Observer_T
) -> None:
    tmp_component = module
    path = attr.split(".")
    for component_name in path[:-1]:
        tmp_component = getattr(tmp_component, component_name)

    setattr(
        tmp_component,
        path[-1],
        patcher(observer)(getattr(tmp_component, path[-1])),
    )
