import pytest

from nebuly.exceptions import AlreadyImportedError
from nebuly.entities import Watched, Package
from nebuly.monkey_patcher import (
    check_no_packages_already_imported,
    _monkey_patch,
    import_and_patch_packages,
)


def test_fails_when_package_already_imported() -> None:
    package = Package("nebuly", ["0.1.0"], ["non_existent"])
    with pytest.raises(AlreadyImportedError):
        check_no_packages_already_imported([package])


def test_monkey_patch() -> None:
    package = Package(
        "tests.to_patch",
        ["0.1.0"],
        [
            "ToPatch.to_patch_one",
            "ToPatch.to_patch_two",
        ],
    )
    observer: list[Watched] = []

    import_and_patch_packages([package], observer.append)

    from .to_patch import ToPatch

    result = ToPatch().to_patch_one(1, 2.0, c=3)
    assert result == 6
    assert len(observer) == 1


def test_monkey_patch_missing_module_doesnt_break() -> None:
    package = Package(
        "non_existent",
        ["0.1.0"],
        [
            "ToPatch.to_patch_one",
        ],
    )
    observer: list[Watched] = []

    import_and_patch_packages([package], observer.append)


def test_monkey_patch_missing_component_doesnt_break_other_patches() -> None:
    package = Package(
        "tests.to_patch",
        ["0.1.0"],
        [
            "ToPatch.non_existent",
            "ToPatch.to_patch_one",
        ],
    )
    observer: list[Watched] = []

    _monkey_patch(package, observer.append)

    from .to_patch import ToPatch

    result = ToPatch().to_patch_one(1, 2.0, c=3)
    assert result == 6
    assert len(observer) == 1
