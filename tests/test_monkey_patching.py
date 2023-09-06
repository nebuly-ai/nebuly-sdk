from datetime import datetime, timezone

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nebuly.entities import Package, Watched
from nebuly.exceptions import AlreadyImportedError
from nebuly.monkey_patching import (
    _monkey_patch,
    _patcher,
    _split_nebuly_kwargs,
    check_no_packages_already_imported,
    import_and_patch_packages,
)

st_any = st.one_of(
    st.integers(), st.floats(), st.text(), st.booleans(), st.none(), st.binary()
)


class Observer:
    def __init__(self) -> None:
        self.watched: list[Watched] = []

    def __call__(self, watched: Watched) -> None:
        self.watched.append(watched)


@given(args=st.tuples(st_any), kwargs=st.dictionaries(st.text(), st_any))
def test_patcher_doesnt_change_any_behavior(args, kwargs) -> None:
    def to_patched(*args: int, **kwargs: str):
        """This is the docstring to be tested"""
        return args, kwargs

    patched = _patcher(lambda _: None)(to_patched)

    assert patched(*args, **kwargs) == to_patched(*args, **kwargs)
    assert patched.__name__ == to_patched.__name__
    assert patched.__doc__ == to_patched.__doc__
    assert patched.__module__ == to_patched.__module__
    assert patched.__qualname__ == to_patched.__qualname__
    assert patched.__annotations__ == to_patched.__annotations__


@given(args=st.tuples(st_any), kwargs=st.dictionaries(st.text(), st_any))
def test_patcher_calls_observer(args, kwargs) -> None:
    def to_patched(*args: int, **kwargs: str):
        """This is the docstring to be tested"""
        return args, kwargs

    observer = Observer()

    patched = _patcher(observer)(to_patched)

    before = datetime.now(timezone.utc)
    patched(*args, **kwargs)
    after = datetime.now(timezone.utc)

    assert len(observer.watched) == 1
    watched = observer.watched[0]
    assert watched.function == to_patched
    assert before <= watched.called_at <= after
    assert watched.called_with_args == args
    assert watched.called_with_kwargs == kwargs
    assert watched.called_with_nebuly_kwargs == {}
    assert watched.returned == to_patched(*args, **kwargs)


def test_watched_is_immutable() -> None:
    def to_patched(mutable: list[int]) -> list[int]:
        mutable.append(1)
        return mutable

    observer = Observer()
    mutable: list[int] = []

    _patcher(observer)(to_patched)(mutable)

    mutable.append(2)

    assert len(observer.watched) == 1
    watched = observer.watched[0]
    assert watched.called_with_args == ([],)
    assert watched.returned == [1]


def test_split_nebuly_kwargs():
    original_dict = {
        "nebuly_segment": "segment",
        "arg1": "arg1",
        "nebuly_project": "project",
        "arg2": "arg2",
    }
    nebuly_kwargs, function_kwargs = _split_nebuly_kwargs(original_dict)
    assert nebuly_kwargs == {
        "nebuly_segment": "segment",
        "nebuly_project": "project",
    }
    assert function_kwargs == {"arg1": "arg1", "arg2": "arg2"}


def test_nebuly_args_are_intercepted():
    def function(a: int, b: int) -> int:
        return a + b

    observer = Observer()
    patched = _patcher(observer)(function)

    patched(1, 2, nebuly_segment="segment", nebuly_project="project")

    assert len(observer.watched) == 1
    watched = observer.watched[0]
    assert watched.called_with_args == (1, 2)
    assert watched.called_with_nebuly_kwargs == {
        "nebuly_segment": "segment",
        "nebuly_project": "project",
    }
    assert watched.returned == 3


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

    from .to_patch import ToPatch  # pylint: disable=import-outside-toplevel

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

    from .to_patch import ToPatch  # pylint: disable=import-outside-toplevel

    result = ToPatch().to_patch_one(1, 2.0, c=3)
    assert result == 6
    assert len(observer) == 1
