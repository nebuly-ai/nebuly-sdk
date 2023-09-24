from __future__ import annotations

from asyncio import sleep
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Generator
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nebuly.contextmanager import InteractionContext, new_interaction
from nebuly.entities import InteractionWatch, Package
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


@patch("nebuly.monkey_patching._add_interaction_span", return_value=None)
@given(args=st.tuples(st_any), kwargs=st.dictionaries(st.text(), st_any))
def test_patcher_doesnt_change_any_behavior(
    mock, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    def to_patched(
        *args: tuple[Any, ...], **kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """This is the docstring to be tested"""
        return args, kwargs

    patched = _patcher(lambda _: None, "module", "0.1.0", "function_name")(to_patched)

    assert patched(*args, **kwargs) == to_patched(*args, **kwargs)
    assert patched.__name__ == to_patched.__name__
    assert patched.__doc__ == to_patched.__doc__
    assert patched.__module__ == to_patched.__module__
    assert patched.__qualname__ == to_patched.__qualname__
    assert patched.__annotations__ == to_patched.__annotations__


@given(args=st.tuples(st_any), kwargs=st.dictionaries(st.text(), st_any))
def test_patcher_calls_observer(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:  # noqa: E501
    def to_patched(
        *args: tuple[Any, ...], **kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """This is the docstring to be tested"""
        return args, kwargs

    observer: list[InteractionWatch] = []

    patched = _patcher(observer.append, "module", "0.1.0", "function_name")(
        to_patched
    )  # noqa: E501

    before = datetime.now(timezone.utc)
    patched(*args, **kwargs)
    after = datetime.now(timezone.utc)

    assert len(observer) == 1
    watched = observer[0]
    assert isinstance(watched, InteractionWatch)
    assert len(watched.spans) == 1
    span = watched.spans[0]
    assert span.function == "function_name"
    assert span.module == "module"
    assert before <= span.called_start <= span.called_end <= after
    assert span.called_with_args == args
    assert span.called_with_kwargs == kwargs
    assert span.returned == to_patched(*args, **kwargs)


def test_watched_is_immutable() -> None:
    def to_patched(mutable: list[int]) -> list[int]:
        mutable.append(1)
        return mutable

    observer: list[InteractionWatch] = []
    mutable: list[int] = []

    _patcher(observer.append, "module", "0.1.0", "function_name")(to_patched)(mutable)

    mutable.append(2)

    assert len(observer) == 1
    watched = observer[0]
    assert isinstance(watched, InteractionWatch)
    assert len(watched.spans) == 1
    span = watched.spans[0]
    assert span.called_with_args == ([],)
    assert span.returned == [1]


def test_split_nebuly_kwargs() -> None:
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


def test_nebuly_args_are_intercepted() -> None:
    def function(a: int, b: int) -> int:
        return a + b

    observer: list[InteractionWatch] = []
    patched = _patcher(observer.append, "module", "0.1.0", "function_name")(function)

    patched(1, 2, nebuly_segment="segment", nebuly_project="project")

    assert len(observer) == 1
    watched = observer[0]
    assert len(watched.spans) == 1
    span = watched.spans[0]
    assert isinstance(watched, InteractionWatch)
    assert span.called_with_args == (1, 2)
    assert span.returned == 3


@patch("nebuly.monkey_patching.logger")
def test_logs_warning_when_package_already_imported(logger: MagicMock) -> None:
    package = Package("nebuly", ("0.1.0",), ("non_existent",))
    check_no_packages_already_imported([package])
    logger.warning.assert_called_once_with("%s already imported", "nebuly")


def test_monkey_patch() -> None:
    package = Package(
        "tests.to_patch",
        ("0.1.0",),
        (
            "ToPatch.to_patch_one",
            "ToPatch.to_patch_two",
        ),
    )
    observer: list[InteractionWatch] = []

    import_and_patch_packages([package], observer.append)

    from .to_patch import ToPatch  # pylint: disable=import-outside-toplevel

    result = ToPatch().to_patch_one(1, 2.0, c=3)
    assert result == 6


def test_monkey_patch_missing_module_doesnt_break() -> None:
    package = Package(
        "non_existent",
        ("0.1.0",),
        ("ToPatch.to_patch_one",),
    )
    observer: list[InteractionWatch] = []

    import_and_patch_packages([package], observer.append)


def test_monkey_patch_missing_component_doesnt_break_other_patches() -> None:
    package = Package(
        "tests.to_patch",
        ("0.1.0",),
        (
            "ToPatch.non_existent",
            "ToPatch.to_patch_one",
        ),
    )
    observer: list[InteractionWatch] = []

    _monkey_patch(package, observer.append)

    from .to_patch import ToPatch  # pylint: disable=import-outside-toplevel

    result = ToPatch().to_patch_one(1, 2.0, c=3)
    assert result == 6


@given(args=st.tuples(st_any), kwargs=st.dictionaries(st.text(), st_any))
def test_patcher_calls_observer_after_generator_has_finished(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    def to_patched_generator(  # pylint: disable=unused-argument
        *args: int, **kwargs: str
    ) -> Generator[int, None, None]:
        """This is the docstring to be tested"""
        for i in range(3):
            yield i

    observer: list[InteractionWatch] = []

    patched = _patcher(observer.append, "module", "0.1.0", "function_name")(
        to_patched_generator
    )

    before = datetime.now(timezone.utc)
    generator = patched(*args, **kwargs)
    datetime.now(timezone.utc)

    consumed_generator = list(generator)
    after = datetime.now(timezone.utc)

    assert consumed_generator == [0, 1, 2]
    assert len(observer) == 1
    watched = observer[0]
    assert isinstance(watched, InteractionWatch)
    assert len(watched.spans) == 1
    span = watched.spans[0]
    assert span.returned == [0, 1, 2]
    assert span.generator is True
    assert span.generator_first_element_timestamp is not None
    assert (
        before
        <= span.called_start
        <= span.generator_first_element_timestamp
        <= span.called_end
        <= after
    )


@pytest.mark.asyncio
async def test_patcher_async_function() -> None:
    async def to_patched_async_function() -> int:
        await sleep(0.1)
        return 1

    patched = _patcher(lambda _: None, "module", "0.1.0", "function_name")(
        to_patched_async_function
    )

    assert await patched() == 1


@pytest.mark.asyncio
async def test_patcher_async_generator() -> None:
    async def to_patched_async_generator() -> AsyncGenerator[int, None]:
        await sleep(0.1)
        yield 1
        await sleep(0.1)
        yield 2

    patched = _patcher(lambda _: None, "module", "0.1.0", "function_name")(
        to_patched_async_generator
    )

    generator = await patched()

    result = [i async for i in generator]
    assert result == [1, 2]


@pytest.mark.asyncio
async def test_patcher_async_return_generator() -> None:
    async def async_range(n: int) -> AsyncGenerator[int, None]:
        for i in range(n):
            await sleep(0.1)
            yield i

    async def to_patched_async_generator() -> AsyncGenerator[int, None]:
        return (i async for i in async_range(3))

    patched = _patcher(lambda _: None, "module", "0.1.0", "function_name")(
        to_patched_async_generator
    )

    generator = await patched()

    result = [i async for i in generator]
    assert result == [0, 1, 2]
