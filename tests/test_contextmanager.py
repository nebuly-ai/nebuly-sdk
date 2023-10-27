# pylint: disable=duplicate-code
from __future__ import annotations

import asyncio
import random
import threading
from time import sleep

import pytest

from nebuly.contextmanager import (
    InteractionContext,
    get_nearest_open_interaction,
    new_interaction,
)
from nebuly.exceptions import (
    AlreadyInInteractionContext,
    InteractionContextInitiationError,
    InteractionMustBeLocalVariable,
)


def test_interaction_context() -> None:
    with new_interaction("test_user", "test_group_profile") as interaction:
        assert get_nearest_open_interaction() is interaction


def test_interaction_context_finish() -> None:
    with new_interaction("test_user", "test_group_profile") as interaction:
        assert get_nearest_open_interaction() is interaction

    with new_interaction("test_other_user", "test_other_group_profile") as interaction:
        assert get_nearest_open_interaction() is interaction


def test_get_interaction_no_save() -> None:
    with pytest.raises(InteractionMustBeLocalVariable):
        with new_interaction("test_user", "test_group_profile"):
            pass


def test_multithreading_context() -> None:
    def thread_func() -> None:
        with new_interaction("test_user", "test_group_profile") as interaction:
            sleep(random.random())
            assert get_nearest_open_interaction() is interaction

    threads = []
    for _ in range(10):
        thread = threading.Thread(target=thread_func)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


@pytest.mark.asyncio
async def test_asyncio_context() -> None:
    async def async_func() -> None:
        with new_interaction("test_user", "test_group_profile") as interaction:
            await asyncio.sleep(random.random())
            assert get_nearest_open_interaction() is interaction

    await asyncio.gather(async_func(), async_func(), async_func())


def test_cannot_directly_create_interaction() -> None:
    with pytest.raises(InteractionContextInitiationError):
        InteractionContext("test_user")


def test_cannot_create_interaction_inside_interaction() -> None:
    with new_interaction(
        "test_user", "test_group_profile"
    ) as interaction:  # noqa: F841  pylint: disable=unused-variable
        with pytest.raises(AlreadyInInteractionContext):
            with new_interaction(
                "test_user"
            ) as interaction2:  # noqa: F841 pylint: disable=unused-variable
                pass


def test_calls_finish_when_exception_raised() -> None:
    with pytest.raises(Exception):
        with new_interaction("test_user") as interaction:
            raise Exception("test")  # pylint: disable=broad-exception-raised

    assert interaction._finished  # pylint: disable=protected-access
