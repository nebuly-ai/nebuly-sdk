import asyncio
import random
import threading
from time import sleep

import pytest

from nebuly.contextmanager import (
    AlreadyInInteractionContext,
    InteractionContext,
    InteractionContextInitiationError,
    InteractionMustBeLocalVariable,
    get_nearest_open_interaction,
    new_interaction,
)


def test_interaction_context() -> None:
    with new_interaction("test_user", "test_group_profile") as interaction:
        observer = []
        interaction.set_observer(observer.append)
        assert get_nearest_open_interaction() is interaction


def test_interaction_context_finish() -> None:
    with new_interaction("test_user", "test_group_profile") as interaction:
        observer = []
        interaction.set_observer(observer.append)
        assert get_nearest_open_interaction() is interaction

    with new_interaction("test_other_user", "test_other_group_profile") as interaction:
        observer = []
        interaction.set_observer(observer.append)
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

    thread1 = threading.Thread(target=thread_func)
    thread1.start()
    thread2 = threading.Thread(target=thread_func)
    thread2.start()
    thread3 = threading.Thread(target=thread_func)
    thread3.start()

    thread1.join()
    thread2.join()
    thread3.join()


@pytest.mark.asyncio
async def test_asyncio_context() -> None:
    async def async_func() -> None:
        with new_interaction("test_user", "test_group_profile") as interaction:
            observer = []
            interaction.set_observer(observer.append)
            await asyncio.sleep(random.random())
            assert get_nearest_open_interaction() is interaction

    await asyncio.gather(async_func(), async_func(), async_func())


def test_cannot_directly_create_interaction() -> None:
    with pytest.raises(InteractionContextInitiationError):
        InteractionContext("test_user")


def test_cannot_create_interaction_inside_interaction() -> None:
    with new_interaction(
        "test_user", "test_group_profile"
    ) as interaction:  # noqa: F841 pylint: disable=unused-variable
        observer = []
        interaction.set_observer(observer.append)
        with pytest.raises(AlreadyInInteractionContext):
            with new_interaction(
                "test_user"
            ) as interaction2:  # noqa: F841 pylint: disable=unused-variable
                pass
