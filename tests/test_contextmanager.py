import asyncio
import random
import threading
from time import sleep

import pytest

from nebuly.contextmanager import (
    InteractionMustBeLocalVariable,
    get_nearest_open_interaction,
    new_interaction,
)


def test_interaction_context() -> None:
    with new_interaction("agus") as interaction:
        assert get_nearest_open_interaction() is interaction


def test_interaction_context_finish() -> None:
    with new_interaction("agus") as interaction:
        assert get_nearest_open_interaction() is interaction

    with new_interaction("foo") as interaction:
        assert get_nearest_open_interaction() is interaction


def test_get_interaction_no_save() -> None:
    with pytest.raises(InteractionMustBeLocalVariable):
        with new_interaction("test"):
            pass


def test_multithreading_context() -> None:
    def thread_func() -> None:
        with new_interaction("agus") as interaction:
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
        with new_interaction("agus") as interaction:
            await asyncio.sleep(random.random())
            assert get_nearest_open_interaction() is interaction

    await asyncio.gather(async_func(), async_func(), async_func())
