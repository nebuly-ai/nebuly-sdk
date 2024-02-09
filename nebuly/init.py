from __future__ import annotations

from functools import partial
from queue import Queue

from nebuly.api_key import get_api_key
from nebuly.config import PACKAGES
from nebuly.consumers import ConsumerWorker
from nebuly.entities import InteractionWatch, Observer
from nebuly.exceptions import NebulyAlreadyInitializedError
from nebuly.monkey_patching import (
    check_no_packages_already_imported,
    import_and_patch_packages,
)
from nebuly.observers import NebulyObserver
from nebuly.requests import post_message

_initialized = False


def init(
    *,
    api_key: str | None = None,
    anonymize: bool = True,
) -> None:
    if not api_key:
        api_key = get_api_key()

    _check_nebuly_is_not_initialized()
    check_no_packages_already_imported(PACKAGES)
    observer = _create_observer_and_start_publisher(
        api_key=api_key, anonymize=anonymize
    )
    import_and_patch_packages(PACKAGES, observer)


def _check_nebuly_is_not_initialized() -> None:
    global _initialized  # pylint: disable=global-statement
    if _initialized:
        raise NebulyAlreadyInitializedError("Nebuly already initialized")
    _initialized = True


def _create_observer_and_start_publisher(
    *, api_key: str, anonymize: bool = True
) -> Observer:
    queue: Queue[InteractionWatch] = Queue()

    ConsumerWorker(queue, partial(post_message, api_key=api_key, anonymize=anonymize))
    observer = NebulyObserver(
        api_key=api_key,
        publish=queue.put,
    )
    return observer.on_event_received
