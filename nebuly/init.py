from __future__ import annotations

import os
from functools import partial
from queue import Queue

from nebuly.config import PACKAGES
from nebuly.consumers import ConsumerWorker
from nebuly.entities import DevelopmentPhase, Observer, Watched
from nebuly.exceptions import APIKeyNotProvidedError, NebulyAlreadyInitializedError
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
    project: str | None = None,
    phase: DevelopmentPhase | None = None,
) -> None:
    if not api_key:
        api_key = _get_api_key()
    _check_nebuly_is_not_initialized()
    check_no_packages_already_imported(PACKAGES)
    observer = _create_observer_and_start_publisher(
        api_key=api_key, project=project, phase=phase
    )
    import_and_patch_packages(PACKAGES, observer)


def _get_api_key() -> str:
    api_key = os.environ.get("NEBULY_API_KEY")
    if not api_key:
        raise APIKeyNotProvidedError(
            "API key not provided and not found in environment"
        )
    return api_key


def _check_nebuly_is_not_initialized() -> None:
    global _initialized  # pylint: disable=global-statement
    if _initialized:
        raise NebulyAlreadyInitializedError("Nebuly already initialized")
    _initialized = True


def _create_observer_and_start_publisher(
    *, api_key: str, project: str | None, phase: DevelopmentPhase | None
) -> Observer:
    queue: Queue[Watched] = Queue()

    ConsumerWorker(queue, partial(post_message, api_key=api_key))
    observer = NebulyObserver(
        api_key=api_key,
        project=project,
        phase=phase,
        publish=queue.put,
    )
    return observer.on_event_received
