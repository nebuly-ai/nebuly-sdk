import json
import os
from queue import Queue

from nebuly.config import PACKAGES
from nebuly.consumers import ConsumerWorker
from nebuly.entities import Message, Observer_T
from nebuly.exceptions import APIKeyNotProvidedError, NebulyAlreadyInitializedError
from nebuly.monkey_patching import (
    check_no_packages_already_imported,
    import_and_patch_packages,
)
from nebuly.observers import NebulyObserver

_initialized = False


def init(
    *,
    api_key: str | None = None,
    project: str | None = None,
    phase: str | None = None,
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


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, "to_dict"):
            return o.to_dict()
        try:
            return json.JSONEncoder.default(self, o)
        except Exception:  # pylint: disable=broad-except
            return str(o)


def _post_message(message: Message) -> None:
    json_data = json.dumps(message, cls=CustomJSONEncoder)
    print(json_data)
    # post_json_data("http://httpbin.org/post", json_data)


def _create_observer_and_start_publisher(
    *, api_key: str, project: str | None, phase: str | None
) -> Observer_T:
    queue: Queue[Message] = Queue()

    ConsumerWorker(queue, _post_message)
    observer = NebulyObserver(
        api_key=api_key,
        project=project,
        phase=phase,
        publish=queue.put,
    )
    return observer.on_event_received
