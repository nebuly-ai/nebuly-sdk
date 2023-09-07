import json
from queue import Queue

from nebuly.config import PACKAGES
from nebuly.consumers import ConsumerWorker
from nebuly.entities import Message, Observer_T
from nebuly.monkey_patching import (
    check_no_packages_already_imported,
    import_and_patch_packages,
)
from nebuly.observers import NebulyObserver
from nebuly.requests import post_json_data


def init(*, api_key: str, project: str, phase: str) -> None:
    check_no_packages_already_imported(PACKAGES)
    observer = _create_observer_and_start_publisher(
        api_key=api_key, project=project, phase=phase
    )
    import_and_patch_packages(PACKAGES, observer)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, "to_dict"):
            return o.to_dict()
        try:
            return json.JSONEncoder.default(self, o)
        except Exception:
            return str(o)


def _post_message(message: Message) -> None:
    json_data = json.dumps(message, cls=CustomJSONEncoder)
    post_json_data("http://httpbin.org/post", json_data)


def _create_observer_and_start_publisher(
    *, api_key: str, project: str, phase: str
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
