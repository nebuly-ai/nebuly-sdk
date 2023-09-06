from dataclasses import asdict
from queue import Queue

from nebuly.entities import Message, Observer_T, Package
from nebuly.monkey_patcher import (
    check_no_packages_already_imported,
    import_and_patch_packages,
)
from nebuly.observer import NebulyObserver
from nebuly.publisher import Publisher
from nebuly.requests import post_json_data

PACKAGES = [
    Package(
        "openai",
        ["0.10.2"],
        [
            "Completion.create",
            "Completion.create",
            "ChatCompletion.create",
            "Edit.create",
            "Image.create",
            "Image.create_edit",
            "Image.create_variation",
            "Embedding.create",
            "Audio.transcribe",
            "Audio.translate",
            "FineTune.create",
            "Moderation.create",
        ],
    )
]


def init(*, api_key: str, project: str, phase: str) -> None:
    check_no_packages_already_imported(PACKAGES)
    observer = _create_observer_and_start_publisher(
        api_key=api_key, project=project, phase=phase
    )
    import_and_patch_packages(PACKAGES, observer)


def _post_message(message: Message) -> None:
    post_json_data("http://localhost:8000/api/messages/", asdict(message))


def _create_observer_and_start_publisher(
    *, api_key: str, project: str, phase: str
) -> Observer_T:
    queue: Queue[Message] = Queue()

    Publisher(queue, _post_message)
    observer = NebulyObserver(
        api_key=api_key,
        project=project,
        phase=phase,
        publish=queue.put,
    )
    return observer.observe
