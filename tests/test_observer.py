from nebuly.entities import Message
from nebuly.observer import NebulyObserver
from nebuly.monkey_patcher import _patcher


class Publisher:
    def __init__(self) -> None:
        self.messages: list[Message] = []

    def publish(self, message: Message) -> None:
        self.messages.append(message)


def function(a: float, b: int, *, c: int = 0) -> int:
    return int(a + b + c)


def test_observer_calls_publisher_when_patched_is_called():
    publisher = Publisher()
    observer = NebulyObserver(
        api_key="test_api_key",
        project="test_project",
        phase="test_phase",
        publisher=publisher.publish,
    )

    patched = _patcher(observer.observe)(function)
    result = patched(1.0, 2, c=3)

    assert result == 6
    assert len(publisher.messages) == 1
    message = publisher.messages[0]
    assert message.phase == "test_phase"
    assert message.project == "test_project"
