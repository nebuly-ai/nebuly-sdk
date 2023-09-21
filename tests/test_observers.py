from __future__ import annotations

from nebuly.entities import SpanWatch
from nebuly.monkey_patching import _patcher
from nebuly.observers import NebulyObserver


def function(a: float, b: int, *, c: int = 0) -> int:
    return int(a + b + c)


def test_observer_calls_publisher_when_patched_is_called() -> None:
    publisher: list[SpanWatch] = []
    observer = NebulyObserver(
        api_key="test_api_key",
        publish=publisher.append,
    )

    patched = _patcher(observer.on_event_received, "module", "0.1.0", "function_name")(
        function
    )
    result = patched(1.0, 2, c=3)

    assert result == 6
    assert len(publisher) == 1


# def test_observer_doesnt_override_nebuly_kwargs() -> None:
#     publisher: list[SpanWatch] = []
#     observer = NebulyObserver(
#         api_key="test_api_key",
#         publish=publisher.append,
#     )
#
#     patched = _patcher(observer.on_event_received, "module", "0.1.0", "function_name")(  # noqa: E501
#         function
#     )
#     result = patched(
#         1.0,
#         2,
#         c=3,
#         nebuly_user="test_user",
#     )
#
#     assert result == 6
#     assert len(publisher) == 1
#     watched = publisher[0]
#     assert watched.called_with_nebuly_kwargs["nebuly_user"] == "test_user"


# def test_observer_adds_undefine_as_user_if_not_passed() -> None:
#     publisher: list[SpanWatch] = []
#     observer = NebulyObserver(
#         api_key="test_api_key",
#         publish=publisher.append,
#     )
#
#     patched = _patcher(observer.on_event_received, "module", "0.1.0", "function_name")(  # noqa: E501
#         function
#     )
#     result = patched(1.0, 2, c=3)
#
#     assert result == 6
#     assert len(publisher) == 1
#     watched = publisher[0]
#     assert watched.called_with_nebuly_kwargs["nebuly_user"] == "undefined"
