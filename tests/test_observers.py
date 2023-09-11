from __future__ import annotations

import pytest

from nebuly.entities import DevelopmentPhase, Watched
from nebuly.monkey_patching import _patcher
from nebuly.observers import NebulyObserver


class Publisher:
    def __init__(self) -> None:
        self.messages: list[Watched] = []

    def publish(self, message: Watched) -> None:
        self.messages.append(message)


def function(a: float, b: int, *, c: int = 0) -> int:
    return int(a + b + c)


def test_observer_calls_publisher_when_patched_is_called():
    publisher = Publisher()
    observer = NebulyObserver(
        api_key="test_api_key",
        project="test_project",
        phase=DevelopmentPhase.EXPERIMENTATION,
        publish=publisher.publish,
    )

    patched = _patcher(observer.on_event_received, "module", "0.1.0", "function_name")(
        function
    )
    result = patched(1.0, 2, c=3)

    assert result == 6
    assert len(publisher.messages) == 1
    watched = publisher.messages[0]
    assert watched.called_with_nebuly_kwargs["nebuly_project"] == "test_project"
    assert (
        watched.called_with_nebuly_kwargs["nebuly_phase"]
        == DevelopmentPhase.EXPERIMENTATION
    )


def test_observer_sets_nebuly_kwargs():
    publisher = Publisher()
    observer = NebulyObserver(
        api_key="test_api_key",
        project="test_project",
        phase=DevelopmentPhase.EXPERIMENTATION,
        publish=publisher.publish,
    )

    patched = _patcher(observer.on_event_received, "module", "0.1.0", "function_name")(
        function
    )
    result = patched(1.0, 2, c=3)

    assert result == 6
    assert len(publisher.messages) == 1
    watched = publisher.messages[0]
    assert watched.called_with_nebuly_kwargs["nebuly_project"] == "test_project"
    assert (
        watched.called_with_nebuly_kwargs["nebuly_phase"]
        == DevelopmentPhase.EXPERIMENTATION
    )


def test_observer_doesnt_override_nebuly_kwargs():
    publisher = Publisher()
    observer = NebulyObserver(
        api_key="test_api_key",
        project="test_project",
        phase=DevelopmentPhase.EXPERIMENTATION,
        publish=publisher.publish,
    )

    patched = _patcher(observer.on_event_received, "module", "0.1.0", "function_name")(
        function
    )
    result = patched(
        1.0,
        2,
        c=3,
        nebuly_project="other_project",
        nebuly_phase=DevelopmentPhase.PREPROCESSING,
    )

    assert result == 6
    assert len(publisher.messages) == 1
    watched = publisher.messages[0]
    assert watched.called_with_nebuly_kwargs["nebuly_project"] == "other_project"
    assert (
        watched.called_with_nebuly_kwargs["nebuly_phase"]
        == DevelopmentPhase.PREPROCESSING
    )


def test_nebuly_observer_raises_exception_if_invalid_phase():
    publisher = Publisher()
    observer = NebulyObserver(
        api_key="test_api_key",
        project="test_project",
        phase="invalid_phase",  # type: ignore
        publish=publisher.publish,
    )
    patched = _patcher(observer.on_event_received, "module", "0.1.0", "function_name")(
        function
    )

    with pytest.raises(ValueError):
        patched(1.0, 2, c=3)


def test_nebuly_observer_raises_exception_if_invalid_phase_override():
    publisher = Publisher()
    observer = NebulyObserver(
        api_key="test_api_key",
        project="test_project",
        phase=DevelopmentPhase.EXPERIMENTATION,
        publish=publisher.publish,
    )
    patched = _patcher(observer.on_event_received, "module", "0.1.0", "function_name")(
        function
    )

    with pytest.raises(ValueError):
        patched(1.0, 2, c=3, nebuly_phase="invalid_phase")
