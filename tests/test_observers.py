from __future__ import annotations

import pytest

from nebuly.entities import DevelopmentPhase, Watched, WatchedEvent
from nebuly.monkey_patching import _patcher
from nebuly.observers import NebulyObserver


def function(a: float, b: int, *, c: int = 0) -> int:
    return int(a + b + c)


def test_observer_calls_publisher_when_patched_is_called() -> None:
    publisher: list[Watched | WatchedEvent] = []
    observer = NebulyObserver(
        api_key="test_api_key",
        project="test_project",
        phase=DevelopmentPhase.EXPERIMENTATION,
        publish=publisher.append,
    )

    patched = _patcher(observer.on_event_received, "module", "0.1.0", "function_name")(
        function
    )
    result = patched(1.0, 2, c=3)

    assert result == 6
    assert len(publisher) == 1
    watched = publisher[0]
    assert watched.called_with_nebuly_kwargs["nebuly_project"] == "test_project"
    assert (
        watched.called_with_nebuly_kwargs["nebuly_phase"]
        == DevelopmentPhase.EXPERIMENTATION
    )


def test_observer_sets_nebuly_kwargs() -> None:
    publisher: list[Watched | WatchedEvent] = []
    observer = NebulyObserver(
        api_key="test_api_key",
        project="test_project",
        phase=DevelopmentPhase.EXPERIMENTATION,
        publish=publisher.append,
    )

    patched = _patcher(observer.on_event_received, "module", "0.1.0", "function_name")(
        function
    )
    result = patched(1.0, 2, c=3)

    assert result == 6
    assert len(publisher) == 1
    watched = publisher[0]
    assert watched.called_with_nebuly_kwargs["nebuly_project"] == "test_project"
    assert (
        watched.called_with_nebuly_kwargs["nebuly_phase"]
        == DevelopmentPhase.EXPERIMENTATION
    )


def test_observer_doesnt_override_nebuly_kwargs() -> None:
    publisher: list[Watched | WatchedEvent] = []
    observer = NebulyObserver(
        api_key="test_api_key",
        project="test_project",
        phase=DevelopmentPhase.EXPERIMENTATION,
        publish=publisher.append,
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
        nebuly_user="test_user",
    )

    assert result == 6
    assert len(publisher) == 1
    watched = publisher[0]
    assert watched.called_with_nebuly_kwargs["nebuly_project"] == "other_project"
    assert (
        watched.called_with_nebuly_kwargs["nebuly_phase"]
        == DevelopmentPhase.PREPROCESSING
    )
    assert watched.called_with_nebuly_kwargs["nebuly_user"] == "test_user"


def test_observer_adds_undefine_as_user_if_not_passed() -> None:
    publisher: list[Watched | WatchedEvent] = []
    observer = NebulyObserver(
        api_key="test_api_key",
        project="test_project",
        phase=DevelopmentPhase.EXPERIMENTATION,
        publish=publisher.append,
    )

    patched = _patcher(observer.on_event_received, "module", "0.1.0", "function_name")(
        function
    )
    result = patched(1.0, 2, c=3)

    assert result == 6
    assert len(publisher) == 1
    watched = publisher[0]
    assert watched.called_with_nebuly_kwargs["nebuly_user"] == "undefined"


def test_nebuly_observer_raises_exception_if_invalid_phase() -> None:
    publisher: list[Watched | WatchedEvent] = []
    observer = NebulyObserver(
        api_key="test_api_key",
        project="test_project",
        phase="invalid_phase",  # type: ignore
        publish=publisher.append,
    )
    patched = _patcher(observer.on_event_received, "module", "0.1.0", "function_name")(
        function
    )

    with pytest.raises(ValueError):
        patched(1.0, 2, c=3)


def test_nebuly_observer_raises_exception_if_invalid_phase_override() -> None:
    publisher: list[Watched | WatchedEvent] = []
    observer = NebulyObserver(
        api_key="test_api_key",
        project="test_project",
        phase=DevelopmentPhase.EXPERIMENTATION,
        publish=publisher.append,
    )
    patched = _patcher(observer.on_event_received, "module", "0.1.0", "function_name")(
        function
    )

    with pytest.raises(ValueError) as e:
        patched(1.0, 2, c=3, nebuly_phase="invalid_phase")
    assert str(e.value) == "nebuly_phase must be a DevelopmentPhase"


def test_nebuly_observer_phase_must_be_set() -> None:
    publisher: list[Watched | WatchedEvent] = []
    observer = NebulyObserver(
        api_key="test_api_key",
        project="test_project",
        phase=None,
        publish=publisher.append,
    )
    patched = _patcher(observer.on_event_received, "module", "0.1.0", "function_name")(
        function
    )

    with pytest.raises(ValueError) as e:
        patched(1.0, 2, c=3)
    assert str(e.value) == "nebuly_phase must be set"


def test_nebuly_observer_project_must_be_set() -> None:
    publisher: list[Watched | WatchedEvent] = []
    observer = NebulyObserver(
        api_key="test_api_key",
        project=None,
        phase=DevelopmentPhase.EXPERIMENTATION,
        publish=publisher.append,
    )
    patched = _patcher(observer.on_event_received, "module", "0.1.0", "function_name")(
        function
    )

    with pytest.raises(ValueError) as e:
        patched(1.0, 2, c=3)
    assert str(e.value) == "nebuly_project must be set"
