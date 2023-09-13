from __future__ import annotations

from datetime import datetime

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
        nebuly_user="test_user",
    )

    assert result == 6
    assert len(publisher.messages) == 1
    watched = publisher.messages[0]
    assert watched.called_with_nebuly_kwargs["nebuly_project"] == "other_project"
    assert (
        watched.called_with_nebuly_kwargs["nebuly_phase"]
        == DevelopmentPhase.PREPROCESSING
    )
    assert watched.called_with_nebuly_kwargs["nebuly_user"] == "test_user"


def test_observer_adds_undefine_as_user_if_not_passed():
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
    assert watched.called_with_nebuly_kwargs["nebuly_user"] == "undefined"


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

    with pytest.raises(ValueError) as e:
        patched(1.0, 2, c=3, nebuly_phase="invalid_phase")
    assert str(e.value) == "nebuly_phase must be a DevelopmentPhase"


def test_nebuly_observer_phase_must_be_set():
    publisher = Publisher()
    observer = NebulyObserver(
        api_key="test_api_key",
        project="test_project",
        phase=None,
        publish=publisher.publish,
    )
    patched = _patcher(observer.on_event_received, "module", "0.1.0", "function_name")(
        function
    )

    with pytest.raises(ValueError) as e:
        patched(1.0, 2, c=3)
    assert str(e.value) == "nebuly_phase must be set"


def test_nebuly_observer_project_must_be_set():
    publisher = Publisher()
    observer = NebulyObserver(
        api_key="test_api_key",
        project=None,
        phase=DevelopmentPhase.EXPERIMENTATION,
        publish=publisher.publish,
    )
    patched = _patcher(observer.on_event_received, "module", "0.1.0", "function_name")(
        function
    )

    with pytest.raises(ValueError) as e:
        patched(1.0, 2, c=3)
    assert str(e.value) == "nebuly_project must be set"


def test_add_open_ai_extras():
    # I don't like having this here, we should not transfer openai api key
    publisher = Publisher()
    observer = NebulyObserver(
        api_key="test_api_key",
        project=None,
        phase=DevelopmentPhase.EXPERIMENTATION,
        publish=publisher.publish,
    )

    watched = Watched(
        module="openai",
        version="0.1.0",
        function="function_name",
        called_start=datetime.now(),
        called_end=datetime.now(),
        called_with_args=(),
        called_with_kwargs={},
        called_with_nebuly_kwargs={
            "nebuly_project": "test_project",
            "nebuly_phase": DevelopmentPhase.EXPERIMENTATION,
        },
        returned=None,
        generator=False,
        generator_first_element_timestamp=None,
        provider_extras=None,
    )

    import openai  # pylint: disable=import-outside-toplevel

    api_key = "api_key"
    openai.api_key = api_key

    observer.on_event_received(watched)

    assert watched.provider_extras == {"api_key": api_key}
