from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Tuple, cast
from uuid import UUID

from nebuly.entities import (
    EventType,
    HistoryEntry,
    InteractionWatch,
    Observer,
    SpanWatch,
)


@dataclass
class EventData:
    type: EventType
    args: tuple[Any, ...] | None = None
    kwargs: dict[str, Any] | None = None
    output: Any | None = None

    def add_end_event_data(
        self,
        output: Any,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        if kwargs is not None:
            kwargs.pop("parent_run_id")
        if self.args is None:
            self.args = tuple()
        if self.kwargs is None:
            self.kwargs = {}
        self.args += args if args is not None else tuple()
        self.kwargs = dict(self.kwargs, **kwargs) if kwargs is not None else self.kwargs
        self.output = output


@dataclass
class EventHierarchy:
    parent_run_id: UUID
    root_run_id: UUID


@dataclass
class Event:
    event_id: UUID
    hierarchy: EventHierarchy | None
    data: EventData
    module: str
    start_time: datetime
    end_time: datetime | None = None

    def as_span_watch(self) -> SpanWatch:
        if self.end_time is None:
            raise ValueError("Event has not been finished yet.")
        return SpanWatch(
            span_id=self.event_id,
            module=self.module,
            version="unknown",
            function=self._get_function(),
            called_start=self.start_time,
            called_end=self.end_time,
            called_with_args=cast(Tuple[Any], self.data.args),
            called_with_kwargs=cast(Dict[str, Any], self.data.kwargs),
            returned=self.data.output,
            generator=False,
            generator_first_element_timestamp=None,
            provider_extras={
                "event_type": self.data.type.value,
            },
            rag_source=self._get_rag_source(),
        )

    def _get_function(self) -> str:
        if self.data.kwargs is None or len(self.data.kwargs) == 0:
            raise ValueError("Event has no kwargs.")
        if self.data.type is EventType.TOOL:
            return self.data.kwargs["serialized"]["name"]  # type: ignore
        return ".".join(self.data.kwargs["serialized"]["id"])

    def _get_rag_source(self) -> str | None:
        if self.data.kwargs is None or len(self.data.kwargs) == 0:
            raise ValueError("Event has no kwargs.")
        if self.data.type is EventType.TOOL:
            return self.data.kwargs["serialized"]["name"]  # type: ignore
        if self.data.type is EventType.RETRIEVAL:
            return self.data.kwargs["serialized"]["id"][-1]  # type: ignore

        return None

    def set_end_time(self) -> None:
        self.end_time = datetime.now(timezone.utc)


@dataclass
class EventsStorage:
    events: dict[UUID, Event] = field(default_factory=dict)

    def get_root_id(self, event_id: UUID) -> UUID:
        if event_id not in self.events:
            raise ValueError(f"Event {event_id} not found in events events storage.")
        event = self.events[event_id]
        if event.hierarchy is None:
            return event_id
        return self.get_root_id(event.hierarchy.root_run_id)

    def add_event(
        self,
        event_id: UUID,
        parent_id: UUID | None,
        data: EventData,
        module: str,
    ) -> None:
        if event_id in self.events:
            raise ValueError(f"Event {event_id} already exists in events storage.")

        # Handle new event
        if parent_id is None:
            hierarchy = None
        else:
            hierarchy = EventHierarchy(
                parent_run_id=parent_id, root_run_id=self.get_root_id(parent_id)
            )
        self.events[event_id] = Event(
            event_id=event_id,
            hierarchy=hierarchy,
            data=data,
            module=module,
            start_time=datetime.now(timezone.utc),
        )

    def delete_events(self, root_id: UUID) -> None:
        if root_id not in self.events:
            raise ValueError(f"Event {root_id} not found in events hierarchy storage.")

        keys_to_delete = [root_id]

        for event_id, event_detail in self.events.items():
            if (
                event_detail.hierarchy is not None
                and event_detail.hierarchy.root_run_id == root_id
            ):
                keys_to_delete.append(event_id)

        for key in keys_to_delete:
            self.events.pop(key)

    def get_spans(self) -> list[SpanWatch]:
        return [event.as_span_watch() for event in self.events.values()]


class InteractionContextError(Exception):
    pass


class InteractionContextInitiationError(InteractionContextError):
    pass


class AlreadyInInteractionContext(InteractionContextError):
    pass


class NotInInteractionContext(InteractionContextError):
    pass


class InteractionMustBeLocalVariable(InteractionContextError):
    pass


class UserNotFoundError(Exception):
    pass


class UserGroupProfileNotFoundError(Exception):
    pass


class InteractionContext:  # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments
        self,
        user: str | None = None,
        user_group_profile: str | None = None,
        initial_input: str | None = None,
        final_output: str | None = None,
        history: list[HistoryEntry] | None = None,
        hierarchy: dict[UUID, UUID | None] | None = None,
        spans: list[SpanWatch] | None = None,
        do_not_call_directly: bool = False,
    ) -> None:
        if not do_not_call_directly:
            raise InteractionContextInitiationError(
                "Interaction cannot be directly instantiate, use the"
                " 'new_interaction' contextmanager"
            )
        self._events_storage: EventsStorage = EventsStorage()
        self._observer: Observer | None = None
        self._finished: bool = False

        self.user = user
        self.user_group_profile = user_group_profile
        self.input = initial_input
        self.output = final_output
        self.spans = [] if spans is None else spans
        self.history = history
        self.hierarchy = hierarchy
        self.time_start = datetime.now(timezone.utc)

    def set_input(self, value: str) -> None:
        self.input = value

    def set_history(self, value: list[dict[str, str]] | list[HistoryEntry]) -> None:
        if len(value) == 0:
            self.history = []
        elif isinstance(value[0], dict):
            value = cast(List[Dict[str, str]], value)
            history = [
                message
                for message in value
                if message["role"].lower() in ["user", "assistant"]
            ]
            if len(history) % 2 != 0:
                raise ValueError(
                    "Odd number of chat history elements, please provide "
                    "a valid history."
                )
            self.history = [
                HistoryEntry(
                    user=history[i]["content"], assistant=history[i + 1]["content"]
                )
                for i in range(len(history) - 1)
            ]
        else:
            value = cast(List[HistoryEntry], value)
            self.history = value

    def set_output(self, value: str) -> None:
        self.output = value

    def _set_observer(self, observer: Observer) -> None:
        self._observer = observer

    def _add_span(self, value: SpanWatch) -> None:
        self.spans.append(value)

    def _finish(self) -> None:
        self._finished = True
        if len(self._events_storage.events) > 0:
            self.spans += self._events_storage.get_spans()
            self.hierarchy = {
                event.event_id: event.hierarchy.parent_run_id
                if event.hierarchy is not None
                else None
                for event in self._events_storage.events.values()
            }
        else:
            self.hierarchy = {}
        for span in self.spans:
            if span.provider_extras is None:
                continue
            parent_id: UUID | None = span.provider_extras.get("parent_run_id")
            if parent_id is not None:
                self.hierarchy[span.span_id] = parent_id
        if self._observer is not None:
            self._observer(self._as_interaction_watch())

    def _set_user(self, value: str) -> None:
        self.user = value

    def _set_user_group_profile(self, value: str) -> None:
        self.user_group_profile = value

    def _validate_interaction(self) -> None:
        if self.input is None:
            raise ValueError("Interaction has no input.")
        if self.output is None:
            raise ValueError("Interaction has no output.")
        if self.history is None:
            raise ValueError("Interaction has no history.")
        if self.hierarchy is None:
            raise ValueError("Interaction has no hierarchy.")
        if self.user is None:
            raise UserNotFoundError("Interaction has no user.")
        if self.user_group_profile is None:
            raise UserGroupProfileNotFoundError(
                "Interaction has no user group profile."
            )

    def _as_interaction_watch(self) -> InteractionWatch:
        self._validate_interaction()
        return InteractionWatch(
            input=self.input,  # type: ignore
            output=self.output,  # type: ignore
            time_start=self.time_start,
            time_end=datetime.now(timezone.utc),
            spans=self.spans,
            history=self.history,  # type: ignore
            hierarchy=self.hierarchy,  # type: ignore
            end_user=self.user,  # type: ignore
            end_user_group_profile=self.user_group_profile,  # type: ignore
        )


def get_nearest_open_interaction() -> InteractionContext:
    frames = inspect.stack()
    for frame in frames[::-1]:
        for v in frame.frame.f_locals.values():
            if (
                isinstance(v, InteractionContext)
                and not v._finished  # pylint: disable=protected-access
            ):
                return v
    raise NotInInteractionContext()


@contextmanager
def new_interaction(
    user_id: str | None = None, user_group_profile: str | None = None
) -> Generator[InteractionContext, None, None]:
    try:
        get_nearest_open_interaction()
        raise AlreadyInInteractionContext()
    except NotInInteractionContext:
        pass

    try:
        yield InteractionContext(user_id, user_group_profile, do_not_call_directly=True)
    finally:
        try:
            interaction = get_nearest_open_interaction()
        except NotInInteractionContext:
            raise InteractionMustBeLocalVariable()  # pylint: disable=raise-missing-from
        else:
            interaction._finish()  # pylint: disable=protected-access
