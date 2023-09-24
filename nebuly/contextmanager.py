import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generator
from uuid import UUID

from nebuly.entities import (
    CallbackKwargs,
    EventHierarchy,
    EventType,
    InteractionWatch,
    Observer,
    SpanWatch,
)


@dataclass
class EventData:
    type: EventType
    args: tuple[Any, ...] | None = field(default_factory=tuple)
    kwargs: dict[str, Any] | None = field(default_factory=dict)
    output: Any | None = None

    def add_end_event_data(
        self,
        output: Any,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        if kwargs is not None:
            kwargs.pop("parent_run_id")
        self.args += args if args is not None else tuple()
        self.kwargs = dict(self.kwargs, **kwargs) if kwargs is not None else self.kwargs
        self.output = output


@dataclass
class Event:
    event_id: UUID
    hierarchy: EventHierarchy | None
    data: EventData
    module: str
    start_time: datetime
    end_time: datetime | None = None

    @staticmethod
    def _get_extra_data(
        inputs: dict[str, Any] | None, outputs: dict[str, Any] | None
    ) -> CallbackKwargs | None:
        if inputs is None and outputs is None:
            return None
        if inputs is None:
            return CallbackKwargs(
                inputs={}, outputs=outputs if outputs is not None else {}
            )
        if outputs is None:
            return CallbackKwargs(inputs=inputs, outputs={})
        return CallbackKwargs(inputs=inputs, outputs=outputs)

    def as_span_watch(self) -> SpanWatch:
        return SpanWatch(
            span_id=self.event_id,
            module=self.module,
            version="unknown",
            function=self._get_function(),
            called_start=self.start_time,
            called_end=self.end_time,
            called_with_args=self.data.args,
            called_with_kwargs=self.data.kwargs,
            returned=self.data.output,
            generator=False,
            generator_first_element_timestamp=None,
            rag_source=self._get_rag_source(),
        )

    def _get_function(self) -> str:
        if self.data.type is EventType.TOOL:
            return self.data.kwargs["serialized"]["name"]
        return ".".join(self.data.kwargs["serialized"]["id"])

    def _get_rag_source(self) -> str | None:
        if self.data.type is EventType.TOOL:
            return self.data.kwargs["serialized"]["name"]
        if self.data.type is EventType.RETRIEVAL:
            return self.data.kwargs["serialized"]["id"][-1]

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


class InteractionContext:  # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments
        self,
        user: str | None = None,
        user_group_profile: str | None = None,
        input: str | None = None,
        output: str | None = None,
        history: list[tuple[str, str]] = None,
        hierarchy: dict[UUID, UUID] = None,
        spans: list[SpanWatch] = None,
        do_not_call_directly: bool = False,
    ) -> None:
        if not do_not_call_directly:
            raise InteractionContextInitiationError(
                "Interaction cannot be directly instantiate, use the"
                " 'new_interaction' contextmanager"
            )
        self.events_storage = EventsStorage()
        self.user = user
        self.user_group_profile = user_group_profile
        self.input = input
        self.output = output
        self.spans = [] if spans is None else spans
        self.history = [] if history is None else history
        self.hierarchy = {} if hierarchy is None else hierarchy
        self.time_start = datetime.now(timezone.utc)
        self.observer = None
        self.finished = False

    def set_input(self, value: str) -> None:
        self.input = value

    def set_history(self, value: list[tuple[str, str]]) -> None:
        self.history = value

    def set_output(self, value: str) -> None:
        self.output = value

    def set_observer(self, observer: Observer) -> None:
        if self.observer is None:
            self.observer = observer

    def add_span(self, value: SpanWatch) -> None:
        self.spans.append(value)

    def finish(self) -> None:
        self.finished = True
        if len(self.events_storage.events) > 0:
            self.spans += self.events_storage.get_spans()
            self.hierarchy = {
                event.event_id: event.hierarchy.parent_run_id
                if event.hierarchy is not None
                else None
                for event in self.events_storage.events.values()
            }
        for span in self.spans:
            if span.provider_extras is None:
                continue
            parent_id = span.provider_extras.get("nebuly_parent_run_id")
            if parent_id is not None:
                self.hierarchy[span.span_id] = parent_id
        self.observer(self.as_interaction_watch())

    def set_user(self, value: str) -> None:
        self.user = value

    def set_user_group_profile(self, value: str) -> None:
        self.user_group_profile = value

    def as_interaction_watch(self) -> InteractionWatch:
        return InteractionWatch(
            input=self.input,
            output=self.output,
            time_start=self.time_start,
            time_end=datetime.now(timezone.utc),
            spans=self.spans,
            history=self.history,
            hierarchy=self.hierarchy,
            end_user=self.user,
            end_user_group_profile=self.user_group_profile,
        )


def get_nearest_open_interaction() -> InteractionContext:
    frames = inspect.stack()
    for frame in frames[::-1]:
        for v in frame.frame.f_locals.values():
            if isinstance(v, InteractionContext) and not v.finished:
                return v
    raise NotInInteractionContext()


@contextmanager
def new_interaction(
    user: str | None = None, group_profile: str | None = None
) -> Generator[InteractionContext, None, None]:
    try:
        get_nearest_open_interaction()
        raise AlreadyInInteractionContext()
    except NotInInteractionContext:
        pass

    try:
        yield InteractionContext(user, group_profile, do_not_call_directly=True)
    finally:
        try:
            interaction = get_nearest_open_interaction()
        except NotInInteractionContext:
            raise InteractionMustBeLocalVariable()  # pylint: disable=raise-missing-from
        else:
            interaction.finish()
