from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import UUID

from nebuly.entities import EventHierarchy, HistoryEntry, SpanWatch


@dataclass
class EventData:
    type: Any
    args: tuple[Any, ...] | None = None
    kwargs: dict[str, Any] | None = None
    output: Any | None = None

    def add_end_event_data(
        self,
        output: Any,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        if self.args is None:
            self.args = tuple()
        if self.kwargs is None:
            self.kwargs = {}
        self.args += args if args is not None else tuple()
        self.kwargs = dict(self.kwargs, **kwargs) if kwargs is not None else self.kwargs
        self.output = output


@dataclass
class Event(abc.ABC):
    event_id: UUID
    hierarchy: EventHierarchy | None
    data: EventData
    module: str
    start_time: datetime
    end_time: datetime | None = None
    input_llm: str | None = None

    @property
    @abc.abstractmethod
    def input(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def output(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def history(self) -> list[HistoryEntry]:
        raise NotImplementedError

    @abc.abstractmethod
    def as_span_watch(self) -> SpanWatch:
        raise NotImplementedError

    def set_end_time(self) -> None:
        self.end_time = datetime.now(timezone.utc)


@dataclass
class EventsStorage:
    events: dict[UUID, Event] = field(default_factory=dict)

    def get_root_id(self, event_id: UUID) -> UUID | None:
        if event_id not in self.events:
            return None
        event = self.events[event_id]
        if event.hierarchy is None:
            return event_id
        return self.get_root_id(event.hierarchy.root_run_id)

    def add_event(  # pylint: disable=too-many-arguments
        self,
        constructor: Callable[..., Event],
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
            root_id = self.get_root_id(parent_id)
            if root_id is None:
                hierarchy = None
            else:
                hierarchy = EventHierarchy(parent_run_id=parent_id, root_run_id=root_id)
        self.events[event_id] = constructor(
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

    def update_chain_input(self, prompt: str) -> None:
        """
        This method overrides the input of the chain event
        """
        chain_events = [
            event for event in self.events.values() if event.data.type.value == "chain"
        ]
        if len(chain_events) == 1:
            chain_input = chain_events[0].input

            if (
                isinstance(chain_input, str)
                and len(chain_input) > 0
                and chain_input in prompt
                and chain_input != prompt
            ):
                chain_events[0].input_llm = prompt
