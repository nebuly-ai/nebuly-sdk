import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, cast
from uuid import UUID

from llama_index.callbacks import CBEventType, EventPayload
from llama_index.callbacks.base_handler import BaseCallbackHandler
from llama_index.llms.base import ChatResponse
from llama_index.response.schema import Response

from nebuly.entities import EventHierarchy, HistoryEntry, InteractionWatch, SpanWatch
from nebuly.requests import post_message


@dataclass
class EventData:
    type: str
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
class Event:
    event_id: UUID
    hierarchy: EventHierarchy | None
    data: EventData
    start_time: datetime
    end_time: datetime | None = None

    @property
    def input(self) -> str | None:
        if CBEventType(self.data.type) is CBEventType.QUERY:
            return self.data.kwargs["payload"][EventPayload.QUERY_STR]
        elif CBEventType(self.data.type) is CBEventType.LLM:
            return self.data.kwargs["payload"][EventPayload.PROMPT]
        else:
            return None

    @property
    def output(self) -> str | None:
        return self.data.output

    @property
    def history(self) -> list[HistoryEntry]:
        return []

    def _get_rag_source(self) -> str | None:
        return None

    def as_span_watch(self) -> SpanWatch:
        if self.end_time is None:
            self.set_end_time()
        return SpanWatch(
            span_id=self.event_id,
            module="llama_index",
            version="unknown",
            function=self.data.type,
            called_start=self.start_time,
            called_end=self.end_time,
            called_with_args=cast(Tuple[Any], self.data.args),
            called_with_kwargs=cast(Dict[str, Any], self.data.kwargs),
            returned=self.data.output,
            generator=False,
            generator_first_element_timestamp=None,
            provider_extras={
                "event_type": self.data.type,
            },
            rag_source=self._get_rag_source(),
        )

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

    def add_event(
        self,
        event_id: UUID,
        parent_id: UUID | None,
        data: EventData,
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
        self.events[event_id] = Event(
            event_id=event_id,
            hierarchy=hierarchy,
            data=data,
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

    def get_spans(
        self, trace_map: dict[str, list[str]] | None = None
    ) -> list[SpanWatch]:
        if trace_map is None:
            raise ValueError("Trace map cannot be None.")
        values = trace_map.values()
        event_ids = [UUID(item) for sublist in values for item in sublist]
        return [
            event.as_span_watch()
            for event in self.events.values()
            if event.event_id in event_ids
        ]


class NebulyTrackingHandler(BaseCallbackHandler):
    def __init__(
        self,
        api_key: str,
        user_id: str,
        user_group_profile: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.api_key = api_key
        self.nebuly_user = user_id
        self.nebuly_user_group = user_group_profile
        self.tags = tags
        self._events_storage = EventsStorage()

    def _trace_map_to_hierarchy(
        self, trace_map: dict[str, list[str]] | None
    ) -> dict[UUID, UUID | None]:
        hierarchy = {}
        for parent_id, child_ids in trace_map.items():
            if parent_id == "root":
                parent_id = None
            else:
                parent_id = UUID(parent_id)
            for child_id in child_ids:
                hierarchy[UUID(child_id)] = parent_id
        return hierarchy

    def _send_interaction(
        self, run_id: uuid.UUID, trace_map: dict[str, list[str]] | None
    ) -> None:
        if (
            len(self._events_storage.events) < 1
            or run_id not in self._events_storage.events
        ):
            raise ValueError(f"Event {run_id} not found in events storage.")

        interaction = InteractionWatch(
            end_user=self.nebuly_user,
            end_user_group_profile=self.nebuly_user_group,
            input=self._events_storage.events[run_id].input,
            output=self._events_storage.events[run_id].output,
            time_start=self._events_storage.events[run_id].start_time,
            time_end=cast(datetime, self._events_storage.events[run_id].end_time),
            history=self._events_storage.events[run_id].history,
            spans=self._events_storage.get_spans(trace_map=trace_map),
            hierarchy=self._trace_map_to_hierarchy(trace_map=trace_map),
            tags=self.tags,
        )
        post_message(interaction, self.api_key)

    def start_trace(self, trace_id: str | None = None) -> None:
        return

    def end_trace(
        self,
        trace_id: str | None = None,
        trace_map: dict[str, list[str]] | None = None,
    ) -> None:
        if trace_id in ["query"]:
            root_id = UUID(trace_map["root"][0])
            self._send_interaction(root_id, trace_map=trace_map)
            self._events_storage.delete_events(root_id)

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        print(f"Event type: {event_type}")
        if payload is not None:
            self._events_storage.add_event(
                event_id=UUID(event_id),
                parent_id=UUID(parent_id)
                if parent_id and parent_id != "root"
                else None,
                data=EventData(
                    type=event_type.value,
                    args=tuple(),
                    kwargs={"payload": payload, **kwargs},
                ),
            )
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        print(f"Event type: {event_type}")
        run_id = UUID(event_id)
        if event_type in [CBEventType.QUERY, CBEventType.LLM]:
            response = payload[EventPayload.RESPONSE]
            if isinstance(response, Response):
                output = response.response
            elif isinstance(response, ChatResponse):
                output = response.message.content
            else:
                raise ValueError(f"Unknown response type: {type(response)}")
        else:
            output = None
        if payload is not None:
            self._events_storage.events[run_id].data.add_end_event_data(
                kwargs=kwargs, output=output
            )
            self._events_storage.events[run_id].set_end_time()
