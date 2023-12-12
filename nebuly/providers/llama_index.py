import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, cast
from uuid import UUID

import llama_index
from llama_index.callbacks import CBEventType, EventPayload
from llama_index.callbacks.base_handler import BaseCallbackHandler
from llama_index.chat_engine.types import AgentChatResponse, StreamingAgentChatResponse
from llama_index.llms.types import ChatResponse
from llama_index.response.schema import Response, StreamingResponse

from nebuly.entities import EventHierarchy, HistoryEntry, InteractionWatch, SpanWatch
from nebuly.requests import post_message

orig_download_loader = llama_index.download_loader


def download_loader_patched(*args: Any, **kwargs: Any) -> Any:
    res = orig_download_loader(*args, **kwargs)
    orig_load = res.load_data

    def load_data(self: Any, *args: Any, **kwargs: Any) -> Any:
        res = orig_load(self, *args, **kwargs)
        for r in res:
            r.metadata["nebuly_rag_source"] = self.__class__.__name__
        return res

    res.load_data = load_data  # type: ignore[method-assign]
    return res


llama_index.download_loader = download_loader_patched


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
    def input(self) -> str:
        if self.data.kwargs is None:
            return ""

        if CBEventType(self.data.type) is CBEventType.QUERY:
            return str(self.data.kwargs["input_payload"][EventPayload.QUERY_STR])
        elif CBEventType(self.data.type) is CBEventType.LLM:
            return str(self.data.kwargs["input_payload"][EventPayload.PROMPT])
        elif CBEventType(self.data.type) is CBEventType.AGENT_STEP:
            return str(self.data.kwargs["input_payload"][EventPayload.MESSAGES][-1])

        return ""

    @property
    def output(self) -> str:
        if self.data.kwargs is None:
            return ""

        if self.data.output:
            return str(self.data.output)
        output = self.data.kwargs["output_payload"][EventPayload.RESPONSE]
        if output.response:
            return str(output.response)
        if isinstance(output, StreamingResponse):
            return str(output.get_response().response)
        if isinstance(output, StreamingAgentChatResponse):
            output_text = ""
            for token in output.response_gen:
                output_text += token
            return output_text

        return ""

    @property
    def history(self) -> list[HistoryEntry]:
        return []

    def _get_rag_source(self) -> str | None:
        if self.data.kwargs is None:
            return None

        if CBEventType(self.data.type) in [CBEventType.QUERY, CBEventType.RETRIEVE]:
            if CBEventType(self.data.type) is CBEventType.QUERY:
                response = self.data.kwargs["output_payload"][EventPayload.RESPONSE]
                source_nodes = getattr(response, "source_nodes", [])
            else:
                source_nodes = self.data.kwargs["output_payload"][EventPayload.NODES]
            if len(source_nodes) > 0:
                file_name = source_nodes[0].metadata.get("file_name", None)
                if file_name is not None:
                    return str(file_name)
                rag_source: str | None = source_nodes[0].metadata.pop(
                    "nebuly_rag_source", None
                )
                return rag_source
        if CBEventType(self.data.type) is CBEventType.FUNCTION_CALL:
            function_name: str | None = self.data.kwargs["input_payload"][
                EventPayload.TOOL
            ].name
            return function_name

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
            called_end=cast(datetime, self.end_time),
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

    @staticmethod
    def _trace_map_to_hierarchy(
        trace_map: dict[str, list[str]] | None
    ) -> dict[UUID, UUID | None]:
        hierarchy = {}
        if trace_map is not None:
            for parent_id, child_ids in trace_map.items():
                if parent_id == "root":
                    p_id = None
                else:
                    p_id = UUID(parent_id)
                for child_id in child_ids:
                    hierarchy[UUID(child_id)] = p_id
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
        if trace_map is not None and "root" in trace_map:
            root_id = UUID(trace_map["root"][0])
            if trace_id in ["query", "chat"]:
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
        if payload is not None:
            self._events_storage.add_event(
                event_id=UUID(event_id),
                parent_id=UUID(parent_id)
                if parent_id and parent_id != "root"
                else None,
                data=EventData(
                    type=event_type.value,
                    args=tuple(),
                    kwargs={"input_payload": payload, **kwargs},
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
        run_id = UUID(event_id)
        if payload is not None:
            if event_type in [
                CBEventType.QUERY,
                CBEventType.LLM,
                CBEventType.AGENT_STEP,
            ]:
                response = payload[EventPayload.RESPONSE]
                if isinstance(response, (Response, AgentChatResponse)):
                    output = response.response
                elif isinstance(response, ChatResponse):
                    output = response.message.content
                elif isinstance(
                    response, (StreamingResponse, StreamingAgentChatResponse)
                ):
                    output = ""
                else:
                    raise ValueError(f"Unknown response type: {type(response)}")
            else:
                output = None

            self._events_storage.events[run_id].data.add_end_event_data(
                kwargs={"output_payload": payload, **kwargs}, output=output
            )
            self._events_storage.events[run_id].set_end_time()
