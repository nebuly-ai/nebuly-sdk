# pylint: disable=duplicate-code
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Tuple, cast
from uuid import UUID

import llama_index
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
)
from llama_index.core.base.response.schema import Response, StreamingResponse
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    StreamingAgentChatResponse,
)

from nebuly.entities import HistoryEntry, InteractionWatch, SpanWatch
from nebuly.events import Event, EventData, EventsStorage
from nebuly.requests import post_message

orig_download_loader = llama_index.core.readers.download_loader


def download_loader_patched(*args: Any, **kwargs: Any) -> Any:
    res = orig_download_loader(*args, **kwargs)
    orig_load = res.load_data

    def load_data(self: Any, *args: Any, **kwargs: Any) -> Any:
        res = orig_load(self, *args, **kwargs)
        for r in res:
            r.metadata["nebuly_rag_source"] = self.__class__.__name__
        return res

    res.load_data = load_data
    return res


# Patch download_loader to add the source of the RAG node in the metadata
llama_index.core.readers.download_loader = download_loader_patched


def _get_spans(
    events: dict[UUID, Event], trace_map: dict[str, list[str]] | None = None
) -> list[SpanWatch]:
    if trace_map is None:
        event_ids = list(events.keys())
    else:
        values = trace_map.values()
        event_ids = [UUID(item) for sublist in values for item in sublist]
    return [
        event.as_span_watch()
        for event in events.values()
        if event.event_id in event_ids
    ]


@dataclass
class LLamaIndexEvent(Event):
    @property
    def input(self) -> str:
        if self.data.kwargs is None:
            return ""

        input_payload = self.data.kwargs["input_payload"]
        if CBEventType(self.data.type) is CBEventType.QUERY:
            return str(input_payload[EventPayload.QUERY_STR])
        if CBEventType(self.data.type) in [CBEventType.LLM, CBEventType.AGENT_STEP]:
            if EventPayload.MESSAGES in input_payload:
                last_user_input = input_payload[EventPayload.MESSAGES][-1]
                if isinstance(last_user_input, ChatMessage):
                    return str(last_user_input.content)
                return str(last_user_input)
            return str(input_payload[EventPayload.PROMPT])

        return ""

    @property
    def output(self) -> str:
        if self.data.kwargs is None:
            return ""

        if self.data.output:
            return str(self.data.output)
        output = self.data.kwargs["output_payload"][EventPayload.RESPONSE]
        if hasattr(output, "response") and output.response:
            return str(output.response)
        if isinstance(output, StreamingResponse):
            return str(output.get_response().response)

        return ""

    @property
    def history(self) -> list[HistoryEntry]:
        if self.data.kwargs is None:
            return []

        if CBEventType(self.data.type) is CBEventType.LLM:
            input_payload = self.data.kwargs.get("input_payload")
            if input_payload is not None:
                messages: list[ChatMessage] = input_payload.get(EventPayload.MESSAGES)
                if messages is not None:
                    messages = [
                        m
                        for m in messages
                        if m.role in [MessageRole.USER, MessageRole.ASSISTANT]
                    ]
                    if len(messages) % 2 != 0:
                        return [
                            HistoryEntry(
                                user=str(messages[i].content),
                                assistant=str(messages[i + 1].content),
                            )
                            for i in range(len(messages[:-1]) - 1)
                        ]
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
                rag_source: str | None = source_nodes[0].metadata.get(
                    "nebuly_rag_source", source_nodes[0].node_id
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


class LlamaIndexTrackingHandler(
    BaseCallbackHandler
):  # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments
        self,
        api_key: str,
        user_id: str,
        user_group_profile: str | None = None,
        nebuly_tags: dict[str, str] | None = None,
        feature_flags: list[str] | None = None,
    ) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.api_key = api_key
        self.nebuly_user = user_id
        self.nebuly_user_group = user_group_profile
        self.tags = nebuly_tags
        self._events_storage = EventsStorage()

        # Attributes needed for handling streaming responses
        self._waiting_stream_response = False
        self._trace_map: dict[str, list[str]] | None = None
        self._trace_id: str | None = None
        self.nebuly_feature_flags = feature_flags

    @staticmethod
    def _trace_map_to_hierarchy(
        trace_map: dict[str, list[str]] | None, run_id: uuid.UUID
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
        else:
            hierarchy[run_id] = None
        return hierarchy

    def _get_event_history(
        self, trace_id: str | None, event: Event, spans: list[SpanWatch]
    ) -> list[HistoryEntry]:
        if trace_id != "chat" or CBEventType(event.data.type) is CBEventType.LLM:
            return event.history
        # If the event is a chat event, and the event is not a LLM event, we need to
        # find the first event containing the LLM response to get the history
        for span in spans:
            if span.function == "llm":
                llm_event = self._events_storage.events[span.span_id]
                return llm_event.history
        return []

    def _send_interaction(
        self,
        trace_id: str | None,
        root_id: uuid.UUID,
        trace_map: dict[str, list[str]] | None,
        final_stream_event_run_id: uuid.UUID | None = None,
    ) -> None:
        if (
            len(self._events_storage.events) < 1
            or root_id not in self._events_storage.events
        ):
            raise ValueError(f"Event {root_id} not found in events storage.")

        event = self._events_storage.events[root_id]
        if final_stream_event_run_id is not None:
            response = self._events_storage.events[  # type: ignore
                final_stream_event_run_id
            ].data.kwargs["output_payload"][EventPayload.RESPONSE]
            output = response.message.content
        else:
            output = event.output
        spans = _get_spans(events=self._events_storage.events, trace_map=trace_map)
        interaction = InteractionWatch(
            end_user=self.nebuly_user,
            end_user_group_profile=self.nebuly_user_group,
            input=event.input,
            output=output,
            time_start=event.start_time,
            time_end=cast(datetime, event.end_time),
            history=self._get_event_history(trace_id, event, spans),
            spans=spans,
            hierarchy=self._trace_map_to_hierarchy(trace_map=trace_map, run_id=root_id),
            tags=self.tags,
            feature_flags=self.nebuly_feature_flags,
        )
        post_message(interaction, self.api_key)

    def start_trace(self, trace_id: str | None = None) -> None:
        return

    def end_trace(
        self,
        trace_id: str | None = None,
        trace_map: dict[str, list[str]] | None = None,
        final_stream_event_run_id: uuid.UUID | None = None,
    ) -> None:
        if (
            trace_id in ["query", "chat"]
            and trace_map is not None
            and "root" in trace_map
        ):
            valid_root_ids = [
                event_id
                for event_id in trace_map["root"]
                if self._events_storage.events[UUID(event_id)].data.type
                in ["query", "agent_step", "llm"]
            ]
            if len(valid_root_ids) == 0:
                return
            root_id = UUID(valid_root_ids[0])
            if final_stream_event_run_id is not None:
                event = self._events_storage.events[final_stream_event_run_id]
            else:
                event = self._events_storage.events[root_id]
            response = event.data.kwargs.get("output_payload", {}).get(  # type: ignore
                EventPayload.RESPONSE, None
            )
            if response is not None:
                if isinstance(
                    response, (StreamingResponse, StreamingAgentChatResponse)
                ):
                    # When streaming responses are used, the end_trace is called before
                    # the end_event, so we need store some variables to send the
                    # interaction when the end_event will be received later
                    self._waiting_stream_response = True
                    self._trace_map = trace_map
                    self._trace_id = trace_id
                else:
                    self._send_interaction(
                        trace_id,
                        root_id,
                        trace_map=trace_map,
                        final_stream_event_run_id=final_stream_event_run_id,
                    )
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
                constructor=LLamaIndexEvent,
                event_id=UUID(event_id),
                parent_id=(
                    UUID(parent_id) if parent_id and parent_id != "root" else None
                ),
                data=EventData(
                    type=event_type.value,
                    args=tuple(),
                    kwargs={"input_payload": payload, **kwargs},
                ),
                module="llama_index",
            )
        return event_id

    def on_event_end(  # pylint: disable=too-many-branches
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
                if EventPayload.RESPONSE in payload:
                    response = payload[EventPayload.RESPONSE]
                elif EventPayload.COMPLETION in payload:
                    response = payload[EventPayload.COMPLETION]
                else:
                    raise ValueError("Unknown response type.")
                if isinstance(response, (Response, AgentChatResponse)):
                    output = response.response
                elif isinstance(response, ChatResponse):
                    output = response.message.content
                elif isinstance(response, CompletionResponse):
                    output = response.text
                elif isinstance(
                    response, (StreamingResponse, StreamingAgentChatResponse)
                ):
                    output = None
                else:
                    raise ValueError(f"Unknown response type: {type(response)}")
            else:
                output = None

            self._events_storage.events[run_id].data.add_end_event_data(
                kwargs={"output_payload": payload, **kwargs}, output=output
            )
            self._events_storage.events[run_id].set_end_time()

            if event_type is CBEventType.LLM and len(self._events_storage.events) == 1:
                # Track single LLM calls outside of a trace
                self._send_interaction(trace_id=None, root_id=run_id, trace_map=None)
                self._events_storage.delete_events(run_id)

            if self._waiting_stream_response:
                # When streaming responses are used, the end_trace is called before
                # the end_event, so we need to wait for the end_event to send the
                # interaction
                self._waiting_stream_response = False
                self.end_trace(
                    self._trace_id,
                    trace_map=self._trace_map,
                    final_stream_event_run_id=run_id,
                )
                self._trace_map = None
                self._trace_id = None
