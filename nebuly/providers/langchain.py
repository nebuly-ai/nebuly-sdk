from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Sequence, Tuple, cast
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import Document
from langchain.schema.messages import AIMessage, BaseMessage, HumanMessage
from langchain.schema.output import LLMResult

from nebuly.entities import EventType, HistoryEntry, InteractionWatch, SpanWatch
from nebuly.requests import post_message

logger = logging.getLogger(__name__)


def _parse_langchain_data(data: str | dict[str, Any] | AIMessage) -> str:
    if isinstance(data, dict):
        return "\n".join([f"{key}: {value}" for key, value in data.items()])
    if isinstance(data, AIMessage):
        return data.content
    return data


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

    @property
    def input(self) -> str:
        if self.data.kwargs is None:
            raise ValueError("Event has no kwargs.")
        if self.data.type is EventType.CHAT_MODEL:
            messages = self.data.kwargs.get("messages")
            if messages is None:
                raise ValueError("Event has no messages.")
            return str(messages[-1][-1].content)
        if self.data.type is EventType.LLM_MODEL:
            prompts = self.data.kwargs.get("prompts")
            if prompts is None:
                raise ValueError("Event has no prompts.")
            return str(prompts[-1])
        if self.data.type is EventType.CHAIN:
            inputs = self.data.kwargs.get("inputs")
            if inputs is None:
                raise ValueError("Event has no inputs.")
            return _parse_langchain_data(inputs)

        raise ValueError(f"Event type {self.data.type} not supported.")

    @property
    def output(self) -> str:
        if self.data.output is None:
            raise ValueError("Event has no output.")
        if self.data.type in [EventType.CHAT_MODEL, EventType.LLM_MODEL]:
            return str(self.data.output.generations[-1][0].text)
        if self.data.type is EventType.CHAIN:
            return _parse_langchain_data(self.data.output)

        raise ValueError(f"Event type {self.data.type} not supported.")

    @property
    def history(self) -> list[HistoryEntry]:
        if self.data.type is EventType.CHAT_MODEL:
            if self.data.kwargs is None:
                raise ValueError("Event has no kwargs.")
            messages = self.data.kwargs.get("messages")
            if messages is None:
                raise ValueError("Event has no messages.")
            history = [
                message
                for message in messages[-1][:-1]
                if isinstance(message, (HumanMessage, AIMessage))
            ]
            if len(history) % 2 != 0:
                raise ValueError(
                    "Odd number of chat history elements, please provide "
                    "a valid history."
                )
            if len(history) == 0:
                return []
            return [
                HistoryEntry(user=history[i].content, assistant=history[i + 1].content)
                for i in range(0, len(history) - 1, 2)
            ]
        if self.data.type in [EventType.LLM_MODEL, EventType.CHAIN]:
            return []

        raise ValueError(f"Event type {self.data.type} not supported.")

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


class LangChainTrackingHandler(BaseCallbackHandler):  # noqa
    def __init__(
        self, api_key: str, user_id: str, user_group_profile: str | None = None
    ) -> None:
        self.api_key = api_key
        self.nebuly_user = user_id
        self.nebuly_user_group = user_group_profile
        self._events_storage = EventsStorage()

    def _send_interaction(self, run_id: uuid.UUID) -> None:
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
            spans=self._events_storage.get_spans(),
            hierarchy={
                event.event_id: event.hierarchy.parent_run_id
                if event.hierarchy is not None
                else None
                for event in self._events_storage.events.values()
            },
        )
        post_message(interaction, self.api_key)

    def on_tool_start(  # pylint: disable=arguments-differ
        self,
        serialized: dict[str, Any],
        input_str: str,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        data = EventData(
            type=EventType.TOOL,
            kwargs={
                "serialized": serialized,
                "input_str": input_str,
                **kwargs,
            },
        )
        self._events_storage.add_event(run_id, parent_run_id, data, module="langchain")

    def on_tool_end(  # pylint: disable=arguments-differ
        self,
        output: str,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._events_storage.events[run_id].data.add_end_event_data(
            kwargs=kwargs, output=output
        )
        self._events_storage.events[run_id].set_end_time()

    def on_retriever_start(  # pylint: disable=arguments-differ
        self,
        serialized: dict[str, Any],
        query: str,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        data = EventData(
            type=EventType.RETRIEVAL,
            kwargs={
                "serialized": serialized,
                "query": query,
                **kwargs,
            },
        )
        self._events_storage.add_event(run_id, parent_run_id, data, module="langchain")

    def on_retriever_end(  # pylint: disable=arguments-differ
        self,
        documents: Sequence[Document],
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._events_storage.events[run_id].data.add_end_event_data(
            kwargs=kwargs, output=documents
        )
        self._events_storage.events[run_id].set_end_time()

    def on_llm_start(  # pylint: disable=arguments-differ
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        data = EventData(
            type=EventType.LLM_MODEL,
            kwargs={
                "serialized": serialized,
                "prompts": prompts,
                **kwargs,
            },
        )
        self._events_storage.add_event(run_id, parent_run_id, data, module="langchain")

    def on_llm_end(  # pylint: disable=arguments-differ
        self,
        response: LLMResult,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._events_storage.events[run_id].data.add_end_event_data(
            kwargs=kwargs, output=response
        )
        self._events_storage.events[run_id].set_end_time()

        if len(self._events_storage.events) == 1:
            self._send_interaction(run_id)
            self._events_storage.delete_events(run_id)

    def on_chat_model_start(  # pylint: disable=arguments-differ
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        data = EventData(
            type=EventType.CHAT_MODEL,
            kwargs={
                "serialized": serialized,
                "messages": messages,
                **kwargs,
            },
        )
        self._events_storage.add_event(run_id, parent_run_id, data, module="langchain")

    def on_chain_start(  # pylint: disable=arguments-differ
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        data = EventData(
            type=EventType.CHAIN,
            kwargs={
                "serialized": serialized,
                "inputs": inputs,
                **kwargs,
            },
        )
        self._events_storage.add_event(run_id, parent_run_id, data, module="langchain")

    def on_chain_end(  # pylint: disable=arguments-differ
        self,
        outputs: dict[str, Any],
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._events_storage.events[run_id].data.add_end_event_data(
            kwargs=kwargs, output=outputs
        )
        self._events_storage.events[run_id].set_end_time()

        if self._events_storage.events[run_id].hierarchy is None:
            self._send_interaction(run_id)
            self._events_storage.delete_events(run_id)
