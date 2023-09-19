import enum
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict

from langchain.schema import Document
from langchain.schema.messages import BaseMessage
from langchain.schema.output import LLMResult
from typing_extensions import Self


class EventType(enum.Enum):
    CHAIN = "chain"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    LLM_MODEL = "llm_model"
    CHAT_MODEL = "chat_model"


@dataclass
class EventData:
    type: EventType
    serialized: dict[str, Any]
    inputs: dict[str, Any]
    input_extras: dict[str, Any] | None
    outputs: dict[str, Any] | None = None
    output_extras: dict[str, Any] | None = None

    @classmethod
    def from_start_event_callback(
        cls,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        event_type: EventType,
        input_extras: dict[str, Any] | None,
    ) -> Self:
        return cls(
            type=event_type,
            serialized=serialized,
            inputs=inputs,
            input_extras=input_extras,
        )

    def set_outputs(
        self, outputs: dict[str, Any], output_extras: dict[str, Any] | None
    ) -> None:
        if output_extras is not None:
            output_extras.pop("parent_run_id")
        self.outputs = outputs
        self.output_extras = output_extras


@dataclass
class EventHierarchy:
    parent_run_id: uuid.UUID
    root_run_id: uuid.UUID


@dataclass
class Event:
    event_id: uuid.UUID
    hierarchy: EventHierarchy | None
    data: EventData


@dataclass
class ExtraData:
    input: dict[str, Any]
    output: dict[str, Any]


@dataclass
class WatchedLangChain:
    run_id: uuid.UUID
    hierarchy: EventHierarchy | None
    type: EventType
    serialized: dict[str, Any]
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    extras: ExtraData | None

    @staticmethod
    def _get_extra_data(
        inputs: dict[str, Any] | None, outputs: dict[str, Any] | None
    ) -> ExtraData | None:
        if inputs is None and outputs is None:
            return None
        if inputs is None:
            return ExtraData(input={}, output=outputs if outputs is not None else {})
        if outputs is None:
            return ExtraData(input=inputs, output={})
        return ExtraData(input=inputs, output=outputs)

    @classmethod
    def from_chain_data(cls, event: Event) -> Self:
        if event.data.outputs is None:
            raise ValueError(
                "Event must have outputs to be converted to WatchedLangChain"
            )
        outputs: dict[str, Any] = event.data.outputs
        return cls(
            run_id=event.event_id,
            hierarchy=event.hierarchy,
            type=event.data.type,
            serialized=event.data.serialized,
            inputs=event.data.inputs,
            outputs=outputs,
            extras=cls._get_extra_data(
                inputs=event.data.input_extras, outputs=event.data.output_extras
            ),
        )


@dataclass
class EventsStorage:
    events: Dict[uuid.UUID, Event] = field(default_factory=dict)

    def get_root_id(self, event_id: uuid.UUID) -> uuid.UUID:
        if event_id not in self.events:
            raise ValueError(f"Event {event_id} not found in events events storage.")
        event = self.events[event_id]
        if event.hierarchy is None:
            return event_id
        return self.get_root_id(event.hierarchy.root_run_id)

    def add_event(
        self, event_id: uuid.UUID, parent_id: uuid.UUID | None, data: EventData
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
        self.events[event_id] = Event(event_id=event_id, hierarchy=hierarchy, data=data)

    def delete_events(self, root_id: uuid.UUID) -> None:
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


@dataclass(frozen=True)
class EventPairingDispatcher:
    events_storage: EventsStorage = field(default_factory=EventsStorage)

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> None:
        data = EventData.from_start_event_callback(
            serialized=serialized,
            inputs=inputs,
            event_type=EventType.CHAIN,
            input_extras=kwargs,
        )
        self.events_storage.add_event(run_id, parent_run_id, data)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> WatchedLangChain:
        self.events_storage.events[run_id].data.set_outputs(outputs, kwargs)
        watched_event = WatchedLangChain.from_chain_data(
            self.events_storage.events[run_id]
        )
        # Delete all events that are part of the chain if the root chain is finished
        if watched_event.hierarchy is None:
            self.events_storage.delete_events(watched_event.run_id)

        # TODO: Publish the event in a queue

        return watched_event

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> None:
        data = EventData.from_start_event_callback(
            serialized=serialized,
            inputs={"query": input_str},
            event_type=EventType.TOOL,
            input_extras=kwargs,
        )
        self.events_storage.add_event(run_id, parent_run_id, data)

    def on_tool_end(
        self,
        output: str,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> WatchedLangChain:
        self.events_storage.events[run_id].data.set_outputs(
            outputs={"result": output}, output_extras=kwargs
        )
        watched_event = WatchedLangChain.from_chain_data(
            self.events_storage.events[run_id]
        )
        # Delete all events that are part of the chain if the root chain is finished
        if watched_event.hierarchy is None:
            self.events_storage.delete_events(watched_event.run_id)

        # TODO: Publish the event in a queue

        return watched_event

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> None:
        data = EventData.from_start_event_callback(
            serialized=serialized,
            inputs={"query": query},
            event_type=EventType.RETRIEVAL,
            input_extras=kwargs,
        )
        self.events_storage.add_event(run_id, parent_run_id, data)

    def on_retriever_end(
        self,
        documents: list[Document],
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> WatchedLangChain:
        self.events_storage.events[run_id].data.set_outputs(
            outputs={"documents": documents}, output_extras=kwargs
        )
        watched_event = WatchedLangChain.from_chain_data(
            self.events_storage.events[run_id]
        )
        # Delete all events that are part of the chain if the root chain is finished
        if watched_event.hierarchy is None:
            self.events_storage.delete_events(watched_event.run_id)

        # TODO: Publish the event in a queue

        return watched_event

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: list[str],
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> None:
        data = EventData.from_start_event_callback(
            serialized=serialized,
            inputs={"prompts": prompts},
            event_type=EventType.LLM_MODEL,
            input_extras=kwargs,
        )
        self.events_storage.add_event(run_id, parent_run_id, data)

    def on_llm_end(
        self,
        response: LLMResult,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> WatchedLangChain:
        self.events_storage.events[run_id].data.set_outputs(
            outputs={"response": response}, output_extras=kwargs
        )
        watched_event = WatchedLangChain.from_chain_data(
            self.events_storage.events[run_id]
        )
        # Delete all events that are part of the chain if the root chain is finished
        if watched_event.hierarchy is None:
            self.events_storage.delete_events(watched_event.run_id)

        # TODO: Publish the event in a queue

        return watched_event

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: list[list[BaseMessage]],
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> None:
        data = EventData.from_start_event_callback(
            serialized=serialized,
            inputs={"messages": messages},
            event_type=EventType.CHAT_MODEL,
            input_extras=kwargs,
        )
        self.events_storage.add_event(run_id, parent_run_id, data)
