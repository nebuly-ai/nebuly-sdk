import enum
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict

from typing_extensions import Self


class EventType(enum.Enum):
    CHAIN = "chain"
    TOOL = "tool"


@dataclass
class EventHierarchy:
    parent_id: uuid.UUID
    root_id: uuid.UUID


@dataclass
class Event:
    event_id: uuid.UUID
    hierarchy: EventHierarchy | None
    data: dict[str, Any]


@dataclass
class WatchedLangChain:
    run_id: uuid.UUID
    parent_run_id: uuid.UUID | None
    root_run_id: uuid.UUID | None
    type: str
    name: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]

    @classmethod
    def from_chain_data(cls, event: Event) -> Self:
        return cls(
            run_id=event.event_id,
            parent_run_id=event.hierarchy.parent_id
            if event.hierarchy is not None
            else None,
            root_run_id=event.hierarchy.root_id
            if event.hierarchy is not None
            else None,
            type=event.data["type"],
            name=event.data["name"],
            inputs=event.data["inputs"],
            outputs=event.data["outputs"],
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
        return self.get_root_id(event.hierarchy.root_id)

    def add_event(
        self, event_id: uuid.UUID, parent_id: uuid.UUID | None, data: dict[str, Any]
    ) -> None:
        if event_id in self.events:
            # Event already exists, add data to it
            self.events[event_id].data.update(data)
            return

        # Handle new event
        if parent_id is None:
            hierarchy = None
        else:
            hierarchy = EventHierarchy(
                parent_id=parent_id, root_id=self.get_root_id(parent_id)
            )
        self.events[event_id] = Event(event_id=event_id, hierarchy=hierarchy, data=data)

    def delete_events(self, root_id: uuid.UUID) -> None:
        if root_id not in self.events:
            raise ValueError(f"Event {root_id} not found in events hierarchy storage.")

        keys_to_delete = [root_id]

        for event_id, event_detail in self.events.items():
            if (
                event_detail.hierarchy is not None
                and event_detail.hierarchy.root_id == root_id
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
    ) -> None:
        data = {
            "type": EventType.CHAIN.value,
            "name": serialized["id"][-1],
            "inputs": inputs,
        }
        self.events_storage.add_event(run_id, parent_run_id, data)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
    ) -> WatchedLangChain:
        data = {
            "outputs": outputs,
        }
        self.events_storage.add_event(run_id, parent_run_id, data)
        watched_event = WatchedLangChain.from_chain_data(
            self.events_storage.events[run_id]
        )
        # Delete all events that are part of the chain if the root chain is finished
        if watched_event.parent_run_id is None:
            self.events_storage.delete_events(watched_event.run_id)

        # TODO: Publish the event in a queue

        return watched_event

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
    ) -> None:
        data = {
            "type": EventType.TOOL.value,
            "name": serialized["name"],
            "inputs": {"input": input_str},
        }
        self.events_storage.add_event(run_id, parent_run_id, data)

    def on_tool_end(
        self,
        output: str,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
    ) -> WatchedLangChain:
        data = {
            "outputs": {"output": output},
        }
        self.events_storage.add_event(run_id, parent_run_id, data)
        watched_event = WatchedLangChain.from_chain_data(
            self.events_storage.events[run_id]
        )
        # Delete all events that are part of the chain if the root chain is finished
        if watched_event.parent_run_id is None:
            self.events_storage.delete_events(watched_event.run_id)

        return watched_event
