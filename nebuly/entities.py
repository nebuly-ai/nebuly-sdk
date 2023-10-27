from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable


class EventType(Enum):
    """
    The type of event generated by LangChain.
    """

    CHAIN = "chain"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    LLM_MODEL = "llm_model"
    CHAT_MODEL = "chat_model"


@dataclass(frozen=True)
class SupportedVersion:
    min_version: str
    max_version: str | None = None


@dataclass(frozen=True)
class Package:
    """
    Package represents a package to be patched.
    """

    name: str
    versions: SupportedVersion
    to_patch: tuple[str, ...]


@dataclass(frozen=True)
class HistoryEntry:
    """
    HistoryEntry represents a user/assistant interaction in a chat history.
    """

    user: str
    assistant: str

    def to_dict(self) -> list[str]:
        """
        to_dict returns a dictionary representation of the HistoryEntry instance.
        """
        return [self.user, self.assistant]


@dataclass(frozen=True)
class ModelInput:
    prompt: str
    history: list[HistoryEntry] = field(default_factory=list)


@dataclass
class SpanWatch:  # pylint: disable=too-many-instance-attributes
    """
    SpanWatch represents a call to a function that was patched.
    """

    module: str
    version: str
    function: str
    called_start: datetime
    called_end: datetime
    called_with_args: tuple[Any, ...]
    called_with_kwargs: dict[str, Any]
    returned: Any
    generator: bool
    generator_first_element_timestamp: datetime | None
    provider_extras: dict[str, Any] | None = None
    rag_source: str | None = None
    span_id: uuid.UUID = field(default_factory=uuid.uuid4)

    def to_dict(self) -> dict[str, Any]:
        """
        to_dict returns a dictionary representation of the SpanWatch instance.
        """
        return {
            "module": self.module,
            "version": self.version,
            "function": self.function,
            "called_start": self.called_start.isoformat(),
            "called_end": self.called_end.isoformat(),
            "called_with_args": self.called_with_args,
            "called_with_kwargs": self.called_with_kwargs,
            "returned": self.returned,
            "generator": self.generator,
            "generator_first_element_timestamp": (
                self.generator_first_element_timestamp.isoformat()
                if self.generator_first_element_timestamp
                else None
            ),
            "provider_extras": self.provider_extras,
            "rag_source": self.rag_source,
            "span_id": str(self.span_id),
        }


@dataclass
class InteractionWatch:  # pylint: disable=too-many-instance-attributes
    input: str
    output: str
    time_end: datetime
    time_start: datetime
    spans: list[SpanWatch]
    history: list[HistoryEntry]
    hierarchy: dict[uuid.UUID, uuid.UUID | None]
    end_user: str
    end_user_group_profile: str | None

    def to_dict(self) -> dict[str, Any]:
        """
        to_dict returns a dictionary representation of the InteractionWatch instance.
        """
        return {
            "input": self.input,
            "output": self.output,
            "time_end": self.time_end.isoformat(),
            "time_start": self.time_start.isoformat(),
            "spans": [span.to_dict() for span in self.spans],
            "history": self.history,
            "hierarchy": {str(k): str(v) for k, v in self.hierarchy.items()},
            "end_user": self.end_user,
            "end_user_group_profile": self.end_user_group_profile,
        }


Observer = Callable[[InteractionWatch], None]

Publisher = Callable[[InteractionWatch], None]
