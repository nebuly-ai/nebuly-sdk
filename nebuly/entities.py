from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Union


class DevelopmentPhase(Enum):
    EXPERIMENTATION = "experimentation"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    FINETUNING = "fine-tuning"
    PRODUCTION = "production"
    UNKNOWN = "unknown"


class EventType(Enum):
    CHAIN = "chain"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    LLM_MODEL = "llm_model"
    CHAT_MODEL = "chat_model"


@dataclass(frozen=True)
class Package:
    """
    Package represents a package to be patched.
    """

    name: str
    versions: tuple[str, ...]
    to_patch: tuple[str, ...]


@dataclass
class Watched:  # pylint: disable=too-many-instance-attributes
    """
    Watched represents a call to a function that was patched.
    """

    module: str
    version: str
    function: str
    called_start: datetime
    called_end: datetime
    called_with_args: tuple[Any, ...]
    called_with_kwargs: dict[str, Any]
    called_with_nebuly_kwargs: dict[str, Any]
    returned: Any
    generator: bool
    generator_first_element_timestamp: datetime | None
    provider_extras: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        to_dict returns a dictionary representation of the Watched instance.
        """
        return {
            "module": self.module,
            "version": self.version,
            "function": self.function,
            "called_start": self.called_start.isoformat(),
            "called_end": self.called_end.isoformat(),
            "called_with_args": self.called_with_args,
            "called_with_kwargs": self.called_with_kwargs,
            "called_with_nebuly_kwargs": self.called_with_nebuly_kwargs
            | {"nebuly_phase": self.called_with_nebuly_kwargs["nebuly_phase"].value},
            "returned": self.returned,
            "generator": self.generator,
            "generator_first_element_timestamp": (
                self.generator_first_element_timestamp.isoformat()
                if self.generator_first_element_timestamp
                else None
            ),
            "provider_extras": self.provider_extras,
        }


@dataclass
class EventHierarchy:
    parent_run_id: uuid.UUID
    root_run_id: uuid.UUID


@dataclass
class ExtraData:
    input: dict[str, Any]
    output: dict[str, Any]


@dataclass
class WatchedEvent:  # pylint: disable=too-many-instance-attributes
    module: str
    run_id: uuid.UUID
    hierarchy: EventHierarchy | None
    type: EventType
    serialized: dict[str, Any]
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    extras: ExtraData | None
    called_with_nebuly_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        to_dict returns a dictionary representation of the WatchedEvent instance.
        """
        return {
            "module": self.module,
            "run_id": self.run_id,
            "hierarchy": self.hierarchy,
            "type": self.type.value,
            "serialized": self.serialized,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "extras": self.extras,
            "called_with_nebuly_kwargs": self.called_with_nebuly_kwargs
            | {"nebuly_phase": self.called_with_nebuly_kwargs["nebuly_phase"].value},
        }


Observer_T = Callable[[Watched | WatchedEvent], None]

Publisher_T = Callable[[str], None]
