from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable


class Task(Enum):
    UNKNOWN = "undetected"  # XXX: should be "unknown"?
    TEXT_GENERATION = "text_generation"
    TEXT_SUMMARIZATION = "text_summarization"
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_EMBEDDING = "text_embedding"
    TEXT_EDITING = "text_editing"
    TEXT_MODERATION = "text_moderation"
    CHAT = "chat"
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDITING = "image_editing"
    IMAGE_VARIATION = "image_variation"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    AUDIO_TRANSLATION = "audio_translation"
    FINETUNING = "fine-tuning"


class DevelopmentPhase(Enum):
    EXPERIMENTATION = "experimentation"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    FINETUNING = "fine-tuning"
    PRODUCTION = "production"
    UNKNOWN = "unknown"


class Provider(Enum):
    UNKNOWN = "unknown"
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"


@dataclass(frozen=True, slots=True)
class Package:
    """
    Package represents a package to be patched.
    """

    name: str
    versions: tuple[str, ...]
    to_patch: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Watched:  # pylint: disable=too-many-instance-attributes
    """
    Watched represents a call to a function that was patched.
    """

    module: str
    function: str
    called_start: datetime
    called_end: datetime
    called_with_args: tuple
    called_with_kwargs: dict[str, Any]
    called_with_nebuly_kwargs: dict[str, Any]
    returned: Any
    generator: bool
    generator_first_element_timestamp: datetime | None

    def to_dict(self) -> dict[str, Any]:
        """
        to_dict returns a dictionary representation of the Watched instance.
        """
        return {
            "module": self.module,
            "function": self.function,
            "called_start": self.called_start.isoformat(),
            "called_end": self.called_end.isoformat(),
            "called_with_args": self.called_with_args,
            "called_with_kwargs": self.called_with_kwargs,
            "called_with_nebuly_kwargs": self.called_with_nebuly_kwargs,
            "returned": self.returned,
            "generator": self.generator,
            "generator_first_element_timestamp": (
                self.generator_first_element_timestamp.isoformat()
                if self.generator_first_element_timestamp
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class Message:
    """
    Message represents a message to be sent to the API.
    """

    api_key: str
    watched: Watched

    def to_dict(self) -> dict[str, Any]:
        """
        to_dict returns a dictionary representation of the Message instance.
        """
        return {
            "api_key": self.api_key,
            "watched": self.watched.to_dict(),
        }


Observer_T = Callable[[Watched], None]

Publisher_T = Callable[[Message], None]
