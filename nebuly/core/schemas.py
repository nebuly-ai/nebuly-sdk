from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class Task(Enum):
    UNDETECTED = "undetected"
    TEXT_GENERATION = "text_generation"
    TEXT_SUMMARIZATION = "text_summarization"
    CHAT = "chat"
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_EDITING = "text_editing"
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDITING = "image_editing"
    IMAGE_VARIATION = "image_variation"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    AUDIO_TRANSLATION = "audio_translation"
    TEXT_EMBEDDING = "text_embedding"
    FINETUNING = "finetuning"
    TEXT_MODERATION = "text_moderation"


class DevelopmentPhase(Enum):
    RESEARCH = "research"
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    UNKNOWN = "unknown"


class Provider(Enum):
    UNKNOWN = "unknown"
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"


@dataclass
class TagData:
    project: Optional[str] = None
    phase: Optional[DevelopmentPhase] = None
    task: Optional[Task] = None


class NebulyDataPackage(BaseModel):
    project: str
    phase: DevelopmentPhase
    task: Task
    timestamp: float

    provider: Optional[Provider] = None
    api_type: Optional[str] = None

    model_name: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    number_of_images: Optional[int] = None
    image_size: Optional[str] = None

    duration_in_seconds: Optional[int] = None

    training_file_id: Optional[str] = None
