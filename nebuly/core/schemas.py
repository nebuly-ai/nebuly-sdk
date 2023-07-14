from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel


class Task(Enum):
    UNKNOWN = "undetected"
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


@dataclass
class TagData:
    project: str = "unknown"
    phase: DevelopmentPhase = DevelopmentPhase.UNKNOWN
    task: Task = Task.UNKNOWN


class GenericProviderAttributes(BaseModel):
    project: str
    phase: DevelopmentPhase
    task: Task
    timestamp: float
    timestamp_end: float


class NebulyDataPackage(BaseModel):
    provider: Provider
    body: GenericProviderAttributes
