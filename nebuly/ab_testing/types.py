from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Sequence, TypedDict, Union

from typing_extensions import TypeAlias


class VariantKind(str, Enum):
    PROMPT = "prompt"
    MODEL = "model"
    RAG_SOURCE = "rag_source"


class ConfigParamType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"


class ConfigParam:
    name: str
    value: Any
    param_type: ConfigParamType


@dataclass
class PromptVariant:
    prompt: str
    config_params: Sequence[ConfigParam]
    kind: Literal[VariantKind.PROMPT] = VariantKind.PROMPT


@dataclass
class ModelVariant:
    model_id: str
    config_params: Sequence[ConfigParam]
    kind: Literal[VariantKind.MODEL] = VariantKind.MODEL


@dataclass
class RagSourceVariant:
    rag_source: str
    config_params: Sequence[ConfigParam]
    kind: Literal[VariantKind.RAG_SOURCE] = VariantKind.RAG_SOURCE


Variant: TypeAlias = Union[PromptVariant, ModelVariant, RagSourceVariant]


@dataclass
class Response:
    variants: Sequence[Variant]


class Request(TypedDict):
    user: str
    project_id: str
    feature_flags: Sequence[str]
