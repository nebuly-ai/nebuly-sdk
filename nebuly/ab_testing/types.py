from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Literal, Sequence, TypedDict, Union

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
    config_params: List[ConfigParam]
    kind: Literal[VariantKind.PROMPT] = VariantKind.PROMPT


@dataclass
class ModelVariant:
    model_id: str
    config_params: List[ConfigParam]
    kind: Literal[VariantKind.MODEL] = VariantKind.MODEL


@dataclass
class RagSourceVariant:
    rag_source: str
    config_params: List[ConfigParam]
    kind: Literal[VariantKind.RAG_SOURCE] = VariantKind.RAG_SOURCE


Variant: TypeAlias = Union[PromptVariant, ModelVariant, RagSourceVariant]


@dataclass
class LookupVariant:
    feature_flag_name: str
    variant: Variant


@dataclass
class Response:
    variants: List[LookupVariant]


class Request(TypedDict):
    user: str
    project_id: str
    feature_flags: Sequence[str]
