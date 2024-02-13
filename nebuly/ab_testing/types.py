from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Sequence, TypedDict, Union

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


@dataclass
class ConfigParam:
    name: str
    value: Any
    param_type: ConfigParamType

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigParam":
        if "name" not in data or "value" not in data or "param_type" not in data:
            raise ValueError("Could not parse ConfigParam")
        return cls(
            name=data["name"],
            value=data["value"],
            param_type=ConfigParamType(data["param_type"]),
        )


@dataclass
class PromptVariant:
    prompt: str
    config_params: List[ConfigParam]
    kind: Literal[VariantKind.PROMPT] = VariantKind.PROMPT

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptVariant":
        if "prompt" not in data or "config_params" not in data:
            raise ValueError("Could not parse PromptVariant")
        return cls(
            prompt=data["prompt"],
            config_params=[ConfigParam(**param) for param in data["config_params"]],
        )


@dataclass
class ModelVariant:
    model_id: str
    config_params: List[ConfigParam]
    kind: Literal[VariantKind.MODEL] = VariantKind.MODEL

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVariant":
        if "model_id" not in data or "config_params" not in data:
            raise ValueError("Could not parse ModelVariant")
        return cls(
            model_id=data["model_id"],
            config_params=[ConfigParam(**param) for param in data["config_params"]],
        )


@dataclass
class RagSourceVariant:
    rag_source: str
    config_params: List[ConfigParam]
    kind: Literal[VariantKind.RAG_SOURCE] = VariantKind.RAG_SOURCE

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RagSourceVariant":
        if "rag_source" not in data or "config_params" not in data:
            raise ValueError("Could not parse RagSourceVariant")
        return cls(
            rag_source=data["rag_source"],
            config_params=[ConfigParam(**param) for param in data["config_params"]],
        )


Variant: TypeAlias = Union[PromptVariant, ModelVariant, RagSourceVariant]


def variant_from_dict(data: Dict[str, Any]) -> Variant:
    if "kind" not in data:
        raise ValueError("Could not parse variant")
    if data["kind"] == VariantKind.PROMPT:
        return PromptVariant.from_dict(data)
    if data["kind"] == VariantKind.MODEL:
        return ModelVariant.from_dict(data)
    if data["kind"] == VariantKind.RAG_SOURCE:
        return RagSourceVariant.from_dict(data)
    raise ValueError("Could not parse variant")


@dataclass
class LookupVariant:
    feature_flag_name: str
    variant: Variant

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LookupVariant":
        if "feature_flag_name" not in data or "variant" not in data:
            raise ValueError("Could not parse LookupVariant")
        return cls(
            feature_flag_name=data["feature_flag_name"],
            variant=variant_from_dict(data["variant"]),
        )


@dataclass
class Response:
    variants: List[LookupVariant]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Response":
        if "variants" not in data:
            raise ValueError("Could not parse response")
        return cls(
            variants=[LookupVariant.from_dict(variant) for variant in data["variants"]]
        )


class Request(TypedDict):
    user: str
    feature_flags: Sequence[str]
