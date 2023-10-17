# pylint: disable=duplicate-code, import-error, no-name-in-module
# mypy: ignore-errors
from __future__ import annotations

import copyreg
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from botocore.eventstream import EventStream
from botocore.response import StreamingBody

from nebuly.entities import ModelInput
from nebuly.providers.base import PicklerHandler, ProviderDataExtractor
from nebuly.providers.common import extract_anthropic_input_and_history

logger = logging.getLogger(__name__)


def is_aws_bedrock_generator(obj: Any) -> bool:
    return isinstance(obj["body"], EventStream)


def is_model_supported(model_id: str) -> bool:
    if model_id.startswith("stability") or "-embed-" in model_id:
        return False
    return True


class AWSBedrockPicklerHandler(PicklerHandler):
    @property
    def _object_key_attribute_mapping(
        self,
    ) -> dict[Callable[[Any], Any], dict[str, list[str]]]:
        return {}

    def _pickle_object(self, obj: Any) -> Any:
        # Ignore the BotocoreRuntime object
        return lambda x: x, (None,)


def handle_aws_bedrock_unpickable_objects(obj: Any) -> None:
    pickler_handler = AWSBedrockPicklerHandler()
    pickler_handler.handle_unpickable_object(
        obj=obj.__class__,
    )

    def extract_aws_streaming_body(obj: StreamingBody) -> Any:  # type: ignore
        def new_read():
            return obj._raw_stream.data  # pylint: disable=protected-access

        data = obj._raw_stream.data  # pylint: disable=protected-access
        obj.read = new_read
        return lambda x: x, (data,)

    copyreg.pickle(StreamingBody, extract_aws_streaming_body)


@dataclass(frozen=True)
class AWSBedrockDataExtractor(ProviderDataExtractor):
    original_args: tuple[Any, ...]
    original_kwargs: dict[str, Any]
    function_name: str

    def extract_input_and_history(self) -> ModelInput:
        if self.function_name == "client.BaseClient._make_api_call":
            model_id = self.original_args[2]["modelId"]
            if model_id.startswith("amazon"):  # Amazon
                return ModelInput(
                    prompt=json.loads(self.original_args[2]["body"])["inputText"]
                )
            if model_id.startswith("anthropic"):  # Anthropic
                last_user_input, history = extract_anthropic_input_and_history(
                    json.loads(self.original_args[2]["body"])["prompt"]
                )
                return ModelInput(prompt=last_user_input, history=history)
            # Cohere and AI21
            return ModelInput(
                prompt=json.loads(self.original_args[2]["body"])["prompt"]
            )

        raise ValueError(f"Unknown function name: {self.function_name}")

    def extract_output(
        self, stream: bool, outputs: dict[str, Any] | list[dict[str, Any]]
    ) -> str:
        if isinstance(outputs, list) and stream:
            return self._extract_output_generator(outputs)

        if (
            isinstance(outputs, dict)
            and self.function_name == "client.BaseClient._make_api_call"
        ):
            response_body = json.loads(outputs["body"].decode("utf-8"))
            if "results" in response_body:  # Amazon
                return response_body["results"][0]["outputText"]
            if "generations" in response_body:  # Cohere
                return response_body["generations"][0]["text"]
            if "completion" in response_body:  # Anthropic
                return response_body["completion"]
            if "completions" in response_body:  # AI21
                return response_body["completions"][0]["data"]["text"]

        raise ValueError(
            f"Unknown function name: {self.function_name} or "
            f"output type: {type(outputs)}"
        )

    @staticmethod
    def _extract_output_generator(outputs: list[dict[str, Any]]) -> str:
        result = ""
        for output in outputs:
            chunk = json.loads(output["chunk"]["bytes"].decode("utf-8"))
            if "outputText" in chunk:  # Amazon
                result += chunk["outputText"]
            elif "text" in chunk:  # Cohere
                result += chunk["text"]
            elif "completion" in chunk:  # Anthropic
                result += chunk["completion"]
            else:
                raise ValueError(f"Unknown output dict: {chunk}")
        return result
