# pylint: disable=duplicate-code, import-error, no-name-in-module
# mypy: ignore-errors
from __future__ import annotations

import copyreg
import json
import logging
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


class AWSBedrockDataExtractor(ProviderDataExtractor):
    def __init__(
        self,
        function_name: str,
        original_args: tuple[Any, ...],
        original_kwargs: dict[str, Any],
    ):
        self.function_name = function_name
        self.original_args = original_args
        self.original_kwargs = original_kwargs
        self.provider = original_args[2]["modelId"].split(".")[0]

    def extract_input_and_history(self, outputs: Any) -> ModelInput:
        if self.function_name == "client.BaseClient._make_api_call":
            if self.provider == "amazon":  # Amazon
                return ModelInput(
                    prompt=json.loads(self.original_args[2]["body"])["inputText"]
                )
            if self.provider == "anthropic":  # Anthropic
                body = json.loads(self.original_args[2]["body"])
                user_input = body["prompt"] if "prompt" in body else body["messages"]
                last_user_input, history = extract_anthropic_input_and_history(
                    user_input
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
            if self.provider == "amazon":
                return response_body["results"][0]["outputText"]
            if self.provider == "cohere":
                return response_body["generations"][0]["text"]
            if self.provider == "anthropic":
                if "completion" in response_body:
                    return response_body["completion"].strip()
                return response_body["content"][0]["text"]
            if self.provider == "ai21":
                return response_body["completions"][0]["data"]["text"]

        raise ValueError(
            f"Unknown function name: {self.function_name} or "
            f"output type: {type(outputs)}"
        )

    def _extract_output_generator(self, outputs: list[dict[str, Any]]) -> str:
        result = ""
        for output in outputs:
            chunk = json.loads(output["chunk"]["bytes"].decode("utf-8"))
            # AI21 does not support streaming
            if self.provider == "amazon":
                result += chunk["outputText"]
            elif self.provider == "cohere":
                # Must check for "text" because it isn't present in the last chunks
                if "text" in chunk and chunk["text"] != "<EOS_TOKEN>":
                    result += chunk["text"]
            elif self.provider == "anthropic":
                result += chunk["completion"]
            else:
                raise ValueError(f"Provider {self.provider} not supported for stream")
        if self.provider == "anthropic":
            result = result.strip()
        return result
