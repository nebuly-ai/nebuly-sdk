from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from anthropic import Anthropic, AsyncAnthropic, AsyncStream, Stream
from anthropic.types import Completion

from nebuly.entities import ModelInput
from nebuly.providers.base import PicklerHandler, ProviderDataExtractor
from nebuly.providers.common import extract_anthropic_input_and_history


def is_anthropic_generator(function: Callable[[Any], Any]) -> bool:
    return isinstance(function, (Stream, AsyncStream))


class AnthropicPicklerHandler(PicklerHandler):
    @property
    def _object_key_attribute_mapping(
        self,
    ) -> dict[Callable[[Any], Any], dict[str, list[str]]]:
        names = [
            "auth_token",
            "base_url",
            "api_key",
            "timeout",
            "max_retries",
            "default_headers",
        ]
        return {
            key: {  # type: ignore
                "key_names": names,
                "attribute_names": names,
            }
            for key in [Anthropic, AsyncAnthropic]
        }


def handle_anthropic_unpickable_objects() -> None:
    pickler_handler = AnthropicPicklerHandler()
    for obj in [Anthropic, AsyncAnthropic]:
        pickler_handler.handle_unpickable_object(
            obj=obj,  # type: ignore
        )


@dataclass(frozen=True)
class AnthropicDataExtractor(ProviderDataExtractor):
    original_args: tuple[Any, ...]
    original_kwargs: dict[str, Any]
    function_name: str

    def extract_input_and_history(self) -> ModelInput:
        if self.function_name in [
            "resources.Completions.create",
            "resources.AsyncCompletions.create",
        ]:
            last_user_input, history = extract_anthropic_input_and_history(
                self.original_kwargs["prompt"]
            )
            return ModelInput(prompt=last_user_input, history=history)

        raise ValueError(f"Unknown function name: {self.function_name}")

    def extract_output(
        self,
        stream: bool,
        outputs: Completion | list[Completion],
    ) -> str:
        if stream and isinstance(outputs, list):
            return self._extract_output_generator(outputs)
        if isinstance(outputs, Completion) and self.function_name in [
            "resources.Completions.create",
            "resources.AsyncCompletions.create",
        ]:
            return outputs.completion.strip()

        raise ValueError(
            f"Unknown function name: {self.function_name} or "
            f"output type: {type(outputs)}"
        )

    def _extract_output_generator(self, outputs: list[Completion]) -> str:
        if all(
            isinstance(output, Completion) for output in outputs
        ) and self.function_name in [
            "resources.Completions.create",
            "resources.AsyncCompletions.create",
        ]:
            return "".join([output.completion for output in outputs]).strip()

        raise ValueError(
            f"Unknown function name: {self.function_name} or "
            f"output type: {type(outputs)}"
        )
