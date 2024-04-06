from __future__ import annotations

from typing import Any, Callable

from anthropic import (
    Anthropic,
    AsyncAnthropic,
    AsyncStream,
    MessageStreamManager,
    Stream,
)
from anthropic.types import Completion, Message

from nebuly.entities import HistoryEntry, ModelInput
from nebuly.providers.base import PicklerHandler, ProviderDataExtractor
from nebuly.providers.common import extract_anthropic_input_and_history, logger


def is_anthropic_generator(function: Callable[[Any], Any]) -> bool:
    return isinstance(function, (Stream, AsyncStream, MessageStreamManager))


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
            key: {
                "key_names": names,
                "attribute_names": names,
            }
            for key in [Anthropic, AsyncAnthropic]
        }


def handle_anthropic_unpickable_objects() -> None:
    pickler_handler = AnthropicPicklerHandler()
    for obj in [Anthropic, AsyncAnthropic]:
        pickler_handler.handle_unpickable_object(
            obj=obj,
        )


class AnthropicDataExtractor(ProviderDataExtractor):
    def __init__(
        self,
        function_name: str,
        original_args: tuple[Any, ...],
        original_kwargs: dict[str, Any],
    ):
        self.function_name = function_name
        self.original_args = original_args
        self.original_kwargs = original_kwargs

    def _extract_history(self) -> list[HistoryEntry]:
        history = self.original_kwargs.get("messages", [])[:-1]

        # Remove messages that are not from the user or the assistant
        history = [
            message
            for message in history
            if len(history) > 1 and message["role"].lower() in ["user", "assistant"]
        ]

        if len(history) % 2 != 0:
            logger.warning("Odd number of chat history elements, ignoring last element")
            history = history[:-1]

        # Convert the history to [(user, assistant), ...] format
        history_processed = [
            HistoryEntry(
                user=history[i]["content"],
                assistant=history[i + 1]["content"],
            )
            for i in range(0, len(history), 2)
            if i < len(history) - 1
        ]
        return history_processed

    def extract_input_and_history(self, outputs: Any) -> ModelInput:
        if self.function_name in [
            "resources.Completions.create",
            "resources.AsyncCompletions.create",
        ]:
            last_user_input, history = extract_anthropic_input_and_history(
                self.original_kwargs["prompt"]
            )
            return ModelInput(prompt=last_user_input, history=history)
        if self.function_name in [
            "resources.Messages.create",
            "resources.AsyncMessages.create",
            "resources.Messages.stream",
            "resources.AsyncMessages.stream",
        ]:
            prompt = self.original_kwargs["messages"][-1]["content"]
            history = self._extract_history()
            return ModelInput(prompt=prompt, history=history)

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
        if isinstance(outputs, Message) and self.function_name in [
            "resources.Messages.create",
            "resources.AsyncMessages.create",
        ]:
            return outputs.content[0].text.strip()

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
