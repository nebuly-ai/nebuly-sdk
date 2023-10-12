# pylint: disable=duplicate-code, import-error, no-name-in-module
# mypy: ignore-errors
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, cast

import openai
from openai import AsyncOpenAI, OpenAI, _ModuleClient  # type: ignore  # noqa: E501
from openai.types.chat.chat_completion import (  # type: ignore  # noqa: E501
    ChatCompletion,
    Choice,
)
from openai.types.chat.chat_completion_chunk import (  # type: ignore  # noqa: E501
    ChatCompletionChunk,
)
from openai.types.completion import Completion  # type: ignore  # noqa: E501
from openai.types.completion_choice import (  # type: ignore  # noqa: E501
    CompletionChoice,
)

from nebuly.entities import HistoryEntry, ModelInput
from nebuly.providers.base import PicklerHandler, ProviderDataExtractor

logger = logging.getLogger(__name__)


class OpenAIPicklerHandler(PicklerHandler):
    @property
    def _object_key_attribute_mapping(
        self,
    ) -> dict[Callable[[Any], Any], dict[str, list[str]]]:
        names = ["api_key", "organization"]
        return {
            key: {
                "key_names": names,
                "attribute_names": names,
            }
            for key in [OpenAI, AsyncOpenAI, _ModuleClient]
        }


def handle_openai_unpickable_objects() -> None:
    pickler_handler = OpenAIPicklerHandler()
    for obj in [OpenAI, AsyncOpenAI, _ModuleClient]:
        pickler_handler.handle_unpickable_object(
            obj=obj,
        )


def is_openai_generator(obj: Any) -> bool:
    return isinstance(
        obj, (openai.Stream, openai.AsyncStream)  # pylint: disable=no-member
    )


@dataclass(frozen=True)
class OpenAIDataExtractor(ProviderDataExtractor):
    original_args: tuple[Any, ...]
    original_kwargs: dict[str, Any]
    function_name: str

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
        history = [
            HistoryEntry(
                user=history[i]["content"], assistant=history[i + 1]["content"]
            )
            for i in range(0, len(history), 2)
            if i < len(history) - 1
        ]
        return history

    def extract_input_and_history(self) -> ModelInput:
        if self.function_name in [
            "resources.completions.Completions.create",
            "resources.completions.AsyncCompletions.create",
        ]:
            return ModelInput(prompt=self.original_kwargs.get("prompt", ""))
        if self.function_name in [
            "resources.chat.completions.Completions.create",
            "resources.chat.completions.AsyncCompletions.create",
        ]:
            prompt = self.original_kwargs.get("messages", [])[-1]["content"]
            history = self._extract_history()
            return ModelInput(prompt=prompt, history=history)

        raise ValueError(f"Unknown function name: {self.function_name}")

    def extract_output(
        self,
        stream: bool,
        outputs: ChatCompletion
        | Completion
        | list[ChatCompletionChunk]
        | list[Completion],
    ) -> str:
        if isinstance(outputs, list) and stream:
            return self._extract_output_generator(outputs)

        if isinstance(outputs, Completion) and self.function_name in [
            "resources.completions.Completions.create",
            "resources.completions.AsyncCompletions.create",
        ]:
            return cast(CompletionChoice, outputs.choices[0]).text
        if isinstance(outputs, ChatCompletion) and self.function_name in [
            "resources.chat.completions.Completions.create",
            "resources.chat.completions.AsyncCompletions.create",
        ]:
            if cast(Choice, outputs.choices[0]).message.content is not None:
                # Normal chat completion
                return cast(str, cast(Choice, outputs.choices[0]).message.content)
            # Chat completion with function call
            return json.dumps(
                {
                    "function_name": cast(
                        Choice, outputs.choices[0]
                    ).message.function_call.name,
                    "arguments": cast(
                        Choice, outputs.choices[0]
                    ).message.function_call.arguments,
                }
            )

        raise ValueError(
            f"Unknown function name: {self.function_name} or "
            f"output type: {type(outputs)}"
        )

    def _extract_output_generator(
        self, outputs: list[Completion] | list[ChatCompletionChunk]
    ) -> str:
        if all(
            isinstance(output, Completion) for output in outputs
        ) and self.function_name in [
            "resources.completions.Completions.create",
            "resources.completions.AsyncCompletions.create",
        ]:
            return "".join(
                [getattr(output.choices[0], "text", "") or "" for output in outputs]
            )
        if all(
            isinstance(output, ChatCompletionChunk) for output in outputs
        ) and self.function_name in [
            "resources.chat.completions.Completions.create",
            "resources.chat.completions.AsyncCompletions.create",
        ]:
            if not all(
                getattr(output.choices[0].delta, "content") is None
                for output in outputs
            ):
                # Normal chat completion
                return "".join(
                    [
                        getattr(output.choices[0].delta, "content", "") or ""
                        for output in outputs
                    ]
                )
            # Chat completion with function call
            return json.dumps(
                {
                    "function_name": "".join(
                        getattr(output.choices[0].delta.function_call, "name", "") or ""
                        for output in outputs
                    ),
                    "arguments": "".join(
                        getattr(output.choices[0].delta.function_call, "arguments", "")
                        or ""
                        for output in outputs
                    ),
                }
            )

        raise ValueError(
            f"Unknown function name: {self.function_name} "
            f"or output type: {type(outputs)}"
        )
