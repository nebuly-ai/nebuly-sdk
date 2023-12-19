# pylint: disable=duplicate-code, wrong-import-position, import-error, no-name-in-module
from __future__ import annotations

import logging
from typing import Any, Callable, cast

from google.generativeai.discuss import ChatResponse  # type: ignore
from google.generativeai.text import Completion  # type: ignore
from google.generativeai.types import discuss_types, text_types  # type: ignore

from nebuly.entities import HistoryEntry, ModelInput
from nebuly.providers.base import PicklerHandler, ProviderDataExtractor
from nebuly.providers.utils import get_argument

logger = logging.getLogger(__name__)


class GooglePicklerHandler(PicklerHandler):
    @property
    def _object_key_attribute_mapping(
        self,
    ) -> dict[Callable[[Any], Any], dict[str, list[str]]]:
        completion_names = ["candidates", "result", "filters", "safety_feedback"]
        chat_names = [
            "model",
            "context",
            "examples",
            "messages",
            "temperature",
            "candidate_count",
            "candidates",
            "filters",
            "top_p",
            "top_k",
        ]

        return {
            Completion: {
                "key_names": completion_names,
                "attribute_names": completion_names,
            },
            ChatResponse: {
                "key_names": chat_names,
                "attribute_names": chat_names,
            },
        }


def handle_google_unpickable_objects() -> None:
    pickler_handler = GooglePicklerHandler()
    for obj in [Completion, ChatResponse]:
        pickler_handler.handle_unpickable_object(
            obj=obj,
        )


class GoogleDataExtractor(ProviderDataExtractor):
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
        if self.function_name == "generativeai.discuss.ChatResponse.reply":
            history = getattr(self.original_args[0], "messages", [])
            history = [message["content"] for message in history]
        else:
            history = self.original_kwargs.get("messages", [])[:-1]

        if len(history) % 2 != 0:
            logger.warning("Odd number of chat history elements, ignoring last element")
            history = history[:-1]

        # Convert the history to [(user, assistant), ...] format
        history = [
            HistoryEntry(user=history[i], assistant=history[i + 1])
            for i in range(0, len(history), 2)
            if i < len(history) - 1
        ]

        return history

    def extract_input_and_history(self, outputs: Any) -> ModelInput:
        if self.function_name == "generativeai.generate_text":
            return ModelInput(prompt=self.original_kwargs.get("prompt", ""))
        if self.function_name in ["generativeai.chat", "generativeai.chat_async"]:
            prompt = self.original_kwargs.get("messages", [])[-1]
            history = self._extract_history()
            return ModelInput(prompt=prompt, history=history)
        if self.function_name == "generativeai.discuss.ChatResponse.reply":
            prompt = get_argument(
                self.original_args, self.original_kwargs, "message", 1
            )
            history = self._extract_history()
            return ModelInput(prompt=prompt, history=history)

        raise ValueError(f"Unknown function name: {self.function_name}")

    def extract_output(
        self, stream: bool, outputs: text_types.Completion | discuss_types.ChatResponse
    ) -> str:
        if stream:
            return self._extract_output_generator(outputs)
        if (
            isinstance(outputs, text_types.Completion)
            and self.function_name == "generativeai.generate_text"
        ):
            return cast(str, outputs.result)
        if isinstance(outputs, discuss_types.ChatResponse) and self.function_name in [
            "generativeai.chat",
            "generativeai.chat_async",
            "generativeai.discuss.ChatResponse.reply",
        ]:
            return cast(str, outputs.messages[-1]["content"])

        raise ValueError(
            f"Unknown function name: {self.function_name} "
            f"or output type: {type(outputs)}"
        )

    def _extract_output_generator(self, outputs: Any) -> str:
        raise NotImplementedError("Google does not support streaming yet.")
