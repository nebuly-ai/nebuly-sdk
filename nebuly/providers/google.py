# pylint: disable=duplicate-code, wrong-import-position, import-error, no-name-in-module
from __future__ import annotations

import copyreg
import logging
from typing import Any

from google.generativeai.discuss import ChatResponse  # type: ignore
from google.generativeai.text import Completion  # type: ignore
from google.generativeai.types import discuss_types, text_types  # type: ignore

from nebuly.entities import HistoryEntry, ModelInput
from nebuly.providers.utils import get_argument

logger = logging.getLogger(__name__)


class EditedCompletion(Completion):  # type: ignore
    """The Completion class must be overridden to be pickled."""

    def __init__(self, **kwargs: Any):  # pylint: disable=super-init-not-called
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.result = None
        if hasattr(self, "candidates") and self.candidates:
            self.result = self.candidates[0]["output"]


def handle_google_unpickable_objects() -> None:
    def _pickle_google_completion(
        obj: Completion,
    ) -> tuple[type[EditedCompletion], tuple[Any, ...], dict[str, Any]]:
        state = {key: value for key, value in obj.__dict__.items() if key != "_client"}
        return EditedCompletion, (), state

    def _pickle_google_chat(
        obj: ChatResponse,
    ) -> tuple[type[ChatResponse], tuple[Any, ...], dict[str, Any]]:
        state = {key: value for key, value in obj.__dict__.items() if key != "_client"}
        return ChatResponse, (), state

    copyreg.pickle(Completion, _pickle_google_completion)
    copyreg.pickle(ChatResponse, _pickle_google_chat)


def _extract_google_history(
    original_args: tuple[Any, ...],
    original_kwargs: dict[str, Any],
    function_name: str,
) -> list[HistoryEntry]:
    if function_name == "generativeai.discuss.ChatResponse.reply":
        history = getattr(original_args[0], "messages", [])
        history = [message["content"] for message in history]
    else:
        history = original_kwargs.get("messages", [])[:-1]

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


def extract_google_input_and_history(
    original_args: tuple[Any, ...],
    original_kwargs: dict[str, Any],
    function_name: str,
) -> ModelInput:
    if function_name == "generativeai.generate_text":
        return ModelInput(prompt=original_kwargs.get("prompt", ""))
    if function_name in ["generativeai.chat", "generativeai.chat_async"]:
        prompt = original_kwargs.get("messages", [])[-1]
        history = _extract_google_history(original_args, original_kwargs, function_name)
        return ModelInput(prompt=prompt, history=history)
    if function_name == "generativeai.discuss.ChatResponse.reply":
        prompt = get_argument(original_args, original_kwargs, "message", 1)
        history = _extract_google_history(original_args, original_kwargs, function_name)
        return ModelInput(prompt=prompt, history=history)

    raise ValueError(f"Unknown function name: {function_name}")


def extract_google_output(
    function_name: str, output: text_types.Completion | discuss_types.ChatResponse
) -> str:
    if function_name == "generativeai.generate_text":
        return output.result  # type: ignore
    if function_name in [
        "generativeai.chat",
        "generativeai.chat_async",
        "generativeai.discuss.ChatResponse.reply",
    ]:
        return output.messages[-1]["content"]  # type: ignore

    raise ValueError(f"Unknown function name: {function_name}")
