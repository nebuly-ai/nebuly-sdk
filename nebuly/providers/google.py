from __future__ import annotations

import copyreg
from typing import Any

from google.generativeai.discuss import ChatResponse
from google.generativeai.text import Completion
from google.generativeai.types import text_types

from nebuly.providers.utils import get_argument


class EditedCompletion(Completion):
    """The Completion class must be overridden to be pickled."""

    def __init__(self, **kwargs: Any):
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


def extract_google_input_and_history(
    original_args: tuple[Any],
    original_kwargs: dict[str, Any],
    function_name: str,
) -> tuple[str, list[tuple[str, Any]]]:
    if function_name == "generativeai.generate_text":
        return original_kwargs.get("prompt"), []
    if function_name in ["generativeai.chat", "generativeai.chat_async"]:
        history = [
            ("user" if i % 2 == 0 else "assistant", el)
            for i, el in enumerate(original_kwargs.get("messages")[:-1])
            if len(original_kwargs.get("messages")) > 1
        ]
        return original_kwargs.get("messages")[-1], history
    if function_name == "generativeai.discuss.ChatResponse.reply":
        prompt = get_argument(original_args, original_kwargs, "message", 1)
        history = [
            ("user" if el["author"] == "0" else "assistant", el["content"])
            for el in getattr(original_args[0], "messages", [])
        ]
        return prompt, history


def extract_google_output(function_name: str, output: text_types.Completion) -> str:
    if function_name == "generativeai.generate_text":
        return output.result
    if function_name in [
        "generativeai.chat",
        "generativeai.chat_async",
        "generativeai.discuss.ChatResponse.reply",
    ]:
        return output.messages[-1]["content"]
