# pylint: disable=duplicate-code, import-error, no-name-in-module
# mypy: ignore-errors
from __future__ import annotations

import logging
from typing import Any

from openai.openai_object import OpenAIObject  # type: ignore

from nebuly.entities import HistoryEntry, ModelInput

logger = logging.getLogger(__name__)


def _extract_openai_history(
    original_kwargs: dict[str, Any],
) -> list[HistoryEntry]:
    history = original_kwargs.get("messages", [])[:-1]

    # Remove messages that are not from the user or the assistant
    history = [
        message
        for message in history
        if len(history) > 1 and message["role"] in ["user", "assistant"]
    ]

    if len(history) % 2 != 0:
        logger.warning("Odd number of chat history elements, ignoring last element")
        history = history[:-1]

    # Convert the history to [(user, assistant), ...] format
    history = [
        HistoryEntry(user=history[i]["content"], assistant=history[i + 1]["content"])
        for i in range(0, len(history), 2)
        if i < len(history) - 1
    ]
    return history


def extract_openai_input_and_history(
    original_kwargs: dict[str, Any],
    function_name: str,
) -> ModelInput:
    if function_name in ["Completion.create", "Completion.acreate"]:
        return ModelInput(prompt=original_kwargs.get("prompt", ""))
    if function_name in ["ChatCompletion.create", "ChatCompletion.acreate"]:
        prompt = original_kwargs.get("messages", [])[-1]["content"]
        history = _extract_openai_history(original_kwargs)
        return ModelInput(prompt=prompt, history=history)

    raise ValueError(f"Unknown function name: {function_name}")


def extract_openai_output(function_name: str, output: OpenAIObject) -> str:
    if function_name in ["Completion.create", "Completion.acreate"]:
        return output["choices"][0]["text"]  # type: ignore
    if function_name in ["ChatCompletion.create", "ChatCompletion.acreate"]:
        return output["choices"][0]["message"]["content"]  # type: ignore

    raise ValueError(f"Unknown function name: {function_name}")


def extract_openai_output_generator(
    function_name: str, outputs: list[OpenAIObject]
) -> str:
    if function_name in ["Completion.create", "Completion.acreate"]:
        return "".join([output["choices"][0].get("text", "") for output in outputs])
    if function_name in ["ChatCompletion.create", "ChatCompletion.acreate"]:
        return "".join(
            [output["choices"][0]["delta"].get("content", "") for output in outputs]
        )

    raise ValueError(f"Unknown function name: {function_name}")
