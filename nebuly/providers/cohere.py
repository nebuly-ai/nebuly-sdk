from __future__ import annotations

import copyreg
import logging
from typing import Any, Callable

from cohere.client import Client  # type: ignore
from cohere.responses.chat import Chat, StreamingChat  # type: ignore
from cohere.responses.generation import (  # type: ignore
    Generations,
    StreamingGenerations,
)

from nebuly.entities import HistoryEntry, ModelInput
from nebuly.providers.utils import get_argument

logger = logging.getLogger(__name__)


def is_cohere_generator(function: Callable[[Any], Any]) -> bool:
    return isinstance(function, (StreamingGenerations, StreamingChat))


def handle_cohere_unpickable_objects() -> None:
    def _pickle_cohere_client(c: Client) -> tuple[type[Client], tuple[Any, ...]]:
        return Client, (
            c.api_key,
            c.num_workers,
            c.request_dict,
            True,
            None,
            c.max_retries,
            c.timeout,
            c.api_url,
        )

    copyreg.pickle(Client, _pickle_cohere_client)


def _extract_cohere_history(
    original_args: tuple[Any, ...],
    original_kwargs: dict[str, Any],
) -> list[HistoryEntry]:
    history = get_argument(original_args, original_kwargs, "chat_history", 7)

    # Remove messages that are not from the user or the assistant
    history = [
        message for message in history if message["user_name"] in ["User", "Chatbot"]
    ]

    if len(history) % 2 != 0:
        logger.warning("Odd number of chat history elements, ignoring last element")
        history = history[:-1]

    # Convert the history to [(user, assistant), ...] format
    history = [
        HistoryEntry(user=history[i]["message"], assistant=history[i + 1]["message"])
        for i in range(0, len(history), 2)
        if i < len(history) - 1
    ]
    return history  # type: ignore


def extract_cohere_input_and_history(
    original_args: tuple[Any, ...],
    original_kwargs: dict[str, Any],
    function_name: str,
) -> ModelInput:
    if function_name in ["Client.generate", "AsyncClient.generate"]:
        prompt = get_argument(original_args, original_kwargs, "prompt", 1)
        return ModelInput(prompt=prompt)
    if function_name in ["Client.chat", "AsyncClient.chat"]:
        prompt = get_argument(original_args, original_kwargs, "message", 1)
        history = _extract_cohere_history(original_args, original_kwargs)
        return ModelInput(prompt=prompt, history=history)

    raise ValueError(f"Unknown function name: {function_name}")


def extract_cohere_output(function_name: str, output: Generations | Chat) -> str:
    if function_name in ["Client.generate", "AsyncClient.generate"]:
        return output.generations[0].text  # type: ignore
    if function_name in ["Client.chat", "AsyncClient.chat"]:
        return output.text  # type: ignore

    raise ValueError(f"Unknown function name: {function_name}")


def extract_cohere_output_generator(
    function_name: str, outputs: StreamingGenerations | StreamingChat
) -> str:
    if function_name in ["Client.generate", "AsyncClient.generate"]:
        return "".join([output.text for output in outputs])
    if function_name in ["Client.chat", "AsyncClient.chat"]:
        return "".join(
            [
                output.text
                for output in outputs
                if output.event_type == "text-generation"
            ]
        )

    raise ValueError(f"Unknown function name: {function_name}")
