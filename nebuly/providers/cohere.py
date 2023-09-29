from __future__ import annotations

import copyreg
from typing import Any, Callable

from cohere.client import Client
from cohere.responses.chat import Chat, StreamingChat
from cohere.responses.generation import Generations, StreamingGenerations

from nebuly.providers.utils import get_argument


def is_cohere_generator(function: Callable) -> bool:
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


def extract_cohere_input_and_history(
    original_args: tuple[Any],
    original_kwargs: dict[str, Any],
    function_name: str,
) -> tuple[str, list[tuple[str, Any]]]:
    if function_name in ["Client.generate", "AsyncClient.generate"]:
        prompt = get_argument(original_args, original_kwargs, "prompt", 1)
        return prompt, []
    if function_name in ["Client.chat", "AsyncClient.chat"]:
        prompt = get_argument(original_args, original_kwargs, "message", 1)
        chat_history = get_argument(original_args, original_kwargs, "chat_history", 7)
        history = [(el["user_name"], el["message"]) for el in chat_history]
        return prompt, history

    raise ValueError(f"Unknown function name: {function_name}")


def extract_cohere_output(function_name: str, output: Generations | Chat) -> str:
    if function_name in ["Client.generate", "AsyncClient.generate"]:
        return output.generations[0].text
    if function_name in ["Client.chat", "AsyncClient.chat"]:
        return output.text

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
