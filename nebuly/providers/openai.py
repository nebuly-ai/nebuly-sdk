from __future__ import annotations

import copyreg
from typing import Any, Callable, Generator

import openai
from openai import _ModuleClient
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion


def extract_openai_input_and_history(
    original_kwargs: dict[str, Any],
    function_name: str,
) -> tuple[str, list[tuple[str, Any]]]:
    if function_name in [
        "resources.completions.Completions.create",
        "resources.completions.Completions.acreate",
    ]:
        return original_kwargs.get("prompt"), []
    if function_name in [
        "resources.chat.completions.Completions.create",
        "resources.chat.completions.Completions.acreate",
    ]:
        history = [
            (el["role"], el["content"])
            for el in original_kwargs.get("messages")[:-1]
            if len(original_kwargs.get("messages", [])) > 1
        ]
        return original_kwargs.get("messages")[-1]["content"], history

    raise ValueError(f"Unknown function name: {function_name}")


def extract_openai_output(
    function_name: str, output: ChatCompletion | Completion
) -> str:
    if function_name in [
        "resources.completions.Completions.create",
        "resources.completions.Completions.acreate",
    ]:
        return output.choices[0].text
    if function_name in [
        "resources.chat.completions.Completions.create",
        "resources.chat.completions.Completions.acreate",
    ]:
        return output.choices[0].message.content

    raise ValueError(f"Unknown function name: {function_name}")


def extract_openai_output_generator(
    function_name: str, outputs: Generator[Completion | ChatCompletion, None, None]
) -> str:
    if function_name in [
        "resources.completions.Completions.create",
        "resources.completions.Completions.acreate",
    ]:
        return "".join(
            [getattr(output.choices[0], "text", "") or "" for output in outputs]
        )
    if function_name in [
        "resources.chat.completions.Completions.create",
        "resources.chat.completions.Completions.acreate",
    ]:
        return "".join(
            [
                getattr(output.choices[0].delta, "content", "") or ""
                for output in outputs
            ]
        )

    raise ValueError(f"Unknown function name: {function_name}")


def handle_openai_unpickable_objects():
    def _unpickle_openai_client(kwargs):
        return _ModuleClient(**kwargs)

    def _pickle_openai_client(c: _ModuleClient) -> tuple[Callable, tuple[Any, ...]]:
        return _unpickle_openai_client, (
            {"api_key": c.api_key, "organization": c.organization},
        )

    copyreg.pickle(_ModuleClient, _pickle_openai_client)


def is_openai_generator(obj: Any) -> bool:
    return isinstance(obj, openai.Stream)
