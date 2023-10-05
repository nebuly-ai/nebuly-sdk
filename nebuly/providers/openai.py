# pylint: disable=duplicate-code, import-error, no-name-in-module
from __future__ import annotations

import copyreg
from typing import Any, Callable, cast

import openai
from openai import AsyncOpenAI, OpenAI, _ModuleClient  # mypy: ignore-errors
from openai.types.chat.chat_completion import (  # mypy: ignore-errors
    ChatCompletion,
    Choice,
)
from openai.types.completion import Completion  # mypy: ignore-errors
from openai.types.completion_choice import CompletionChoice  # mypy: ignore-errors


def extract_openai_input_and_history(
    original_kwargs: dict[str, Any],
    function_name: str,
) -> tuple[str, list[tuple[str, Any]]]:
    if function_name in [
        "resources.completions.Completions.create",
        "resources.completions.AsyncCompletions.create",
    ]:
        return original_kwargs.get("prompt", ""), []
    if function_name in [
        "resources.chat.completions.Completions.create",
        "resources.chat.completions.AsyncCompletions.create",
    ]:
        history = [
            (el["role"], el["content"])
            for el in original_kwargs.get("messages", [])[:-1]
            if len(original_kwargs.get("messages", [])) > 1
        ]
        return original_kwargs.get("messages", [])[-1]["content"], history

    raise ValueError(f"Unknown function name: {function_name}")


def extract_openai_output(
    function_name: str, output: ChatCompletion | Completion
) -> str:
    if function_name in [
        "resources.completions.Completions.create",
        "resources.completions.AsyncCompletions.create",
    ]:
        return cast(CompletionChoice, output.choices[0]).text
    if function_name in [
        "resources.chat.completions.Completions.create",
        "resources.chat.completions.AsyncCompletions.create",
    ]:
        return cast(str, cast(Choice, output.choices[0]).message.content)

    raise ValueError(f"Unknown function name: {function_name}")


def extract_openai_output_generator(
    function_name: str, outputs: list[Completion | ChatCompletion]
) -> str:
    if function_name in [
        "resources.completions.Completions.create",
        "resources.completions.AsyncCompletions.create",
    ]:
        return "".join(
            [getattr(output.choices[0], "text", "") or "" for output in outputs]
        )
    if function_name in [
        "resources.chat.completions.Completions.create",
        "resources.chat.completions.AsyncCompletions.create",
    ]:
        return "".join(
            [
                getattr(output.choices[0].delta, "content", "") or ""  # type: ignore
                for output in outputs
            ]
        )

    raise ValueError(f"Unknown function name: {function_name}")


def handle_openai_unpickable_objects() -> None:
    def _unpickle_openai_client(
        constructor: Callable[[Any], Any], kwargs: dict[str, Any]
    ) -> Any:
        return constructor(**kwargs)  # type: ignore

    def _pickle_openai_client(
        c: _ModuleClient | OpenAI | AsyncOpenAI,
    ) -> Any:
        if isinstance(c, AsyncOpenAI):
            class_constructor = AsyncOpenAI
        elif isinstance(c, OpenAI):
            class_constructor = OpenAI  # type: ignore
        else:
            class_constructor = _ModuleClient
        return _unpickle_openai_client, (
            class_constructor,
            {"api_key": c.api_key, "organization": c.organization},
        )

    copyreg.pickle(_ModuleClient, _pickle_openai_client)
    copyreg.pickle(OpenAI, _pickle_openai_client)
    copyreg.pickle(AsyncOpenAI, _pickle_openai_client)


def is_openai_generator(obj: Any) -> bool:
    return isinstance(
        obj, (openai.Stream, openai.AsyncStream)
    )  # mypy: ignore-errors  # pylint: disable=no-member  # noqa: E501
