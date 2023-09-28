from __future__ import annotations

import copyreg
from typing import Any, Callable

from anthropic import Anthropic, AsyncAnthropic, AsyncStream, Stream
from anthropic.types import Completion


def is_anthropic_generator(function: Callable) -> bool:
    return isinstance(function, (Stream, AsyncStream))


def handle_anthropic_unpickable_objects() -> None:
    def _pickle_anthropic_client(
        c: Anthropic,
    ) -> tuple[type[Anthropic], tuple[Any, ...], dict[str, Any]]:
        return (
            Anthropic,
            (),
            {
                "auth_token": c.auth_token,
                "base_url": c.base_url,
                "api_key": c.api_key,
                "timeout": c.timeout,
                "max_retries": c.max_retries,
                "default_headers": c.default_headers,
            },
        )

    def _pickle_async_anthropic_client(
        c: AsyncAnthropic,
    ) -> tuple[type[AsyncAnthropic], tuple[Any, ...], dict[str, Any]]:
        return (
            AsyncAnthropic,
            (),
            {
                "auth_token": c.auth_token,
                "base_url": c.base_url,
                "api_key": c.api_key,
                "timeout": c.timeout,
                "max_retries": c.max_retries,
                "default_headers": c.default_headers,
            },
        )

    copyreg.pickle(Anthropic, _pickle_anthropic_client)
    copyreg.pickle(AsyncAnthropic, _pickle_async_anthropic_client)


def extract_anthropic_input_and_history(
    original_kwargs: dict[str, Any],
    function_name: str,
) -> tuple[str, list[tuple[str, Any]]]:
    if function_name in [
        "resources.Completions.create",
        "resources.AsyncCompletions.create",
    ]:
        return original_kwargs.get("prompt"), []


def extract_anthropic_output(function_name: str, output: Completion) -> str:
    if function_name in [
        "resources.Completions.create",
        "resources.AsyncCompletions.create",
    ]:
        return output.completion


def extract_anthropic_output_generator(
    function_name: str, outputs: Stream[Completion] | AsyncStream[Completion]
) -> str:
    if function_name in [
        "resources.Completions.create",
        "resources.AsyncCompletions.create",
    ]:
        return "".join([output.completion for output in outputs])
