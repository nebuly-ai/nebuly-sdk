from __future__ import annotations

import copyreg
from dataclasses import dataclass
from typing import Any, Callable

from anthropic import Anthropic, AsyncAnthropic, AsyncStream, Stream
from anthropic.types import Completion

from nebuly.entities import ModelInput
from nebuly.providers.base import ProviderDataExtractor


def is_anthropic_generator(function: Callable[[Any], Any]) -> bool:
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


@dataclass(frozen=True)
class AnthropicDataExtractor(ProviderDataExtractor):
    original_args: tuple[Any, ...]
    original_kwargs: dict[str, Any]
    function_name: str

    def extract_input_and_history(self) -> ModelInput:
        if self.function_name in [
            "resources.Completions.create",
            "resources.AsyncCompletions.create",
        ]:
            return ModelInput(prompt=self.original_kwargs["prompt"])

        raise ValueError(f"Unknown function name: {self.function_name}")

    def extract_output(
        self,
        stream: bool,
        outputs: Completion | list[Completion],
    ) -> str:
        if stream and isinstance(outputs, list):
            return self._extract_output_generator(outputs)
        if isinstance(outputs, Completion) and self.function_name in [
            "resources.Completions.create",
            "resources.AsyncCompletions.create",
        ]:
            return outputs.completion

        raise ValueError(
            f"Unknown function name: {self.function_name} or "
            f"output type: {type(outputs)}"
        )

    def _extract_output_generator(self, outputs: list[Completion]) -> str:
        if all(
            isinstance(output, Completion) for output in outputs
        ) and self.function_name in [
            "resources.Completions.create",
            "resources.AsyncCompletions.create",
        ]:
            return "".join([output.completion for output in outputs])

        raise ValueError(
            f"Unknown function name: {self.function_name} or "
            f"output type: {type(outputs)}"
        )
