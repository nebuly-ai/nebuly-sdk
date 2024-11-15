# pylint: disable=duplicate-code, import-error, no-name-in-module
# mypy: ignore-errors
from __future__ import annotations

import json
import logging
from typing import Any

from openai.openai_object import OpenAIObject  # type: ignore

from nebuly.entities import HistoryEntry, ModelInput
from nebuly.providers.base import ProviderDataExtractor

logger = logging.getLogger(__name__)


class OpenAILegacyDataExtractor(ProviderDataExtractor):
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
        history = self.original_kwargs.get("messages", [])[:-1]

        # Remove messages that are not from the user or the assistant
        history = [
            message
            for message in history
            if len(history) > 1 and message["role"].lower() in ["user", "assistant"]
        ]

        if len(history) % 2 != 0:
            logger.warning("Odd number of chat history elements, ignoring last element")
            history = history[:-1]

        # Convert the history to [(user, assistant), ...] format
        history = [
            HistoryEntry(
                user=history[i]["content"], assistant=history[i + 1]["content"]
            )
            for i in range(0, len(history), 2)
            if i < len(history) - 1
        ]
        return history

    def extract_input_and_history(self, outputs: Any) -> ModelInput:
        if self.function_name in ["Completion.create", "Completion.acreate"]:
            return ModelInput(prompt=self.original_kwargs.get("prompt", ""))
        if self.function_name in ["ChatCompletion.create", "ChatCompletion.acreate"]:
            prompt = self.original_kwargs.get("messages", [])[-1]["content"]
            history = self._extract_history()
            return ModelInput(prompt=prompt, history=history)

        raise ValueError(f"Unknown function name: {self.function_name}")

    def extract_output(
        self,
        stream: bool,
        outputs: (
            dict[str, Any] | list[dict[str, Any]] | OpenAIObject | list[OpenAIObject]
        ),
    ) -> str:
        if isinstance(outputs, list) and stream:
            return self._extract_output_generator(outputs)
        if isinstance(outputs, (OpenAIObject, dict)) and self.function_name in [
            "Completion.create",
            "Completion.acreate",
        ]:
            return outputs["choices"][0]["text"]  # type: ignore
        if isinstance(outputs, (OpenAIObject, dict)) and self.function_name in [
            "ChatCompletion.create",
            "ChatCompletion.acreate",
        ]:
            if outputs["choices"][0]["message"].get("content") is not None:
                # Normal chat completion
                return outputs["choices"][0]["message"]["content"]  # type: ignore
            # Function call
            return json.dumps(
                {
                    "function_name": outputs["choices"][0]["message"]["function_call"][
                        "name"
                    ],
                    "arguments": outputs["choices"][0]["message"]["function_call"][
                        "arguments"
                    ],
                }
            )

        raise ValueError(
            f"Unknown function name: {self.function_name} or "
            f"type of outputs: {type(outputs)}"
        )

    def _extract_output_generator(
        self, outputs: list[OpenAIObject] | list[dict[str, Any]]
    ) -> str:
        if all(
            isinstance(output, (OpenAIObject, dict)) for output in outputs
        ) and self.function_name in ["Completion.create", "Completion.acreate"]:
            return "".join([output["choices"][0].get("text", "") for output in outputs])
        if all(
            isinstance(output, (OpenAIObject, dict)) for output in outputs
        ) and self.function_name in ["ChatCompletion.create", "ChatCompletion.acreate"]:
            if not all(
                output["choices"][0]["delta"].get("content") is None
                for output in outputs
            ):
                # Normal chat completion
                return "".join(
                    [
                        output["choices"][0]["delta"].get("content", "")
                        for output in outputs
                    ]
                )
            # Function call
            return json.dumps(
                {
                    "function_name": "".join(
                        output["choices"][0]["delta"]
                        .get("function_call", {})
                        .get("name", "")
                        for output in outputs
                    ),
                    "arguments": "".join(
                        output["choices"][0]["delta"]
                        .get("function_call", {})
                        .get("arguments", "")
                        for output in outputs
                    ),
                }
            )

        raise ValueError(
            f"Unknown function name: {self.function_name} or "
            f"type of outputs: {type(outputs)}"
        )
