# pylint: disable=duplicate-code, import-error, no-name-in-module
# mypy: ignore-errors
from __future__ import annotations

from typing import Any

from openai.openai_object import OpenAIObject  # type: ignore


def extract_openai_input_and_history(
    original_kwargs: dict[str, Any],
    function_name: str,
) -> tuple[str, list[tuple[str, Any]]]:
    if function_name in ["Completion.create", "Completion.acreate"]:
        return original_kwargs.get("prompt", ""), []
    if function_name in ["ChatCompletion.create", "ChatCompletion.acreate"]:
        history = [
            (el["role"], el["content"])
            for el in original_kwargs.get("messages", [])[:-1]
            if len(original_kwargs.get("messages", [])) > 1
        ]
        return original_kwargs.get("messages", [])[-1]["content"], history

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