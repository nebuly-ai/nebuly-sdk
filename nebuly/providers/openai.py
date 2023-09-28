from __future__ import annotations

from typing import Any, Generator

from openai.openai_object import OpenAIObject


def extract_openai_input_and_history(
    original_kwargs: dict[str, Any],
    function_name: str,
) -> tuple[str, list[tuple[str, Any]]]:
    if function_name in ["Completion.create", "Completion.acreate"]:
        return original_kwargs.get("prompt"), []
    if function_name in ["ChatCompletion.create", "ChatCompletion.acreate"]:
        history = [
            (el["role"], el["content"])
            for el in original_kwargs.get("messages")[:-1]
            if len(original_kwargs.get("messages", [])) > 1
        ]
        return original_kwargs.get("messages")[-1]["content"], history


def extract_openai_output(function_name: str, output: OpenAIObject) -> str:
    if function_name in ["Completion.create", "Completion.acreate"]:
        return output["choices"][0]["text"]
    if function_name in ["ChatCompletion.create", "ChatCompletion.acreate"]:
        return output["choices"][0]["message"]["content"]


def extract_openai_output_generator(
    function_name: str, outputs: Generator[OpenAIObject, None, None]
) -> str:
    if function_name in ["Completion.create", "Completion.acreate"]:
        return "".join([output["choices"][0].get("text", "") for output in outputs])
    if function_name in ["ChatCompletion.create", "ChatCompletion.acreate"]:
        return "".join(
            [output["choices"][0]["delta"].get("content", "") for output in outputs]
        )
