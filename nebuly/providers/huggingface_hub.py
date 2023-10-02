from __future__ import annotations

from typing import Any, Iterator

from huggingface_hub.inference._text_generation import (  # type: ignore
    TextGenerationResponse,
    TextGenerationStreamResponse,
)
from huggingface_hub.inference._types import ConversationalOutput  # type: ignore

from nebuly.providers.utils import get_argument


def extract_hf_hub_input_and_history(
    original_args: tuple[Any, ...],
    original_kwargs: dict[str, Any],
    function_name: str,
) -> tuple[str, list[tuple[str, Any]]]:
    if function_name == "InferenceClient.conversational":
        prompt = get_argument(original_args, original_kwargs, "text", 1)
        generated_responses = get_argument(
            original_args, original_kwargs, "generated_responses", 2
        )
        past_user_inputs = get_argument(
            original_args, original_kwargs, "past_user_inputs", 3
        )
        history = []
        for user_input, assistant_response in zip(
            past_user_inputs if past_user_inputs is not None else [],
            generated_responses if generated_responses is not None else [],
        ):
            history.append(("user", user_input))
            history.append(("assistant", assistant_response))
        return prompt, history
    if function_name == "InferenceClient.text_generation":
        prompt = get_argument(original_args, original_kwargs, "prompt", 1)
        return prompt, []

    raise ValueError(f"Unknown function name: {function_name}")


def extract_hf_hub_output(
    function_name: str, output: str | ConversationalOutput | TextGenerationResponse
) -> str:
    if function_name == "InferenceClient.conversational":
        return output["generated_text"]  # type: ignore
    if function_name == "InferenceClient.text_generation":
        if isinstance(output, TextGenerationResponse):
            return output.generated_text  # type: ignore
        return output

    raise ValueError(f"Unknown function name: {function_name}")


def extract_hf_hub_output_generator(
    function_name: str, outputs: Iterator[str | TextGenerationStreamResponse]
) -> str:
    if function_name == "InferenceClient.text_generation":
        return "".join(
            [
                output
                if isinstance(output, str)
                else output.generated_text
                if output.generated_text is not None
                else ""
                for output in outputs
            ]
        )

    raise ValueError(f"Unknown function name: {function_name}")
