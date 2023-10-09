# pylint: disable=duplicate-code
from __future__ import annotations

from typing import Any, cast

from transformers.pipelines import (  # type: ignore
    Conversation,
    ConversationalPipeline,
    Pipeline,
    TextGenerationPipeline,
)


def is_pipeline_supported(pipeline: Pipeline) -> bool:
    return isinstance(pipeline, (ConversationalPipeline, TextGenerationPipeline))


def extract_hf_pipeline_input_and_history(
    original_args: tuple[Any, ...],
    function_name: str,
) -> tuple[str, list[tuple[str, Any]]]:
    if isinstance(original_args[0], ConversationalPipeline):
        conversations = original_args[1]
        if isinstance(conversations, list):
            conversations = conversations[0]
        prompt = conversations.messages[-1]["content"]
        generated_responses = conversations.generated_responses
        past_user_inputs = conversations.past_user_inputs
        history = []
        for user_input, assistant_response in zip(
            past_user_inputs if past_user_inputs is not None else [],
            generated_responses if generated_responses is not None else [],
        ):
            history.append(("user", user_input))
            history.append(("assistant", assistant_response))
        return prompt, history
    if isinstance(original_args[0], TextGenerationPipeline):
        prompt = original_args[1]
        if isinstance(prompt, list):
            prompt = prompt[0]
        return prompt, []

    raise ValueError(f"Unknown function name: {function_name}")


def extract_hf_pipeline_output(
    function_name: str, output: Conversation | list[Conversation]
) -> str:
    if isinstance(output, Conversation):
        return cast(str, output.generated_responses[-1])
    if isinstance(output, list):
        if isinstance(output[0], Conversation):
            return cast(str, output[0].generated_responses[-1])
        if isinstance(output[0], dict) and "generated_text" in output[0]:
            return cast(str, output[0]["generated_text"])
        if (
            isinstance(output[0], list)
            and isinstance(output[0][0], dict)
            and "generated_text" in output[0][0]
        ):
            return cast(str, output[0][0]["generated_text"])

    raise ValueError(f"Unknown function name: {function_name}")
