# pylint: disable=duplicate-code
from __future__ import annotations

import logging
from typing import Any, cast

from transformers.pipelines import (  # type: ignore
    Conversation,
    ConversationalPipeline,
    Pipeline,
    TextGenerationPipeline,
)

logger = logging.getLogger(__name__)


def is_pipeline_supported(pipeline: Pipeline) -> bool:
    return isinstance(pipeline, (ConversationalPipeline, TextGenerationPipeline))


def _extract_hf_pipeline_history(
    conversations: Conversation,
) -> list[tuple[str, str]]:
    history = conversations.messages[:-1]

    # Remove messages that are not from the user or the assistant
    history = [
        message
        for message in history
        if len(history) > 1 and message["role"] in ["user", "assistant"]
    ]

    if len(history) % 2 != 0:
        logger.warning("Odd number of chat history elements, ignoring last element")
        history = history[:-1]

    # Convert the history to [(user, assistant), ...] format
    history = [
        (history[i]["content"], history[i + 1]["content"])
        for i in range(0, len(history), 2)
        if i < len(history) - 1
    ]

    return history  # type: ignore


def extract_hf_pipeline_input_and_history(
    original_args: tuple[Any, ...],
    function_name: str,
) -> tuple[str, list[tuple[str, Any]]]:
    if isinstance(original_args[0], ConversationalPipeline):
        conversations = original_args[1]
        if isinstance(conversations, list):
            conversations = conversations[0]
        prompt = conversations.messages[-1]["content"]
        history = _extract_hf_pipeline_history(conversations)
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
