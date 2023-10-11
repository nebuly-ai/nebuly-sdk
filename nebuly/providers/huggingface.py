# pylint: disable=duplicate-code
from __future__ import annotations

import logging
from typing import Any, cast

from transformers.pipelines import (  # type: ignore
    Conversation,
    ConversationalPipeline,
    TextGenerationPipeline,
)

from nebuly.entities import HistoryEntry, ModelInput

logger = logging.getLogger(__name__)


TextGeneration = list[dict[str, str]]


def _extract_hf_pipeline_history(
    conversations: Conversation,
) -> list[HistoryEntry]:
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
        HistoryEntry(user=history[i]["content"], assistant=history[i + 1]["content"])
        for i in range(0, len(history), 2)
        if i < len(history) - 1
    ]

    return history  # type: ignore


def extract_hf_pipeline_input_and_history(
    original_args: tuple[Any, ...],
    function_name: str,
) -> ModelInput | list[ModelInput]:
    if isinstance(original_args[0], ConversationalPipeline):
        conversations: Conversation | list[Conversation] = original_args[1]
        if isinstance(conversations, Conversation):
            prompt = conversations.messages[-1]["content"]
            history = _extract_hf_pipeline_history(conversations)
            return ModelInput(prompt=prompt, history=history)
        return [
            ModelInput(
                prompt=conversation.messages[-1]["content"],
                history=_extract_hf_pipeline_history(conversation),
            )
            for conversation in conversations
        ]
    if isinstance(original_args[0], TextGenerationPipeline):
        prompts: str | list[str] = original_args[1]
        if isinstance(prompts, str):
            return ModelInput(prompt=prompts)
        return [ModelInput(prompt=prompt) for prompt in prompts]

    raise ValueError(f"Unknown function name: {function_name}")


def extract_hf_pipeline_output(
    function_name: str,
    output: Conversation | list[Conversation] | TextGeneration | list[TextGeneration],
) -> str | list[str]:
    if isinstance(output, Conversation):
        result = cast(Conversation, output)
        return cast(str, result.generated_responses[-1])
    if isinstance(output, list):
        if isinstance(output[0], Conversation):
            result = cast(list[Conversation], output)
            return [out.generated_responses[-1] for out in result]
        if isinstance(output[0], dict) and "generated_text" in output[0]:
            result = cast(TextGeneration, output)
            return cast(str, result[0]["generated_text"])
        if (
            isinstance(output[0], list)
            and isinstance(output[0][0], dict)
            and "generated_text" in output[0][0]
        ):
            result = cast(list[TextGeneration], output)
            return [cast(str, out[0]["generated_text"]) for out in result]

    raise ValueError(f"Unknown function name: {function_name}")
