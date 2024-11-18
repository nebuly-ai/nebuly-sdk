# pylint: disable=duplicate-code
from __future__ import annotations

import logging
from typing import Any, Dict, List, cast

from transformers.pipelines import (  # type: ignore
    Conversation,
    ConversationalPipeline,
    TextGenerationPipeline,
)
from typing_extensions import TypeAlias

from nebuly.entities import HistoryEntry, ModelInput
from nebuly.providers.base import ProviderDataExtractor

logger = logging.getLogger(__name__)


TextGeneration: TypeAlias = List[Dict[str, str]]


class HuggingFaceDataExtractor(ProviderDataExtractor):
    def __init__(
        self,
        function_name: str,
        original_args: tuple[Any, ...],
        original_kwargs: dict[str, Any],
    ):
        self.function_name = function_name
        self.original_args = original_args
        self.original_kwargs = original_kwargs

    @staticmethod
    def _extract_history(conversations: Conversation) -> list[HistoryEntry]:
        history = conversations.messages[:-1]

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

        return history  # type: ignore

    def extract_input_and_history(self, outputs: Any) -> ModelInput | list[ModelInput]:
        pipeline: ConversationalPipeline | TextGenerationPipeline = self.original_args[
            0
        ]  # noqa: E501
        if isinstance(pipeline, ConversationalPipeline):
            conversations: Conversation | list[Conversation] = self.original_args[1]
            if isinstance(conversations, Conversation):
                prompt = conversations.messages[-1]["content"]
                history = self._extract_history(conversations)
                return ModelInput(prompt=prompt, history=history)
            return [
                ModelInput(
                    prompt=conversation.messages[-1]["content"],
                    history=self._extract_history(conversation),
                )
                for conversation in conversations
            ]
        if isinstance(pipeline, TextGenerationPipeline):
            prompts: str | list[str] = self.original_args[1]
            if isinstance(prompts, str):
                return ModelInput(prompt=prompts)
            return [ModelInput(prompt=prompt) for prompt in prompts]

        raise ValueError(f"Unknown function name: {self.function_name}")

    def extract_output(
        self,
        stream: bool,
        outputs: (
            Conversation | list[Conversation] | TextGeneration | list[TextGeneration]
        ),
    ) -> str | list[str]:
        if stream:
            return self._extract_output_generator(outputs)

        if isinstance(outputs, Conversation):
            return cast(str, outputs.generated_responses[-1])
        if isinstance(outputs, list):
            if all(isinstance(output, Conversation) for output in outputs):
                return [out.generated_responses[-1] for out in outputs]  # type: ignore
            if all(isinstance(output, dict) for output in outputs):  # TextGeneration
                return outputs[0]["generated_text"]  # type: ignore
            if all(  # list[TextGeneration]
                isinstance(output, list) for output in outputs
            ) and all(
                isinstance(output[0], dict) for output in outputs  # type: ignore
            ):
                return [out[0]["generated_text"] for out in outputs]  # type: ignore

        raise ValueError(
            f"Unknown function name: {self.function_name} "
            f"or output type: {type(outputs)}"
        )

    def _extract_output_generator(self, outputs: Any) -> str:
        raise NotImplementedError("HuggingFace does not support streaming yet")
