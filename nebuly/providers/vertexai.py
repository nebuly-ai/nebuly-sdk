# pylint: disable=duplicate-code
from __future__ import annotations

import logging
from typing import Any, Callable, cast

from vertexai.language_models import (  # type: ignore
    ChatMessage,
    ChatModel,
    ChatSession,
    TextGenerationModel,
    TextGenerationResponse,
)

from nebuly.entities import HistoryEntry, ModelInput
from nebuly.providers.base import PicklerHandler, ProviderDataExtractor
from nebuly.providers.utils import get_argument

logger = logging.getLogger(__name__)


class VertexAIPicklerHandler(PicklerHandler):
    @property
    def _object_key_attribute_mapping(
        self,
    ) -> dict[Callable[[Any], Any], dict[str, list[str]]]:
        return {
            TextGenerationModel: {
                "key_names": ["model_id", "endpoint_name"],
                "attribute_names": ["_model_id", "_endpoint_name"],
            },
            ChatModel: {
                "key_names": ["model_id", "endpoint_name"],
                "attribute_names": ["_model_id", "_endpoint_name"],
            },
            ChatSession: {
                "key_names": [
                    "model",
                    "context",
                    "examples",
                    "max_output_tokens",
                    "temperature",
                    "top_k",
                    "top_p",
                    "message_history",
                    "stop_sequences",
                ],
                "attribute_names": [
                    "_model._model_id",
                    "_context",
                    "_examples",
                    "_max_output_tokens",
                    "_temperature",
                    "_top_k",
                    "_top_p",
                    "_message_history",
                    "_stop_sequences",
                ],
            },
        }


def handle_vertexai_unpickable_objects() -> None:
    pickler_handler = VertexAIPicklerHandler()

    for obj in [TextGenerationModel, ChatModel, ChatSession]:
        pickler_handler.handle_unpickable_object(
            obj=obj,
        )


class VertexAIDataExtractor(ProviderDataExtractor):
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
        message_history: list[ChatMessage] = getattr(
            self.original_args[0], "message_history", []
        )

        # Remove messages that are not from the user or the assistant
        message_history = [
            message
            for message in message_history
            if len(message_history) > 1 and message.author.lower() in ["user", "bot"]
        ]

        if len(message_history) % 2 != 0:
            logger.warning("Odd number of chat history elements, ignoring last element")
            message_history = message_history[:-1]

        # Convert the history to [(user, assistant), ...] format
        history: list[HistoryEntry] = [
            HistoryEntry(
                user=message_history[i].content,
                assistant=message_history[i + 1].content,
            )
            for i in range(0, len(message_history), 2)
            if i < len(message_history) - 1
        ]

        return history

    def extract_input_and_history(self, outputs: Any) -> ModelInput:
        if self.function_name in [
            "language_models.ChatSession.send_message",
            "language_models.ChatSession.send_message_async",
            "language_models.ChatSession.send_message_streaming",
            "language_models.TextGenerationModel.predict",
            "language_models.TextGenerationModel.predict_async",
            "language_models.TextGenerationModel.predict_streaming",
        ]:
            prompt = get_argument(
                args=self.original_args,
                kwargs=self.original_kwargs,
                arg_name=(
                    "prompt"
                    if "TextGenerationModel" in self.function_name
                    else "message"
                ),
                arg_idx=1,
            )
            history = self._extract_history()
            return ModelInput(prompt=prompt, history=history)

        raise ValueError(f"Unknown function name: {self.function_name}")

    def extract_output(
        self,
        stream: bool,
        outputs: TextGenerationResponse | list[TextGenerationResponse],
    ) -> str:
        if stream and isinstance(outputs, list):
            return self._extract_output_generator(outputs)
        if isinstance(outputs, TextGenerationResponse) and self.function_name in [
            "language_models.TextGenerationModel.predict",
            "language_models.TextGenerationModel.predict_async",
            "language_models.ChatSession.send_message",
            "language_models.ChatSession.send_message_async",
        ]:
            return cast(str, outputs.text)

        raise ValueError(
            f"Unknown function name: {self.function_name} "
            f"or output type: {type(outputs)}"
        )

    def _extract_output_generator(self, outputs: list[TextGenerationResponse]) -> str:
        if all(
            isinstance(output, TextGenerationResponse) for output in outputs
        ) and self.function_name in [
            "language_models.TextGenerationModel.predict_streaming",
            "language_models.ChatSession.send_message_streaming",
        ]:
            return "".join([output.text for output in outputs])

        raise ValueError(
            f"Unknown function name: {self.function_name} "
            f"or output type: {type(outputs)}"
        )
