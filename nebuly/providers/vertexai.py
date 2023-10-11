# pylint: disable=duplicate-code
from __future__ import annotations

import copyreg
import logging
from dataclasses import dataclass
from typing import Any, cast

from vertexai.language_models import (  # type: ignore
    ChatMessage,
    ChatModel,
    ChatSession,
    TextGenerationModel,
    TextGenerationResponse,
)

from nebuly.entities import HistoryEntry, ModelInput
from nebuly.providers.base import ProviderDataExtractor
from nebuly.providers.utils import get_argument

logger = logging.getLogger(__name__)


def handle_vertexai_unpickable_objects() -> None:
    def _pickle_text_generation_model(
        c: TextGenerationModel,
    ) -> tuple[type[TextGenerationModel], tuple[Any, ...]]:
        return TextGenerationModel, (
            c._model_id,  # pylint: disable=protected-access
            c._endpoint_name,  # pylint: disable=protected-access
        )

    def _pickle_chat_session(
        c: ChatSession,
    ) -> tuple[type[ChatSession], tuple[Any, ...]]:
        return ChatSession, (
            c._model._model_id,  # pylint: disable=protected-access
            c._context,  # pylint: disable=protected-access
            c._examples,  # pylint: disable=protected-access
            c._max_output_tokens,  # pylint: disable=protected-access
            c._temperature,  # pylint: disable=protected-access
            c._top_k,  # pylint: disable=protected-access
            c._top_p,  # pylint: disable=protected-access
            c._message_history,  # pylint: disable=protected-access
            c._stop_sequences,  # pylint: disable=protected-access
        )

    copyreg.pickle(TextGenerationModel, _pickle_text_generation_model)
    copyreg.pickle(ChatModel, _pickle_text_generation_model)
    copyreg.pickle(ChatSession, _pickle_chat_session)


@dataclass(frozen=True)
class VertexAIDataExtractor(ProviderDataExtractor):
    original_args: tuple[Any, ...]
    original_kwargs: dict[str, Any]
    function_name: str

    def _extract_history(self) -> list[HistoryEntry]:
        message_history: list[ChatMessage] = getattr(
            self.original_args[0], "message_history", []
        )

        # Remove messages that are not from the user or the assistant
        message_history = [
            message
            for message in message_history
            if len(message_history) > 1 and message.author in ["user", "bot"]
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

    def extract_input_and_history(self) -> ModelInput:
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
                arg_name="prompt"
                if "TextGenerationModel" in self.function_name
                else "message",
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
