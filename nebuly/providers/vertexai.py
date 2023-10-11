from __future__ import annotations

import copyreg
import logging
from typing import Any, Iterator

from vertexai.language_models import (  # type: ignore
    ChatMessage,
    ChatModel,
    ChatSession,
    TextGenerationModel,
    TextGenerationResponse,
)

from nebuly.entities import HistoryEntry, ModelInput
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


def _extract_vertexai_history(
    original_args: tuple[Any, ...],
) -> list[HistoryEntry]:
    message_history: list[ChatMessage] = getattr(
        original_args[0], "message_history", []
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
            user=message_history[i].content, assistant=message_history[i + 1].content
        )
        for i in range(0, len(message_history), 2)
        if i < len(message_history) - 1
    ]

    return history


def extract_vertexai_input_and_history(
    original_args: tuple[Any, ...],
    original_kwargs: dict[str, Any],
    function_name: str,
) -> ModelInput:
    if function_name in [
        "language_models.ChatSession.send_message",
        "language_models.ChatSession.send_message_async",
        "language_models.ChatSession.send_message_streaming",
        "language_models.TextGenerationModel.predict",
        "language_models.TextGenerationModel.predict_async",
        "language_models.TextGenerationModel.predict_streaming",
    ]:
        prompt = get_argument(
            args=original_args,
            kwargs=original_kwargs,
            arg_name="prompt" if "TextGenerationModel" in function_name else "message",
            arg_idx=1,
        )
        history = _extract_vertexai_history(original_args)
        return ModelInput(prompt=prompt, history=history)

    raise ValueError(f"Unknown function name: {function_name}")


def extract_vertexai_output(function_name: str, output: TextGenerationResponse) -> str:
    if function_name in [
        "language_models.TextGenerationModel.predict",
        "language_models.TextGenerationModel.predict_async",
        "language_models.ChatSession.send_message",
        "language_models.ChatSession.send_message_async",
    ]:
        return output.text  # type: ignore

    raise ValueError(f"Unknown function name: {function_name}")


def extract_vertexai_output_generator(
    function_name: str, outputs: Iterator[TextGenerationResponse]
) -> str:
    if function_name in [
        "language_models.TextGenerationModel.predict_streaming",
        "language_models.ChatSession.send_message_streaming",
    ]:
        return "".join([output.text for output in outputs])

    raise ValueError(f"Unknown function name: {function_name}")
