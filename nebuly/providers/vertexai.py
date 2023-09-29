from __future__ import annotations

import copyreg
from typing import Any, Iterator

from vertexai.language_models import (
    ChatModel,
    TextGenerationModel,
    TextGenerationResponse,
)

from nebuly.providers.utils import get_argument


def handle_vertexai_unpickable_objects() -> None:
    def _pickle_text_generation_model(
        c: TextGenerationModel,
    ) -> tuple[type[TextGenerationModel], tuple[Any, ...]]:
        return TextGenerationModel, (
            c._model_id,  # pylint: disable=protected-access
            c._endpoint_name,  # pylint: disable=protected-access
        )

    copyreg.pickle(TextGenerationModel, _pickle_text_generation_model)
    copyreg.pickle(ChatModel, _pickle_text_generation_model)


def extract_vertexai_input_and_history(
    original_args: tuple[Any],
    original_kwargs: dict[str, Any],
    function_name: str,
) -> tuple[str, list[tuple[str, Any]]]:
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
        history = [
            ("user" if el.author == "user" else "assistant", el.content)
            for el in getattr(original_args[0], "message_history", [])
        ]
        return prompt, history

    raise ValueError(f"Unknown function name: {function_name}")


def extract_vertexai_output(function_name: str, output: TextGenerationResponse) -> str:
    if function_name in [
        "language_models.TextGenerationModel.predict",
        "language_models.TextGenerationModel.predict_async",
        "language_models.ChatSession.send_message",
        "language_models.ChatSession.send_message_async",
    ]:
        return output.text

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
