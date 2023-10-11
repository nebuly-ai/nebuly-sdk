# pylint: disable=duplicate-code
from __future__ import annotations

import copyreg
import logging
from dataclasses import dataclass
from typing import Any, Callable, List, cast

from cohere.client import Client  # type: ignore
from cohere.responses.chat import (  # type: ignore
    Chat,
    StreamEnd,
    StreamingChat,
    StreamStart,
    StreamTextGeneration,
)
from cohere.responses.generation import (  # type: ignore
    Generations,
    StreamingGenerations,
    StreamingText,
)
from typing_extensions import TypeAlias

from nebuly.entities import HistoryEntry, ModelInput
from nebuly.providers.base import ProviderDataExtractor
from nebuly.providers.utils import get_argument

logger = logging.getLogger(__name__)


StreamingChatEntry: TypeAlias = StreamStart | StreamEnd | StreamTextGeneration


def is_cohere_generator(function: Callable[[Any], Any]) -> bool:
    return isinstance(function, (StreamingGenerations, StreamingChat))


def handle_cohere_unpickable_objects() -> None:
    def _pickle_cohere_client(c: Client) -> tuple[type[Client], tuple[Any, ...]]:
        return Client, (
            c.api_key,
            c.num_workers,
            c.request_dict,
            True,
            None,
            c.max_retries,
            c.timeout,
            c.api_url,
        )

    copyreg.pickle(Client, _pickle_cohere_client)


@dataclass(frozen=True)
class CohereDataExtractor(ProviderDataExtractor):
    original_args: tuple[Any, ...]
    original_kwargs: dict[str, Any]
    function_name: str

    def _extract_history(self) -> list[HistoryEntry]:
        history = get_argument(
            self.original_args, self.original_kwargs, "chat_history", 7
        )

        # Remove messages that are not from the user or the assistant
        history = [
            message
            for message in history
            if message["user_name"] in ["User", "Chatbot"]
        ]

        if len(history) % 2 != 0:
            logger.warning("Odd number of chat history elements, ignoring last element")
            history = history[:-1]

        # Convert the history to [(user, assistant), ...] format
        history = [
            HistoryEntry(
                user=history[i]["message"], assistant=history[i + 1]["message"]
            )
            for i in range(0, len(history), 2)
            if i < len(history) - 1
        ]
        return history  # type: ignore

    def extract_input_and_history(self) -> ModelInput:
        if self.function_name in ["Client.generate", "AsyncClient.generate"]:
            prompt = get_argument(self.original_args, self.original_kwargs, "prompt", 1)
            return ModelInput(prompt=prompt)
        if self.function_name in ["Client.chat", "AsyncClient.chat"]:
            prompt = get_argument(
                self.original_args, self.original_kwargs, "message", 1
            )
            history = self._extract_history()
            return ModelInput(prompt=prompt, history=history)

        raise ValueError(f"Unknown function name: {self.function_name}")

    def extract_output(
        self,
        stream: bool,
        outputs: Generations | Chat | list[StreamingText] | list[StreamingChatEntry],
    ) -> str:
        if stream and isinstance(outputs, list):
            return self._extract_output_generator(outputs)

        if isinstance(outputs, Generations) and self.function_name in [
            "Client.generate",
            "AsyncClient.generate",
        ]:
            return cast(str, outputs.generations[0].text)
        if isinstance(outputs, Chat) and self.function_name in [
            "Client.chat",
            "AsyncClient.chat",
        ]:
            return cast(str, outputs.text)

        raise ValueError(
            f"Unknown function name: {self.function_name} "
            f"or output type: {type(outputs)}"
        )

    def _extract_output_generator(
        self, outputs: list[StreamingText] | list[StreamingChatEntry]
    ) -> str:
        if all(
            isinstance(elem, StreamingText) for elem in outputs
        ) and self.function_name in ["Client.generate", "AsyncClient.generate"]:
            return "".join([output.text for output in outputs])
        if all(
            isinstance(elem, (StreamStart, StreamEnd, StreamTextGeneration))
            for elem in outputs
        ) and self.function_name in ["Client.chat", "AsyncClient.chat"]:
            messages = cast(List[StreamingChatEntry], outputs)
            return "".join(
                [
                    output.text
                    for output in messages
                    if output.event_type == "text-generation"
                ]
            )

        raise ValueError(
            f"Unknown function name: {self.function_name} "
            f"or output type: {type(outputs)}"
        )
