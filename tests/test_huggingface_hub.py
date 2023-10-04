# pylint: disable=duplicate-code
from __future__ import annotations

import json
from collections.abc import AsyncIterable
from typing import Any
from unittest.mock import patch

import pytest
from huggingface_hub import AsyncInferenceClient, InferenceClient  # type: ignore
from huggingface_hub.inference._text_generation import (  # type: ignore
    Details,
    FinishReason,
    StreamDetails,
    TextGenerationResponse,
    TextGenerationStreamResponse,
    Token,
)
from huggingface_hub.inference._types import ConversationalOutput  # type: ignore

from nebuly.contextmanager import new_interaction
from nebuly.entities import InteractionWatch, SpanWatch
from nebuly.observers import NebulyObserver
from nebuly.requests import CustomJSONEncoder
from tests.common import nebuly_init


@pytest.fixture(name="hf_hub_text_generation_str")
def fixture_hf_hub_text_generation_str() -> str:
    return "a beautiful library for interacting with the Hugging Face Hub."


def test_hf_hub_text_generation_str(hf_hub_text_generation_str: str) -> None:
    with patch("huggingface_hub.InferenceClient.text_generation") as mock_generation:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_generation.return_value = hf_hub_text_generation_str
            nebuly_init(observer=mock_observer)

            client = InferenceClient()
            result = client.text_generation(
                "The huggingface_hub library is ",
                user_id="test_user",
                user_group_profile="test_group",
            )

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "The huggingface_hub library is "
            assert interaction_watch.history == []
            assert interaction_watch.output == hf_hub_text_generation_str
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="hf_hub_text_generation_list")
def fixture_hf_hub_text_generation_list() -> list[str]:
    return ["a", " beautiful", " library", "."]


def test_hf_hub_text_generation_stream(hf_hub_text_generation_list: list[str]) -> None:
    with patch("huggingface_hub.InferenceClient.text_generation") as mock_generation:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_generation.return_value = (el for el in hf_hub_text_generation_list)
            nebuly_init(observer=mock_observer)

            client = InferenceClient()
            for _ in client.text_generation(  # pylint: disable=not-an-iterable
                "The huggingface_hub library is ", stream=True
            ):
                ...

            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "The huggingface_hub library is "
            assert interaction_watch.history == []
            assert interaction_watch.output == "".join(hf_hub_text_generation_list)
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="hf_hub_text_generator_async")
async def fixture_hf_hub_text_generator_async() -> AsyncIterable[str]:
    texts = ["a", " beautiful", " library", "."]
    for text in texts:
        yield text


@pytest.mark.asyncio
async def test_hf_hub_text_generation_stream_async(
    hf_hub_text_generator_async: AsyncIterable[str],
) -> None:
    with patch(
        "huggingface_hub.AsyncInferenceClient.text_generation"
    ) as mock_generation:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_generation.return_value = hf_hub_text_generator_async
            nebuly_init(observer=mock_observer)

            client = AsyncInferenceClient()
            async for _ in await client.text_generation(  # pylint: disable=not-an-iterable  # noqa: E501
                "The huggingface_hub library is ", stream=True
            ):
                ...

            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "The huggingface_hub library is "
            assert interaction_watch.history == []
            assert interaction_watch.output == "a beautiful library."
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="hf_hub_text_generation_details")
def fixture_hf_hub_text_generation_details() -> TextGenerationResponse:
    return TextGenerationResponse(
        generated_text="100% open source and built to be easy to "
        "use.\n\nTo install it, you can",
        details=Details(
            finish_reason=FinishReason.Length,
            generated_tokens=20,
            seed=None,
            prefill=[],
            tokens=[
                Token(id=1425, text="100", logprob=-1.0175781, special=False),
                Token(id=16, text="%", logprob=-0.046295166, special=False),
            ],
            best_of_sequences=None,
        ),
    )


def test_hf_hub_text_generation_details(
    hf_hub_text_generation_details: TextGenerationResponse,
) -> None:
    with patch("huggingface_hub.InferenceClient.text_generation") as mock_generation:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_generation.return_value = hf_hub_text_generation_details
            nebuly_init(observer=mock_observer)

            client = InferenceClient()
            result = client.text_generation(
                "The huggingface_hub library is ", details=True
            )

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "The huggingface_hub library is "
            assert interaction_watch.history == []
            assert (
                interaction_watch.output
                == hf_hub_text_generation_details.generated_text
            )
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="hf_hub_text_generation_details_list")
def fixture_hf_hub_text_generation_details_list() -> list[TextGenerationStreamResponse]:
    return [
        TextGenerationStreamResponse(
            token=Token(id=1425, text="100", logprob=-1.0175781, special=False),
            generated_text=None,
            details=None,
        ),
        TextGenerationStreamResponse(
            token=Token(id=1425, text="100", logprob=-1.0175781, special=False),
            generated_text=None,
            details=None,
        ),
        TextGenerationStreamResponse(
            token=Token(id=1425, text="100", logprob=-1.0175781, special=False),
            generated_text="a beautiful.",
            details=StreamDetails(
                finish_reason=FinishReason.Length, generated_tokens=20, seed=None
            ),
        ),
    ]


def test_hf_hub_text_generation_details_stream(
    hf_hub_text_generation_details_list: list[TextGenerationStreamResponse],
) -> None:
    with patch("huggingface_hub.InferenceClient.text_generation") as mock_generation:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_generation.return_value = (
                el for el in hf_hub_text_generation_details_list
            )
            nebuly_init(observer=mock_observer)

            client = InferenceClient()
            for _ in client.text_generation(  # pylint: disable=not-an-iterable
                "The huggingface_hub library is ", stream=True, details=True
            ):
                ...

            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "The huggingface_hub library is "
            assert interaction_watch.history == []
            assert interaction_watch.output == "".join(
                [
                    el.generated_text
                    for el in hf_hub_text_generation_details_list
                    if el.generated_text is not None
                ]
            )
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.mark.asyncio
async def test_hf_hub_text_generation_async(hf_hub_text_generation_str: str) -> None:
    with patch(
        "huggingface_hub.AsyncInferenceClient.text_generation"
    ) as mock_generation:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_generation.return_value = hf_hub_text_generation_str
            nebuly_init(observer=mock_observer)

            client = AsyncInferenceClient()
            result = await client.text_generation(
                "The huggingface_hub library is ",
                user_id="test_user",
                user_group_profile="test_group",
            )

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "The huggingface_hub library is "
            assert interaction_watch.history == []
            assert interaction_watch.output == hf_hub_text_generation_str
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="hf_hub_sample_input")
def fixture_hf_hub_sample_input() -> dict[str, Any]:
    return {
        "generated_text": " My name is samantha, and I am a student "
        "at the University of Pittsburgh.",
        "conversation": {
            "generated_responses": [
                " My name is samantha, and I am a student at "
                "the University of Pittsburgh."
            ],
            "past_user_inputs": ["Hello, who are you?"],
        },
    }


@pytest.fixture(name="hf_hub_conversational")
def fixture_hf_hub_conversational() -> ConversationalOutput:
    return ConversationalOutput(
        **{
            "generated_text": " It is, but I am studying to be a nurse. "
            "What do you do?",
            "conversation": {
                "generated_responses": [
                    " My name is samantha, and I am a student at the "
                    "University of Pittsburgh.",
                    " It is, but I am studying to be a nurse. What do you do?",
                ],
                "past_user_inputs": ["Hello, who are you?", "Wow, that's scary!"],
            },
        }
    )


def test_hf_hub_conversational__no_context_manager__no_history(
    hf_hub_conversational: ConversationalOutput,
) -> None:
    with patch("huggingface_hub.InferenceClient.conversational") as mock_conversational:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_conversational.return_value = hf_hub_conversational
            nebuly_init(observer=mock_observer)

            client = InferenceClient()
            result = client.conversational(
                text="Wow, that's scary!",
            )

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Wow, that's scary!"
            assert interaction_watch.history == []
            assert (
                interaction_watch.output
                == " It is, but I am studying to be a nurse. What do you do?"
            )
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_hf_hub_conversational__no_context_manager__with_history(
    hf_hub_sample_input: dict[str, Any], hf_hub_conversational: ConversationalOutput
) -> None:
    with patch("huggingface_hub.InferenceClient.conversational") as mock_conversational:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_conversational.return_value = hf_hub_conversational
            nebuly_init(observer=mock_observer)

            client = InferenceClient()
            result = client.conversational(
                "Wow, that's scary!",
                generated_responses=hf_hub_sample_input["conversation"][
                    "generated_responses"
                ],
                past_user_inputs=hf_hub_sample_input["conversation"][
                    "past_user_inputs"
                ],
            )

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Wow, that's scary!"
            assert interaction_watch.history == [
                ("user", "Hello, who are you?"),
                (
                    "assistant",
                    " My name is samantha, and I am a student at "
                    "the University of Pittsburgh.",
                ),
            ]
            assert (
                interaction_watch.output
                == " It is, but I am studying to be a nurse. What do you do?"
            )
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_hf_hub_conversational__with_context_manager__no_history(
    hf_hub_conversational: ConversationalOutput,
) -> None:
    with patch("huggingface_hub.InferenceClient.conversational") as mock_conversational:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_conversational.return_value = hf_hub_conversational
            nebuly_init(observer=mock_observer)
            client = InferenceClient()
            with new_interaction(
                user_id="test_user", user_group_profile="test_group"
            ) as interaction:
                interaction.set_input("Wow, that's scary!")
                result = client.conversational(
                    "Wow, that's scary!",
                )
                interaction.set_output(
                    " It is, but I am studying to be a nurse. What do you do?"
                )

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Wow, that's scary!"
            assert interaction_watch.history == []
            assert (
                interaction_watch.output
                == " It is, but I am studying to be a nurse. What do you do?"
            )
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_hf_hub_conversational__with_context_manager__with_history(
    hf_hub_sample_input: dict[str, Any], hf_hub_conversational: ConversationalOutput
) -> None:
    with patch("huggingface_hub.InferenceClient.conversational") as mock_conversational:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_conversational.return_value = hf_hub_conversational
            nebuly_init(observer=mock_observer)
            client = InferenceClient()
            with new_interaction(
                user_id="test_user", user_group_profile="test_group"
            ) as interaction:
                interaction.set_input("Wow, that's scary!")
                interaction.set_history(
                    [
                        ("user", "Hello, who are you?"),
                        (
                            "assistant",
                            " My name is samantha, and I am a student at "
                            "the University of Pittsburgh.",
                        ),
                    ]
                )
                result = client.conversational(
                    "Wow, that's scary!",
                    generated_responses=hf_hub_sample_input["conversation"][
                        "generated_responses"
                    ],
                    past_user_inputs=hf_hub_sample_input["conversation"][
                        "past_user_inputs"
                    ],
                )
                interaction.set_output(
                    " It is, but I am studying to be a nurse. What do you do?"
                )

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Wow, that's scary!"
            assert interaction_watch.history == [
                ("user", "Hello, who are you?"),
                (
                    "assistant",
                    " My name is samantha, and I am a student at "
                    "the University of Pittsburgh.",
                ),
            ]
            assert (
                interaction_watch.output
                == " It is, but I am studying to be a nurse. What do you do?"
            )
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.mark.asyncio
async def test_hf_hub_conversational__async(
    hf_hub_conversational: ConversationalOutput,
) -> None:
    with patch(
        "huggingface_hub.AsyncInferenceClient.conversational"
    ) as mock_conversational:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_conversational.return_value = hf_hub_conversational
            nebuly_init(observer=mock_observer)

            client = AsyncInferenceClient()
            result = await client.conversational(
                text="Wow, that's scary!",
            )

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Wow, that's scary!"
            assert interaction_watch.history == []
            assert (
                interaction_watch.output
                == " It is, but I am studying to be a nurse. What do you do?"
            )
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )
