# pylint: disable=duplicate-code, unexpected-keyword-arg
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from google.cloud import aiplatform
from vertexai.language_models import (  # type: ignore
    ChatMessage,
    ChatModel,
    InputOutputTextPair,
    TextGenerationModel,
    TextGenerationResponse,
)

from nebuly.contextmanager import new_interaction
from nebuly.entities import HistoryEntry, InteractionWatch, SpanWatch
from nebuly.observers import NebulyObserver
from nebuly.requests import CustomJSONEncoder
from tests.providers.common import nebuly_init


@pytest.fixture(name="palm_completion")
def fixture_palm_completion() -> TextGenerationResponse:
    return TextGenerationResponse(
        is_blocked=False,
        _prediction_response=aiplatform.models.Prediction(
            predictions=[
                {
                    "content": "1. What is your experience with project management?\n2."
                    " What are your strengths and weaknesses as a project "
                    "manager?\n3. How do you handle conflict and difficult "
                    "situations?\n4. How do you communicate with "
                    "stakeholders and keep them informed?\n5. How do you "
                    "manage your time and resources effectively?\n6. What "
                    "is your approach to risk management?\n7. How do you "
                    "measure the success of your projects?\n8. What are "
                    "your goals for the future?\n9. What are your salary "
                    "expectations?\n10. Why do you want to work for our "
                    "company?",
                    "citationMetadata": {"citations": []},
                    "safetyAttributes": {
                        "blocked": False,
                        "scores": [0.7, 0.1, 0.1],
                        "categories": ["Finance", "Health", "Toxic"],
                    },
                }
            ],
            deployed_model_id="",
            model_version_id="",
            model_resource_name="",
            explanations=None,
        ),
        safety_attributes={"Finance": 0.7, "Health": 0.1, "Toxic": 0.1},
        text="1. What is your experience with project management?\n2."
        " What are your strengths and weaknesses as a project "
        "manager?\n3. How do you handle conflict and difficult "
        "situations?\n4. How do you communicate with "
        "stakeholders and keep them informed?\n5. How do you "
        "manage your time and resources effectively?\n6. What "
        "is your approach to risk management?\n7. How do you "
        "measure the success of your projects?\n8. What are "
        "your goals for the future?\n9. What are your salary "
        "expectations?\n10. Why do you want to work for our "
        "company?",
    )


def test_vertexai_completion__no_context_manager(
    palm_completion: TextGenerationResponse,
) -> None:
    with patch(
        "vertexai.language_models.TextGenerationModel.predict"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = palm_completion
            nebuly_init(observer=mock_observer)

            parameters = {
                "temperature": 0,
                "max_output_tokens": 256,
                "top_p": 0.8,
                "top_k": 40,
            }
            with patch(
                "google.cloud.aiplatform.Endpoint._construct_sdk_resource_from_gapic",
                return_value="endpoint",
            ):
                model = TextGenerationModel(model_id="text-bison@001")
                result = model.predict(
                    "Give me ten interview questions for the role of program manager.",
                    **parameters,
                    user_id="user_id",
                    user_group_profile="test_group",
                )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert (
                interaction_watch.input
                == "Give me ten interview questions for the role of program manager."
            )
            assert interaction_watch.output == palm_completion.text
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_vertexai_completion__with_context_manager(
    palm_completion: TextGenerationResponse,
) -> None:
    with patch(
        "vertexai.language_models.TextGenerationModel.predict"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = palm_completion
            nebuly_init(observer=mock_observer)

            with new_interaction(
                user_id="user_id", user_group_profile="test_group"
            ) as interaction:
                interaction.set_input(
                    "Give me ten interview questions for the role of program manager."
                )
                parameters = {
                    "temperature": 0,
                    "max_output_tokens": 256,
                    "top_p": 0.8,
                    "top_k": 40,
                }
                with patch(
                    "google.cloud.aiplatform.Endpoint._construct_sdk_resource_"
                    "from_gapic",
                    return_value="endpoint",
                ):
                    model = TextGenerationModel(model_id="text-bison@001")
                    result = model.predict(
                        "Give me ten interview questions for the role of "
                        "program manager.",
                        **parameters,
                    )
                    interaction.set_output(result.text)
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert interaction_watch.output == palm_completion.text
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.mark.asyncio
async def test_vertexai_completion__async(
    palm_completion: TextGenerationResponse,
) -> None:
    with patch(
        "vertexai.language_models.TextGenerationModel.predict_async"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = palm_completion
            nebuly_init(observer=mock_observer)

            parameters = {
                "temperature": 0,
                "max_output_tokens": 256,
                "top_p": 0.8,
                "top_k": 40,
            }

            with patch(
                "google.cloud.aiplatform.Endpoint._construct_sdk_resource_from_gapic",
                return_value="endpoint",
            ):
                model = TextGenerationModel(model_id="text-bison@001")
                result = await model.predict_async(
                    "Give me ten interview questions for the role of program manager.",
                    **parameters,
                    user_id="user_id",
                    user_group_profile="test_group",
                )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert (
                interaction_watch.input
                == "Give me ten interview questions for the role of program manager."
            )
            assert interaction_watch.output == palm_completion.text
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="palm_completion_stream")
def fixture_palm_completion_stream() -> list[TextGenerationResponse]:
    return [
        TextGenerationResponse(
            is_blocked=False,
            _prediction_response=aiplatform.models.Prediction(
                predictions=[
                    {
                        "content": "1. What is your experience with project "
                        "management?\n",
                        "citationMetadata": {"citations": []},
                        "safetyAttributes": {
                            "blocked": False,
                            "scores": [0.7, 0.1, 0.1],
                            "categories": ["Finance", "Health", "Toxic"],
                        },
                    }
                ],
                deployed_model_id="",
                model_version_id="",
                model_resource_name="",
                explanations=None,
            ),
            safety_attributes={"Finance": 0.7, "Health": 0.1, "Toxic": 0.1},
            text="1. What is your experience with project management?\n",
        ),
        TextGenerationResponse(
            is_blocked=False,
            _prediction_response=aiplatform.models.Prediction(
                predictions=[
                    {
                        "content": "2. What are your strengths and weaknesses as "
                        "a project manager?\n",
                        "citationMetadata": {"citations": []},
                        "safetyAttributes": {
                            "blocked": False,
                            "scores": [0.7, 0.1, 0.1],
                            "categories": ["Finance", "Health", "Toxic"],
                        },
                    }
                ],
                deployed_model_id="",
                model_version_id="",
                model_resource_name="",
                explanations=None,
            ),
            safety_attributes={"Finance": 0.7, "Health": 0.1, "Toxic": 0.1},
            text="2. What are your strengths and weaknesses as a project manager?\n",
        ),
    ]


def test_vertexai_completion_stream(
    palm_completion_stream: list[TextGenerationResponse],
) -> None:
    with patch(
        "vertexai.language_models.TextGenerationModel.predict_streaming"
    ) as mock_completion_stream:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_stream.return_value = (el for el in palm_completion_stream)
            nebuly_init(observer=mock_observer)
            parameters = {
                "temperature": 0,
                "max_output_tokens": 256,
                "top_p": 0.8,
                "top_k": 40,
            }

            with patch(
                "google.cloud.aiplatform.Endpoint._construct_sdk_resource_from_gapic",
                return_value="endpoint",
            ):
                model = TextGenerationModel(model_id="text-bison@001")
                result = ""
                for chunk in model.predict_streaming(
                    prompt="Give me two interview questions for the role of program "
                    "manager.",
                    **parameters,
                    user_id="user_id",
                    user_group_profile="test_group",
                ):
                    result += chunk.text

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert (
                interaction_watch.input
                == "Give me two interview questions for the role of program manager."
            )
            assert interaction_watch.output == "".join(
                el.text for el in palm_completion_stream
            )
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_vertexai_chat__no_context_manager(
    palm_completion: TextGenerationResponse,
) -> None:
    with patch(
        "vertexai.language_models.ChatSession.send_message"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = palm_completion
            nebuly_init(observer=mock_observer)

            parameters = {
                "temperature": 0,
                "max_output_tokens": 256,
                "top_p": 0.95,
                "top_k": 40,
            }
            with patch(
                "google.cloud.aiplatform.Endpoint._construct_sdk_resource_from_gapic",
                return_value="endpoint",
            ):
                chat_model = ChatModel("chat-bison@001")
                chat = chat_model.start_chat(
                    context="My name is Miles. You are an astronomer, knowledgeable "
                    "about the solar system.",
                    examples=[
                        InputOutputTextPair(
                            input_text="How many moons does Mars have?",
                            output_text="The planet Mars has two moons, Phobos and "
                            "Deimos.",
                        ),
                    ],
                )

                result = chat.send_message(
                    "How many planets are there in the solar system?",
                    **parameters,
                    user_id="user_id",
                    user_group_profile="test_group",
                )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert (
                interaction_watch.input
                == "How many planets are there in the solar system?"
            )
            assert interaction_watch.output == palm_completion.text
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_vertexai_chat__no_context_manager__with_history(
    palm_completion: TextGenerationResponse,
) -> None:
    with patch(
        "vertexai.language_models.ChatSession.send_message"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = palm_completion
            nebuly_init(observer=mock_observer)

            parameters = {
                "temperature": 0,
                "max_output_tokens": 256,
                "top_p": 0.95,
                "top_k": 40,
            }
            with patch(
                "google.cloud.aiplatform.Endpoint._construct_sdk_resource_from_gapic",
                return_value="endpoint",
            ):
                chat_model = ChatModel("chat-bison@001")
                chat = chat_model.start_chat(
                    context="My name is Miles. You are an astronomer, knowledgeable "
                    "about the solar system.",
                    examples=[
                        InputOutputTextPair(
                            input_text="How many moons does Mars have?",
                            output_text="The planet Mars has two moons, Phobos and "
                            "Deimos.",
                        ),
                    ],
                    message_history=[
                        ChatMessage(
                            content="How many moons does Mars have?",
                            author="user",
                        ),
                        ChatMessage(
                            content="The planet Mars has two moons, Phobos and Deimos.",
                            author="bot",
                        ),
                    ],
                )

                result = chat.send_message(
                    "How many planets are there in the solar system?",
                    **parameters,
                    user_id="user_id",
                    user_group_profile="test_group",
                )

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert (
                interaction_watch.input
                == "How many planets are there in the solar system?"
            )
            assert interaction_watch.history == [
                HistoryEntry(
                    user="How many moons does Mars have?",
                    assistant="The planet Mars has two moons, Phobos and Deimos.",
                )
            ]
            assert interaction_watch.output == palm_completion.text
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_vertexai_chat__with_context_manager(
    palm_completion: TextGenerationResponse,
) -> None:
    with patch(
        "vertexai.language_models.ChatSession.send_message"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = palm_completion
            nebuly_init(observer=mock_observer)

            with new_interaction(
                user_id="user_id", user_group_profile="test_group"
            ) as interaction:
                interaction.set_input("How many planets are there in the solar system?")
                parameters = {
                    "temperature": 0,
                    "max_output_tokens": 256,
                    "top_p": 0.95,
                    "top_k": 40,
                }
                with patch(
                    "google.cloud.aiplatform.Endpoint._construct_sdk_resource_"
                    "from_gapic",
                    return_value="endpoint",
                ):
                    chat_model = ChatModel("chat-bison@001")
                    chat = chat_model.start_chat(
                        context="My name is Miles. You are an astronomer, "
                        "knowledgeable about the solar system.",
                        examples=[
                            InputOutputTextPair(
                                input_text="How many moons does Mars have?",
                                output_text="The planet Mars has two moons, Phobos and "
                                "Deimos.",
                            ),
                        ],
                    )

                    result = chat.send_message(
                        "How many planets are there in the solar system?", **parameters
                    )
                    interaction.set_output(result.text)
            assert result is not None
            assert mock_completion_create.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert interaction_watch.output == palm_completion.text
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.mark.asyncio
async def test_vertexai_chat__async(palm_completion: TextGenerationResponse) -> None:
    with patch(
        "vertexai.language_models.ChatSession.send_message_async"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = palm_completion
            nebuly_init(observer=mock_observer)

            parameters = {
                "temperature": 0,
                "max_output_tokens": 256,
                "top_p": 0.95,
                "top_k": 40,
            }

            with patch(
                "google.cloud.aiplatform.Endpoint._construct_sdk_resource_from_gapic",
                return_value="endpoint",
            ):
                chat_model = ChatModel("chat-bison@001")
                chat = chat_model.start_chat(
                    context="My name is Miles. You are an astronomer, knowledgeable "
                    "about the solar system.",
                    examples=[
                        InputOutputTextPair(
                            input_text="How many moons does Mars have?",
                            output_text="The planet Mars has two moons, Phobos and "
                            "Deimos.",
                        ),
                    ],
                )

                result = await chat.send_message_async(
                    "How many planets are there in the solar system?",
                    **parameters,
                    user_id="user_id",
                    user_group_profile="test_group",
                )

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert (
                interaction_watch.input
                == "How many planets are there in the solar system?"
            )
            assert interaction_watch.output == palm_completion.text
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_vertexai_chat__stream(
    palm_completion_stream: list[TextGenerationResponse],
) -> None:
    with patch(
        "vertexai.language_models.ChatSession.send_message_streaming"
    ) as mock_completion_stream:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_stream.return_value = (el for el in palm_completion_stream)
            nebuly_init(observer=mock_observer)

            parameters = {
                "temperature": 0,
                "max_output_tokens": 256,
                "top_p": 0.95,
                "top_k": 40,
            }

            with patch(
                "google.cloud.aiplatform.Endpoint._construct_sdk_resource_from_gapic",
                return_value="endpoint",
            ):
                chat_model = ChatModel("chat-bison@001")
                chat = chat_model.start_chat(
                    context="My name is Miles. You are an astronomer, knowledgeable "
                    "about the solar system.",
                    examples=[
                        InputOutputTextPair(
                            input_text="How many moons does Mars have?",
                            output_text="The planet Mars has two moons, Phobos and "
                            "Deimos.",
                        ),
                    ],
                )

                result = ""
                for chunk in chat.send_message_streaming(
                    message="How many planets are there in the solar system?",
                    **parameters,
                    user_id="user_id",
                    user_group_profile="test_group",
                ):
                    result += chunk.text

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert (
                interaction_watch.input
                == "How many planets are there in the solar system?"
            )
            assert interaction_watch.output == "".join(
                el.text for el in palm_completion_stream
            )
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )
