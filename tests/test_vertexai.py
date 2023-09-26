from unittest.mock import patch

import pytest
from google.cloud import aiplatform
from vertexai.language_models import (
    ChatMessage,
    ChatModel,
    InputOutputTextPair,
    TextGenerationModel,
    TextGenerationResponse,
)

import nebuly
from nebuly.contextmanager import new_interaction
from nebuly.entities import InteractionWatch, SpanWatch
from nebuly.observers import NebulyObserver


@pytest.fixture()
def palm_completion() -> TextGenerationResponse:
    return TextGenerationResponse(
        is_blocked=False,
        _prediction_response=aiplatform.models.Prediction(
            predictions=[
                {
                    "content": "1. What is your experience with project management?\n2. What are your strengths and weaknesses as a project manager?\n3. How do you handle conflict and difficult situations?\n4. How do you communicate with stakeholders and keep them informed?\n5. How do you manage your time and resources effectively?\n6. What is your approach to risk management?\n7. How do you measure the success of your projects?\n8. What are your goals for the future?\n9. What are your salary expectations?\n10. Why do you want to work for our company?",
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
        text="1. What is your experience with project management?\n2. What are your strengths and weaknesses as a project manager?\n3. How do you handle conflict and difficult situations?\n4. How do you communicate with stakeholders and keep them informed?\n5. How do you manage your time and resources effectively?\n6. What is your approach to risk management?\n7. How do you measure the success of your projects?\n8. What are your goals for the future?\n9. What are your salary expectations?\n10. Why do you want to work for our company?",
    )


def test_vertexai_completion__no_context_manager(palm_completion):
    with patch(
        "vertexai.language_models.TextGenerationModel.predict"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = palm_completion
            nebuly.init(api_key="test", disable_checks=True)

            parameters = {
                "temperature": 0,
                "max_output_tokens": 256,
                "top_p": 0.8,
                "top_k": 40,
            }

            model = TextGenerationModel.from_pretrained("text-bison@001")
            result = model.predict(
                "Give me ten interview questions for the role of program manager.",
                **parameters,
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


def test_vertexai_completion__with_context_manager(palm_completion):
    with patch(
        "vertexai.language_models.TextGenerationModel.predict"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = palm_completion
            nebuly.init(api_key="test", disable_checks=True)

            with new_interaction(
                "Give me ten interview questions for the role of program manager."
            ) as interaction_watch:
                interaction_watch.set_input(
                    "Give me ten interview questions for the role of program manager."
                )
                parameters = {
                    "temperature": 0,
                    # Temperature controls the degree of randomness in token selection.
                    "max_output_tokens": 256,
                    # Token limit determines the maximum amount of text output.
                    "top_p": 0.8,
                    # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
                    "top_k": 40,
                    # A top_k of 1 means the selected token is the most probable among all tokens.
                }
                model = TextGenerationModel.from_pretrained("text-bison@001")
                result = model.predict(
                    "Give me ten interview questions for the role of program manager.",
                    **parameters,
                )
                interaction_watch.set_output(result.text)
            assert result is not None
            assert mock_observer.call_count == 1
            assert interaction_watch.output == palm_completion.text
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


@pytest.mark.asyncio
async def test_vertexai_completion__async(palm_completion):
    with patch(
        "vertexai.language_models.TextGenerationModel.predict_async"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = palm_completion
            nebuly.init(api_key="test", disable_checks=True)

            parameters = {
                "temperature": 0,
                # Temperature controls the degree of randomness in token selection.
                "max_output_tokens": 256,
                # Token limit determines the maximum amount of text output.
                "top_p": 0.8,
                # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
                "top_k": 40,
                # A top_k of 1 means the selected token is the most probable among all tokens.
            }

            model = TextGenerationModel.from_pretrained("text-bison@001")
            result = await model.predict_async(
                "Give me ten interview questions for the role of program manager.",
                **parameters,
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


@pytest.fixture()
def palm_completion_stream() -> list[TextGenerationResponse]:
    return [
        TextGenerationResponse(
            is_blocked=False,
            _prediction_response=aiplatform.models.Prediction(
                predictions=[
                    {
                        "content": "1. What is your experience with project management?\n",
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
                        "content": "2. What are your strengths and weaknesses as a project manager?\n",
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


def test_vertexai_completion_stream(palm_completion_stream):
    with patch(
        "vertexai.language_models.TextGenerationModel.predict_streaming"
    ) as mock_completion_stream:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_stream.return_value = (el for el in palm_completion_stream)
            nebuly.init(api_key="test", disable_checks=True)
            parameters = {
                "temperature": 0,
                # Temperature controls the degree of randomness in token selection.
                "max_output_tokens": 256,
                # Token limit determines the maximum amount of text output.
                "top_p": 0.8,
                # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
                "top_k": 40,
                # A top_k of 1 means the selected token is the most probable among all tokens.
            }

            model = TextGenerationModel.from_pretrained("text-bison@001")
            result = ""
            for chunk in model.predict_streaming(
                prompt="Give me two interview questions for the role of program manager.",
                **parameters,
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


def test_vertexai_chat__no_context_manager(palm_completion):
    with patch(
        "vertexai.language_models.ChatSession.send_message"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = palm_completion
            nebuly.init(api_key="test", disable_checks=True)

            parameters = {
                "temperature": 0,
                "max_output_tokens": 256,
                "top_p": 0.95,
                "top_k": 40,
            }
            chat_model = ChatModel.from_pretrained("chat-bison@001")
            chat = chat_model.start_chat(
                context="My name is Miles. You are an astronomer, knowledgeable about the solar system.",
                examples=[
                    InputOutputTextPair(
                        input_text="How many moons does Mars have?",
                        output_text="The planet Mars has two moons, Phobos and Deimos.",
                    ),
                ],
            )

            result = chat.send_message(
                "How many planets are there in the solar system?", **parameters
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


def test_vertexai_chat__no_context_manager__with_history(palm_completion):
    with patch(
        "vertexai.language_models.ChatSession.send_message"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = palm_completion
            nebuly.init(api_key="test", disable_checks=True)

            parameters = {
                "temperature": 0,
                "max_output_tokens": 256,
                "top_p": 0.95,
                "top_k": 40,
            }
            chat_model = ChatModel.from_pretrained("chat-bison@001")
            chat = chat_model.start_chat(
                context="My name is Miles. You are an astronomer, knowledgeable about the solar system.",
                examples=[
                    InputOutputTextPair(
                        input_text="How many moons does Mars have?",
                        output_text="The planet Mars has two moons, Phobos and Deimos.",
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
                "How many planets are there in the solar system?", **parameters
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
                ("user", "How many moons does Mars have?"),
                ("assistant", "The planet Mars has two moons, Phobos and Deimos."),
            ]
            assert interaction_watch.output == palm_completion.text
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


def test_vertexai_chat__with_context_manager(palm_completion):
    with patch(
        "vertexai.language_models.ChatSession.send_message"
    ) as mock_completion_create:
        mock_completion_create.return_value = palm_completion
        nebuly.init(api_key="test", disable_checks=True)

        with new_interaction(
            "How many planets are there in the solar system?"
        ) as interaction_watch:
            interaction_watch.set_input(
                "How many planets are there in the solar system?"
            )
            parameters = {
                "temperature": 0,
                "max_output_tokens": 256,
                "top_p": 0.95,
                "top_k": 40,
            }
            chat_model = ChatModel.from_pretrained("chat-bison@001")
            chat = chat_model.start_chat(
                context="My name is Miles. You are an astronomer, knowledgeable about the solar system.",
                examples=[
                    InputOutputTextPair(
                        input_text="How many moons does Mars have?",
                        output_text="The planet Mars has two moons, Phobos and Deimos.",
                    ),
                ],
            )

            result = chat.send_message(
                "How many planets are there in the solar system?", **parameters
            )
            interaction_watch.set_output(result.text)
        assert result is not None
        assert mock_completion_create.call_count == 1
        assert interaction_watch.output == palm_completion.text
        assert len(interaction_watch.spans) == 1
        span = interaction_watch.spans[0]
        assert isinstance(span, SpanWatch)


@pytest.mark.asyncio
async def test_vertexai_chat__async(palm_completion):
    with patch(
        "vertexai.language_models.ChatSession.send_message_async"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = palm_completion
            nebuly.init(api_key="test", disable_checks=True)

            parameters = {
                "temperature": 0,
                "max_output_tokens": 256,
                "top_p": 0.95,
                "top_k": 40,
            }

            chat_model = ChatModel.from_pretrained("chat-bison@001")
            chat = chat_model.start_chat(
                context="My name is Miles. You are an astronomer, knowledgeable about the solar system.",
                examples=[
                    InputOutputTextPair(
                        input_text="How many moons does Mars have?",
                        output_text="The planet Mars has two moons, Phobos and Deimos.",
                    ),
                ],
            )

            result = await chat.send_message_async(
                "How many planets are there in the solar system?", **parameters
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


def test_vertexai_chat__stream(palm_completion_stream):
    with patch(
        "vertexai.language_models.ChatSession.send_message_streaming"
    ) as mock_completion_stream:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_stream.return_value = (el for el in palm_completion_stream)
            nebuly.init(api_key="test", disable_checks=True)

            parameters = {
                "temperature": 0,
                "max_output_tokens": 256,
                "top_p": 0.95,
                "top_k": 40,
            }

            chat_model = ChatModel.from_pretrained("chat-bison@001")
            chat = chat_model.start_chat(
                context="My name is Miles. You are an astronomer, knowledgeable about the solar system.",
                examples=[
                    InputOutputTextPair(
                        input_text="How many moons does Mars have?",
                        output_text="The planet Mars has two moons, Phobos and Deimos.",
                    ),
                ],
            )

            result = ""
            for chunk in chat.send_message_streaming(
                message="How many planets are there in the solar system?", **parameters
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
