from unittest.mock import patch

import google.generativeai as palm
import pytest
from google.generativeai.discuss import ChatResponse
from google.generativeai.text import Completion
from google.generativeai.types.discuss_types import MessageDict
from google.generativeai.types.safety_types import HarmCategory, HarmProbability

import nebuly
from nebuly.contextmanager import new_interaction
from nebuly.entities import InteractionWatch, SpanWatch
from nebuly.observers import NebulyObserver


@pytest.fixture()
def palm_completion() -> Completion:
    return Completion(
        candidates=[
            {
                "output": "cold.",
                "safety_ratings": [
                    {
                        "category": HarmCategory.HARM_CATEGORY_DEROGATORY,
                        "probability": HarmProbability.NEGLIGIBLE,
                    },
                    {
                        "category": HarmCategory.HARM_CATEGORY_TOXICITY,
                        "probability": HarmProbability.NEGLIGIBLE,
                    },
                    {
                        "category": HarmCategory.HARM_CATEGORY_VIOLENCE,
                        "probability": HarmProbability.NEGLIGIBLE,
                    },
                    {
                        "category": HarmCategory.HARM_CATEGORY_SEXUAL,
                        "probability": HarmProbability.NEGLIGIBLE,
                    },
                    {
                        "category": HarmCategory.HARM_CATEGORY_MEDICAL,
                        "probability": HarmProbability.NEGLIGIBLE,
                    },
                    {
                        "category": HarmCategory.HARM_CATEGORY_DANGEROUS,
                        "probability": HarmProbability.NEGLIGIBLE,
                    },
                ],
            }
        ],
        result="cold.",
        filters=[],
        safety_feedback=[],
    )


def test_google_palm_completion__no_context_manager(palm_completion):
    with patch("google.generativeai.generate_text") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = palm_completion
            nebuly.init(api_key="test", disable_checks=True)
            palm.configure(api_key="AIzaSyDuQlj_CeVZUFOj8oRm_dwotC4ucAElF_w")

            result = palm.generate_text(prompt="The opposite of hot is")
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "The opposite of hot is"
            assert interaction_watch.output == "cold."
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


def test_google_palm_completion__with_context_manager(palm_completion):
    with patch("google.generativeai.generate_text") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = palm_completion
            nebuly.init(api_key="test", disable_checks=True)
            palm.configure(api_key="AIzaSyDuQlj_CeVZUFOj8oRm_dwotC4ucAElF_w")

            with new_interaction(
                user="test_user", group_profile="test_group"
            ) as interaction:
                interaction.set_input("Another input")
                result = palm.generate_text(prompt="The opposite of hot is")
                interaction.set_output("Another output")

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch: InteractionWatch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Another input"
            assert interaction_watch.output == "Another output"
            assert interaction.user == "test_user"
            assert interaction.user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span: SpanWatch = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


@pytest.fixture()
def palm_chat_response() -> ChatResponse:
    return ChatResponse(
        model="models/chat-bison-001",
        context="",
        examples=[],
        messages=[
            MessageDict(**{"author": "0", "content": "Hello."}),
            MessageDict(
                **{"author": "1", "content": "Hello! How can I help you today?"}
            ),
        ],
        temperature=None,
        candidate_count=None,
        candidates=[
            MessageDict(
                **{"author": "1", "content": "Hello! How can I help you today?"}
            )
        ],
        filters=[],
        top_p=None,
        top_k=None,
        _client=None,
    )


def test_google_palm_chat__first_interaction__no_context_manager(palm_chat_response):
    with patch("google.generativeai.chat") as mock_chat:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_chat.return_value = palm_chat_response
            nebuly.init(api_key="test", disable_checks=True)
            palm.configure(api_key="test")

            result = palm.chat(messages=["Hello."])
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch: InteractionWatch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Hello."
            assert interaction_watch.output == "Hello! How can I help you today?"
            assert len(interaction_watch.spans) == 1
            span: SpanWatch = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


def test_google_palm_chat__first_interaction__with_context_manager(palm_chat_response):
    with patch("google.generativeai.chat") as mock_chat:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_chat.return_value = palm_chat_response
            nebuly.init(api_key="test", disable_checks=True)
            palm.configure(api_key="test")

            with new_interaction(
                user="test_user", group_profile="test_group"
            ) as interaction:
                interaction.set_input("Another input")
                result = palm.chat(messages=["Hello."])
                interaction.set_output("Another output")

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch: InteractionWatch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Another input"
            assert interaction_watch.output == "Another output"
            assert interaction.user == "test_user"
            assert interaction.user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span: SpanWatch = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


@pytest.fixture()
def palm_chat_response_with_history() -> ChatResponse:
    return ChatResponse(
        model="models/chat-bison-001",
        context="",
        examples=[],
        messages=[
            MessageDict(**{"author": "0", "content": "Hello."}),
            MessageDict(
                **{"author": "1", "content": "Hello! How can I help you today?"}
            ),
            MessageDict(
                **{
                    "author": "0",
                    "content": "What can you do?",
                }
            ),
            MessageDict(
                **{
                    "author": "1",
                    "content": "I am a large language model, also known as a "
                    "conversational AI or chatbot trained to be informative "
                    "and comprehensive. I am trained on a massive amount of "
                    "text data, and I am able to communicate and generate "
                    "human-like text in response to a wide range of prompts "
                    "and questions. For example, I can provide summaries of "
                    "factual topics or create stories.",
                }
            ),
        ],
        temperature=None,
        candidate_count=None,
        candidates=[
            MessageDict(
                **{
                    "author": "1",
                    "content": "I am a large language model, also known as a "
                    "conversational AI or chatbot trained to be informative "
                    "and comprehensive. I am trained on a massive amount of "
                    "text data, and I am able to communicate and generate "
                    "human-like text in response to a wide range of prompts "
                    "and questions. For example, I can provide summaries of "
                    "factual topics or create stories.",
                }
            )
        ],
        filters=[],
        top_p=None,
        top_k=None,
    )


def test_google_palm_chat__with_history__no_context_manager(
    palm_chat_response_with_history,
):
    with patch("google.generativeai.chat") as mock_chat:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_chat.return_value = palm_chat_response_with_history
            nebuly.init(api_key="test", disable_checks=True)
            palm.configure(api_key="test")

            result = palm.chat(
                messages=[
                    "Hello.",
                    "Hello! How can I help you today?",
                    "What can you do?",
                ]
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch: InteractionWatch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "What can you do?"
            assert interaction_watch.history == [
                ("user", "Hello."),
                ("assistant", "Hello! How can I help you today?"),
            ]
            assert (
                interaction_watch.output
                == palm_chat_response_with_history.messages[-1]["content"]
            )
            assert len(interaction_watch.spans) == 1
            span: SpanWatch = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


def test_google_palm_chat__with_history__with_context_manager(
    palm_chat_response_with_history,
):
    with patch("google.generativeai.chat") as mock_chat:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_chat.return_value = palm_chat_response
            nebuly.init(api_key="test", disable_checks=True)
            palm.configure(api_key="test")

            with new_interaction(
                user="test_user", group_profile="test_group"
            ) as interaction:
                interaction.set_input("Another input")
                result = palm.chat(
                    messages=[
                        "Hello.",
                        "Hello! How can I help you today?",
                        "What can you do?",
                    ]
                )
                interaction.set_output("Another output")

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch: InteractionWatch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Another input"
            assert interaction_watch.output == "Another output"
            assert interaction.user == "test_user"
            assert interaction.user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span: SpanWatch = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


def test_google_palm_chat__reply__no_context_manager(
    palm_chat_response, palm_chat_response_with_history
):
    with patch.object(ChatResponse, "reply") as mock_reply:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_reply.return_value = palm_chat_response_with_history
            nebuly.init(api_key="test", disable_checks=True)
            palm.configure(api_key="test")

            result = palm_chat_response.reply("What can you do?")

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch: InteractionWatch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "What can you do?"
            assert interaction_watch.history == [
                ("user", "Hello."),
                ("assistant", "Hello! How can I help you today?"),
            ]
            assert (
                interaction_watch.output
                == palm_chat_response_with_history.messages[-1]["content"]
            )
            assert len(interaction_watch.spans) == 1
            span: SpanWatch = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


def test_google_palm_chat__reply__with_context_manager(
    palm_chat_response, palm_chat_response_with_history
):
    with patch.object(ChatResponse, "reply") as mock_reply:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_reply.return_value = palm_chat_response_with_history
            nebuly.init(api_key="test", disable_checks=True)
            palm.configure(api_key="test")

            with new_interaction(
                user="test_user", group_profile="test_group"
            ) as interaction:
                interaction.set_input("Another input")
                result = palm_chat_response.reply("What can you do?")
                interaction.set_output("Another output")

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch: InteractionWatch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Another input"
            assert interaction_watch.output == "Another output"
            assert interaction.user == "test_user"
            assert interaction.user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span: SpanWatch = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


@pytest.mark.asyncio
async def test_google_palm_chat__async(palm_chat_response):
    with patch("google.generativeai.chat_async") as mock_chat:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_chat.return_value = palm_chat_response
            nebuly.init(api_key="test", disable_checks=True)
            palm.configure(api_key="test")

            result = await palm.chat_async(messages=["Hello."])
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch: InteractionWatch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Hello."
            assert interaction_watch.output == "Hello! How can I help you today?"
            assert len(interaction_watch.spans) == 1
            span: SpanWatch = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
