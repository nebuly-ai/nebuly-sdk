from unittest.mock import Mock, patch

import cohere
import pytest
from cohere.responses import Chat, Generation, Generations

import nebuly
from nebuly.contextmanager import new_interaction
from nebuly.entities import InteractionWatch, SpanWatch
from nebuly.observers import NebulyObserver


@pytest.fixture()
def cohere_generate() -> list[Generation]:
    return Generations.from_dict(
        {
            "id": "5dd1e0ae-ee97-42ac-91df-6ffe5eb4b498",
            "generations": [
                {
                    "id": "a3d047ef-27ba-4d28-b3d5-0e5aa2c2cf77",
                    "text": ' LLMs, or "AI language models", are a type of artificial intelligence that can understand and respond',
                }
            ],
            "prompt": "Please explain to me how LLMs work",
            "meta": {"api_version": {"version": "1"}},
        },
        return_likelihoods=None,
    )


def test_cohere_generate__no_context_manager(cohere_generate):
    with patch("cohere.Client.generate") as mock_generate:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_generate.return_value = cohere_generate
            nebuly.init(api_key="test")

            co = cohere.Client("test")
            result = co.generate(
                prompt="Please explain to me how LLMs work",
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Please explain to me how LLMs work"
            assert (
                interaction_watch.output
                == ' LLMs, or "AI language models", are a type of artificial intelligence that can understand and respond'
            )
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


def test_cohere_generate__with_context_manager(cohere_generate):
    with patch("cohere.Client.generate") as mock_generate:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_generate.return_value = cohere_generate
            nebuly.init(api_key="test")

            co = cohere.Client("test")
            with new_interaction(
                user="test_user", group_profile="test_group"
            ) as interaction:
                interaction.set_input("Sample input 1")
                result = co.generate(
                    prompt="Please explain to me how LLMs work",
                )
                interaction.set_output("Sample output 1")
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Sample input 1"
            assert interaction_watch.output == "Sample output 1"
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


@pytest.fixture()
def cohere_chat() -> Chat:
    return Chat.from_dict(
        {
            "response_id": "e7a48048-078e-4941-a54a-1e87efcb03b3",
            "text": "I'm doing well, thanks for asking!",
            "generation_id": "39f41e01-14bb-4823-8a8d-17f719e0b46c",
            "token_count": {
                "prompt_tokens": 90,
                "response_tokens": 396,
                "total_tokens": 486,
                "billed_tokens": 480,
            },
            "meta": {"api_version": {"version": "1"}},
            "search_queries": [],
        },
        message="How are you?",
        client=Mock(),
    )


def test_cohere_chat__no_context_manager(cohere_chat):
    with patch("cohere.Client.chat") as mock_chat:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_chat.return_value = cohere_chat
            nebuly.init(api_key="test")

            co = cohere.Client("test")
            result = co.chat(
                message="How are you?",
                chat_history=[
                    {"user_name": "User", "message": "Hi!"},
                    {"user_name": "Chatbot", "message": "How can I help you today?"},
                ],
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "How are you?"
            assert interaction_watch.history == [
                ("User", "Hi!"),
                ("Chatbot", "How can I help you today?"),
            ]
            assert interaction_watch.output == "I'm doing well, thanks for asking!"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


def test_cohere_chat__with_context_manager(cohere_chat):
    with patch("cohere.Client.chat") as mock_chat:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_chat.return_value = cohere_chat
            nebuly.init(api_key="test")

            co = cohere.Client("test")
            with new_interaction(
                user="test_user", group_profile="test_group"
            ) as interaction:
                interaction.set_input("Sample input 1")
                interaction.set_history(
                    [
                        ("User", "Hi!"),
                        ("Chatbot", "How can I help you today?"),
                    ]
                )
                result = co.chat(
                    message="How are you?",
                    chat_history=[
                        {"user_name": "User", "message": "Hi!"},
                        {
                            "user_name": "Chatbot",
                            "message": "How can I help you today?",
                        },
                    ],
                )
                interaction.set_output("Sample output 1")
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Sample input 1"
            assert interaction_watch.history == [
                ("User", "Hi!"),
                ("Chatbot", "How can I help you today?"),
            ]
            assert interaction_watch.output == "Sample output 1"
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
